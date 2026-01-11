"""
Compare All Strategies with Exit Engine - standardized stops and trailing.
A copy of the Compare Strategies page but runs all models through the ExitEngine.
"""
import sys
from pathlib import Path
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from data import CandleFetcher
from backtest import ExitEngineBacktester, ExitEngineConfig, CostModel
from strategies import STRATEGIES
from dashboard.components.exit_engine_settings import render_exit_engine_settings, display_config_summary


def render_exit_engine_page():
    """Render the Exit Engine comparison page."""
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(249, 115, 22, 0.3);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">🎯 Models (Stops & Trailing)</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Same models with standardized Exit Engine: hard stops + trailing stops</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Exit Engine Settings
    config = render_exit_engine_settings()
    
    # Settings row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        window = st.selectbox(
            "📅 Time Window",
            options=["24h", "7d", "14d", "30d", "90d", "180d"],
            index=1,
            help="24h, 7d, 14d, 30d (1 month), 90d (3 months), 180d (6 months)",
            key="ee_window",
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Candle Interval",
            options=["5m", "15m", "1h", "4h"],
            index=0,
            key="ee_interval",
        )
    
    with col3:
        capital = st.number_input(
            "💰 Initial Capital ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            key="ee_capital",
        )
    
    with col4:
        risk = st.slider(
            "⚡ Risk per Trade (%)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            key="ee_risk",
        ) / 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display config summary
    display_config_summary(config)
    
    # Run comparison button
    if st.button("🚀 Compare All with Exit Engine", type="primary", use_container_width=True):
        run_exit_engine_comparison(window, interval, capital, risk, config)
    
    # Export buttons
    if "ee_comparison_results" in st.session_state and st.session_state.ee_comparison_results:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Summary CSV", use_container_width=True):
                export_summary_csv()
        with col2:
            if st.button("📥 Export All Trades CSV", use_container_width=True):
                export_trades_csv()
    
    # Show results
    if "ee_comparison_results" in st.session_state and st.session_state.ee_comparison_results:
        display_exit_engine_results(st.session_state.ee_comparison_results)


def run_exit_engine_comparison(window: str, interval: str, capital: float, risk: float, config: ExitEngineConfig):
    """Run all strategies through the Exit Engine."""
    
    with st.spinner("📡 Fetching data from Hyperliquid..."):
        try:
            fetcher = CandleFetcher(coin="BTC", use_cache=True)
            data = fetcher.fetch_data(interval=interval, window=window)
            
            if data.empty:
                st.error("No data received. Please try again.")
                return
            
            st.success(f"✅ Fetched {len(data)} candles for analysis")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return
    
    cost_model = CostModel(
        maker_fee=0.0001,
        taker_fee=0.00035,
        slippage_bps=1.0,
    )
    
    results = {}
    progress = st.progress(0, text="Running strategies with Exit Engine...")
    
    strategy_names = list(STRATEGIES.keys())
    for i, strategy_key in enumerate(strategy_names):
        progress.progress((i + 1) / len(strategy_names), text=f"Running {strategy_key.replace('_', ' ').title()}...")
        
        try:
            strategy_class = STRATEGIES[strategy_key]
            strategy = strategy_class()
            
            engine = ExitEngineBacktester(config=config, cost_model=cost_model)
            result = engine.run(
                strategy=strategy,
                data=data.copy(),
                initial_capital=capital,
                risk_per_trade=risk,
            )
            
            results[strategy_key] = result
        except Exception as e:
            st.warning(f"⚠️ {strategy_key} failed: {e}")
            results[strategy_key] = None
    
    progress.empty()
    st.session_state.ee_comparison_results = results
    st.session_state.ee_comparison_settings = {
        "window": window,
        "interval": interval,
        "capital": capital,
        "config": config,
    }
    st.rerun()


def export_summary_csv():
    """Export summary metrics to CSV."""
    results = st.session_state.get("ee_comparison_results", {})
    if not results:
        return
    
    summary_data = []
    for key, result in results.items():
        if result is not None:
            summary_data.append(result.get_summary_dict())
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary",
            data=csv,
            file_name="exit_engine_summary.csv",
            mime="text/csv",
        )


def export_trades_csv():
    """Export all trades to CSV."""
    results = st.session_state.get("ee_comparison_results", {})
    if not results:
        return
    
    all_trades = []
    for key, result in results.items():
        if result is not None:
            trades_df = result.get_trades_df()
            if not trades_df.empty:
                trades_df["model"] = key
                all_trades.append(trades_df)
    
    if all_trades:
        df = pd.concat(all_trades, ignore_index=True)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades",
            data=csv,
            file_name="exit_engine_trades.csv",
            mime="text/csv",
        )


def display_exit_engine_results(results: dict):
    """Display Exit Engine comparison results."""
    
    settings = st.session_state.get("ee_comparison_settings", {})
    config = settings.get("config", ExitEngineConfig())
    
    st.header("📊 Results Overview")
    st.caption(f"Window: **{settings.get('window', 'N/A')}** • Interval: **{settings.get('interval', 'N/A')}** • Capital: **${settings.get('capital', 0):,}**")
    
    # Strategy display names (reuse from original page or generate)
    strategy_display = {key: ("📊", key.replace("_", " ").title()) for key in STRATEGIES.keys()}
    strategy_display.update({
        "trend_pullback": ("📈", "Trend Pullback"),
        "breakout": ("🚀", "Breakout"),
        "vwap_reversion": ("🔄", "VWAP Reversion"),
        "ma_crossover": ("📊", "MA Crossover"),
        "supertrend": ("⚡", "Supertrend"),
        "donchian_turtle": ("🐢", "Donchian Turtle"),
        "rsi2_dip": ("📉", "RSI-2 Dip"),
        "bb_squeeze": ("🎯", "BB Squeeze"),
    })
    
    # Strategy cards - 4 per row
    num_strategies = len(results)
    num_cols = 4
    items = list(results.items())
    
    for row_start in range(0, num_strategies, num_cols):  # Show ALL strategies as cards
        cols = st.columns(num_cols)
        for col_idx, (key, result) in enumerate(items[row_start:row_start + num_cols]):
            icon, name = strategy_display.get(key, ("📊", key))
            
            with cols[col_idx]:
                if result is None:
                    st.error(f"**{icon} {name}**\n\nFailed to run")
                else:
                    m = result.metrics
                    
                    # Determine border color
                    if m.net_return > 0:
                        border_color = "#22c55e"
                    elif m.net_return > -5:
                        border_color = "#f59e0b"
                    else:
                        border_color = "#ef4444"
                    
                    # Card container
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border: 3px solid {border_color};
                        border-radius: 16px;
                        padding: 1rem;
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        margin-bottom: 1rem;
                    ">
                        <div style="font-size: 1.3rem; margin-bottom: 0.15rem;">{icon}</div>
                        <div style="font-weight: 600; font-size: 0.85rem; color: #1e293b; margin-bottom: 0.3rem;">{name}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {border_color};">{m.net_return:+.2f}%</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;">${m.net_return_dollars:+,.0f}</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; text-align: left; border-top: 1px solid #e2e8f0; padding-top: 0.5rem; font-size: 0.7rem;">
                            <div><span style="color: #94a3b8;">Win</span><br><strong>{m.win_rate:.0f}%</strong></div>
                            <div><span style="color: #94a3b8;">Trades</span><br><strong>{m.total_trades}</strong></div>
                            <div><span style="color: #94a3b8;">DD</span><br><strong style="color: #ef4444;">{m.max_drawdown:.1f}%</strong></div>
                            <div><span style="color: #94a3b8;">Avg R</span><br><strong>{result.avg_r_multiple:.2f}</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Winner/Loser announcement
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_key = max(valid_results.keys(), key=lambda k: valid_results[k].metrics.net_return)
        best_result = valid_results[best_key]
        icon, name = strategy_display.get(best_key, ("📊", best_key))
        
        if best_result.metrics.net_return > 0:
            st.success(f"🏆 **Best Performer:** {icon} {name} with **{best_result.metrics.net_return:+.2f}%** return (Avg R: {best_result.avg_r_multiple:.2f})")
        else:
            st.warning(f"📉 **Least Loss:** {icon} {name} with **{best_result.metrics.net_return:+.2f}%** return")
    
    st.divider()
    
    # Detailed metrics table with Exit Engine columns
    st.header("📋 Detailed Metrics")
    
    table_data = []
    for key, result in results.items():
        if result is None:
            continue
        
        icon, name = strategy_display.get(key, ("📊", key))
        m = result.metrics
        
        pf = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "∞"
        cost_per_trade = (result.total_fees + result.total_slippage) / max(1, m.total_trades)
        
        table_data.append({
            "Strategy": f"{icon} {name}",
            "Net Return": f"{m.net_return:+.2f}%",
            "Max DD": f"{m.max_drawdown:.1f}%",
            "PF": pf,
            "Avg R": f"{result.avg_r_multiple:.2f}",
            "Win Rate": f"{m.win_rate:.1f}%",
            "Trades": m.total_trades,
            "Avg Bars": f"{result.avg_hold_bars:.1f}",
            "Gross PnL": f"${result.gross_pnl:+,.2f}",
            "Net PnL": f"${m.pnl_after_costs:+,.2f}",
            "Total Fees": f"${result.total_fees:.2f}",
            "Total Slip": f"${result.total_slippage:.2f}",
            "Cost/Trade": f"${cost_per_trade:.2f}",
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        # Sort by Net Return descending
        df["_sort"] = [float(x.replace("%", "").replace("+", "")) for x in df["Net Return"]]
        df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Equity curves
    st.header("📈 Equity Curves")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Color palette
    color_palette = [
        "#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899",
        "#06b6d4", "#84cc16", "#f97316", "#6366f1", "#14b8a6",
        "#a855f7", "#ef4444", "#0ea5e9", "#d946ef", "#10b981",
    ]
    
    for i, (key, result) in enumerate(valid_results.items()):
        icon, name = strategy_display.get(key, ("📊", key))
        color = color_palette[i % len(color_palette)]
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name=name,
            line=dict(color=color, width=2),
        ))
    
    if valid_results:
        first_result = list(valid_results.values())[0]
        fig.add_hline(
            y=first_result.initial_capital,
            line_dash="dash",
            line_color="#94a3b8",
            opacity=0.7,
            annotation_text="Initial Capital",
        )
    
    fig.update_layout(
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis_title="",
        yaxis_title="Equity ($)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Trade distribution by exit reason
    st.header("🎯 Exit Reason Distribution")
    
    exit_reasons = {}
    for key, result in valid_results.items():
        if result is None:
            continue
        for trade in result.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1
    
    if exit_reasons:
        reason_df = pd.DataFrame([
            {"Exit Reason": k, "Count": v} 
            for k, v in sorted(exit_reasons.items(), key=lambda x: -x[1])
        ])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(reason_df, use_container_width=True, hide_index=True)
        with col2:
            import plotly.express as px
            fig = px.pie(reason_df, values="Count", names="Exit Reason", hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# Run the page
render_exit_engine_page()
