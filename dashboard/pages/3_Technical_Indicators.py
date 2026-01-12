"""
Compare Technical Indicator Strategies - run all TA strategies and compare results.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from data import CandleFetcher
from backtest import BacktestEngine, CostModel
from strategies import STRATEGIES


def render_comparison_page():
    """Render the strategy comparison page."""
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">📊 Technical Indicators</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Compare 90 rule-based technical indicator strategies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        window = st.selectbox(
            "📅 Time Window",
            options=["24h", "7d", "14d", "30d", "90d", "180d"],
            index=1,
            help="24h, 7d, 14d, 30d (1 month), 90d (3 months), 180d (6 months)",
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Candle Interval",
            options=["5m", "15m", "1h", "4h"],
            index=0,
        )
    
    with col3:
        capital = st.number_input(
            "💰 Initial Capital ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
        )
    
    with col4:
        risk = st.slider(
            "⚡ Risk per Trade (%)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
        ) / 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run comparison button
    if st.button("🚀 Compare All Strategies", type="primary", use_container_width=True):
        run_comparison(window, interval, capital, risk)
    
    # Show previous results
    if "comparison_results" in st.session_state and st.session_state.comparison_results:
        display_comparison_results(st.session_state.comparison_results)


def run_comparison(window: str, interval: str, capital: float, risk: float):
    """Run all strategies and store results."""
    
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
    progress = st.progress(0, text="Running strategies...")
    
    strategy_names = list(STRATEGIES.keys())
    for i, strategy_key in enumerate(strategy_names):
        progress.progress((i + 1) / len(strategy_names), text=f"Running {strategy_key.replace('_', ' ').title()}...")
        
        try:
            strategy_class = STRATEGIES[strategy_key]
            strategy = strategy_class()
            
            engine = BacktestEngine(cost_model=cost_model)
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
    st.session_state.comparison_results = results
    st.session_state.comparison_settings = {
        "window": window,
        "interval": interval,
        "capital": capital,
    }
    st.rerun()


def display_comparison_results(results: dict):
    """Display comparison results using Streamlit native components."""
    
    settings = st.session_state.get("comparison_settings", {})
    
    st.header("📊 Results Overview")
    st.caption(f"Window: **{settings.get('window', 'N/A')}** • Interval: **{settings.get('interval', 'N/A')}** • Capital: **${settings.get('capital', 0):,}**")
    
    strategy_display = {
        "trend_pullback": ("📈", "Trend Pullback"),
        "breakout": ("🚀", "Breakout"),
        "vwap_reversion": ("🔄", "VWAP Reversion"),
        "ma_crossover": ("📊", "MA Crossover"),
        "supertrend": ("⚡", "Supertrend"),
        "donchian_turtle": ("🐢", "Donchian Turtle"),
        "rsi2_dip": ("📉", "RSI-2 Dip"),
        "bb_squeeze": ("🎯", "BB Squeeze"),
        "inside_bar": ("📦", "Inside Bar"),
        "orb": ("🌅", "Opening Range"),
        "breakout_retest": ("🔁", "Breakout Retest"),
        "regime_switcher": ("🔀", "Regime Switcher"),
        # Selective strategies
        "atr_channel": ("📡", "ATR Channel"),
        "volume_breakout": ("📢", "Volume Breakout"),
        "zscore_reversion": ("📐", "Z-Score Revert"),
        "chandelier_trend": ("💎", "Chandelier"),
        "avwap_pullback": ("⚓", "AVWAP Pullback"),
        "regression_slope": ("📉", "Regression Slope"),
        # Anti-chop strategies
        "bb_mean_reversion": ("🔙", "BB Mean Revert"),
        "prev_day_range": ("📅", "Prev Day Range"),
        "ts_momentum": ("📊", "TS Momentum"),
    }
    
    # Strategy cards using Streamlit columns - 4 per row
    num_strategies = len(results)
    num_cols = 4
    items = list(results.items())
    
    for row_start in range(0, num_strategies, num_cols):
        cols = st.columns(num_cols)
        for col_idx, (key, result) in enumerate(items[row_start:row_start + num_cols]):
            icon, name = strategy_display.get(key, ("📊", key))
            
            # Get strategy description
            strategy_class = STRATEGIES.get(key)
            desc = ""
            if strategy_class:
                try:
                    desc = strategy_class().description or ""
                except:
                    desc = ""
            # Truncate description for display
            short_desc = desc[:60] + "..." if len(desc) > 60 else desc
            
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
                    
                    # Card container with description tooltip
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border: 3px solid {border_color};
                        border-radius: 16px;
                        padding: 1rem;
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        margin-bottom: 1rem;
                    " title="{desc}">
                        <div style="font-size: 1.3rem; margin-bottom: 0.15rem;">{icon}</div>
                        <div style="font-weight: 600; font-size: 0.85rem; color: #1e293b; margin-bottom: 0.3rem;">{name}</div>
                        <div style="font-size: 0.65rem; color: #94a3b8; margin-bottom: 0.5rem; min-height: 2rem; line-height: 1.3;">{short_desc}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {border_color};">{m.net_return:+.2f}%</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;">${m.net_return_dollars:+,.0f}</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; text-align: left; border-top: 1px solid #e2e8f0; padding-top: 0.5rem; font-size: 0.7rem;">
                            <div><span style="color: #94a3b8;">Win</span><br><strong>{m.win_rate:.0f}%</strong></div>
                            <div><span style="color: #94a3b8;">Trades</span><br><strong>{m.total_trades}</strong></div>
                            <div><span style="color: #94a3b8;">DD</span><br><strong style="color: #ef4444;">{m.max_drawdown:.1f}%</strong></div>
                            <div><span style="color: #94a3b8;">PF</span><br><strong>{m.profit_factor:.2f}</strong></div>
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
            st.success(f"🏆 **Best Performer:** {icon} {name} with **{best_result.metrics.net_return:+.2f}%** return")
        else:
            st.warning(f"📉 **Least Loss:** {icon} {name} with **{best_result.metrics.net_return:+.2f}%** return")
    
    st.divider()
    
    # Detailed metrics table
    st.header("📋 Detailed Metrics")
    
    table_data = []
    for key, result in results.items():
        if result is None:
            continue
        
        icon, name = strategy_display.get(key, ("📊", key))
        m = result.metrics
        
        pf = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "∞"
        avg_cost = m.total_costs / m.total_trades if m.total_trades > 0 else 0
        
        table_data.append({
            "Strategy": f"{icon} {name}",
            "Gross PnL": f"${m.pnl_before_costs:+,.2f}",
            "Net PnL": f"${m.pnl_after_costs:+,.2f}",
            "Return (%)": f"{m.net_return:+.2f}%",
            "Win Rate": f"{m.win_rate:.1f}%",
            "Trades": m.total_trades,
            "T/Day": f"{m.trades_per_day:.1f}",
            "PF": pf,
            "Max DD": f"{m.max_drawdown:.1f}%",
            "Fees": f"${m.total_fees:.2f}",
            "Slip": f"${m.total_slippage:.2f}",
            "Avg Cost": f"${avg_cost:.2f}",
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Equity curves
    st.header("📈 Equity Curves")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    colors = {
        "trend_pullback": "#22c55e",
        "breakout": "#3b82f6",
        "vwap_reversion": "#f59e0b",
        "ma_crossover": "#8b5cf6",
        "supertrend": "#ec4899",
        "donchian_turtle": "#06b6d4",
        "rsi2_dip": "#84cc16",
        "bb_squeeze": "#f97316",
        "inside_bar": "#6366f1",
        "orb": "#14b8a6",
        "breakout_retest": "#a855f7",
        "regime_switcher": "#ef4444",
        # Selective strategies
        "atr_channel": "#0ea5e9",
        "volume_breakout": "#d946ef",
        "zscore_reversion": "#10b981",
        "chandelier_trend": "#fbbf24",
        "avwap_pullback": "#4f46e5",
        "regression_slope": "#7c3aed",
        # Anti-chop strategies
        "bb_mean_reversion": "#f43f5e",
        "prev_day_range": "#06b6d4",
        "ts_momentum": "#8b5cf6",
    }
    
    for key, result in valid_results.items():
        icon, name = strategy_display.get(key, ("📊", key))
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name=name,
            line=dict(color=colors.get(key, "#888"), width=2.5),
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


# Run the page
render_comparison_page()
