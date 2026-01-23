"""
Robustness Validation Dashboard Page
Displays results of the BEST strategy configuration across 5 historical periods.
Best Config: w200_s10_lb5_atr2.5_r30%_ml0.3
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def load_trade_logs():
    """Load trade logs from JSON file."""
    try:
        log_path = Path(__file__).parent.parent / "data" / "trade_logs.json"
        with open(log_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load trade logs: {e}")
        return None

def render_robustness_page():
    st.set_page_config(layout="wide", page_title="Strategy Robustness")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">🛡️ Strategy Robustness Validation</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">MLEnhancedBreakout + Compound | 5 Independent Periods | Full Trade Logs</p>
    </div>
    """, unsafe_allow_html=True)

    # Load trade logs
    trade_logs = load_trade_logs()
    
    if trade_logs:
        total_roi = [p['roi_pct'] for p in trade_logs]
        avg_roi = np.mean(total_roi)
        min_roi = min(total_roi)
        avg_sharpe = np.mean([p['sharpe'] for p in trade_logs])
        min_sharpe = min(p['sharpe'] for p in trade_logs)
        total_trades = sum(p['num_trades'] for p in trade_logs)
        
        all_sharpe_pass = all(p['sharpe'] >= 2.0 for p in trade_logs)
        
        if all_sharpe_pass:
            st.success(f"✅ **Sharpe > 2.0 achieved on ALL 5 periods!** Loaded {total_trades} trades.")
        else:
            st.warning(f"⚠️ Some periods have Sharpe < 2.0")
    else:
        avg_roi = 0
        min_roi = 0
        avg_sharpe = 0
        min_sharpe = 0
        total_trades = 0
    
    # Best Config Banner
    st.markdown("### 🏆 Best Configuration: `w200_s10_lb5_atr2.5_r30%_ml0.3`")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg ROI", f"+{avg_roi:,.0f}%")
    col2.metric("Min ROI", f"{min_roi:+,.0f}%")
    col3.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
    col4.metric("Min Sharpe", f"{min_sharpe:.2f}", delta="Target: >2.0")
    
    # Period Summary
    st.markdown("### 📅 Period-by-Period Performance")
    
    if trade_logs:
        period_data = []
        for p in trade_logs:
            sharpe_status = "✅" if p['sharpe'] >= 2.0 else "❌"
            period_data.append({
                "Period": p['period'],
                "Date Range": p['date_range'],
                "Start": f"${p['start_capital']:,.0f}",
                "End": f"${p['end_capital']:,.0f}",
                "ROI": f"{p['roi_pct']:+,.0f}%",
                "Sharpe": f"{p['sharpe']:.2f} {sharpe_status}",
                "Trades": p['num_trades'],
                "Win Rate": f"{p['win_rate']:.0f}%"
            })
        df_periods = pd.DataFrame(period_data)
        st.dataframe(df_periods, use_container_width=True, hide_index=True)
    
    # Replication Guide
    st.markdown("---")
    st.markdown("### 🛠️ Replication Guide")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Strategy Parameters")
        st.code("""
# MLEnhancedBreakout Config
window_size = 200     # hours
step_size = 10        # hours  
lookback = 5          # Donchian bars
atr_period = 14       # ATR Period
atr_multiplier = 2.5  # Stop Loss Width

# ML Config (EnhancedConfig)
min_ml_confidence = 0.30
forward_bars = 8
n_estimators = 100
        """, language="python")

    with c2:
        st.subheader("Risk & Execution")
        st.code("""
# Risk Settings (HARD CONSTRAINTS)
RISK_PER_TRADE = 0.30  # 30% capital allocation
MAX_LEVERAGE = 10      # 10x max leverage
TIMEFRAME = '1h'       # Hourly candles

# Position Sizing Formula:
# Notional = Capital × 0.30 × 10
# Size = Notional / Entry Price
        """, language="python")
    
    # Detailed Trade Logs
    st.markdown("---")
    st.markdown("### 📊 Detailed Trade Logs (Expandable)")
    
    if trade_logs:
        for p in trade_logs:
            with st.expander(f"📈 Period {p['period']}: {p['date_range']} | ROI: {p['roi_pct']:+,.0f}% | Sharpe: {p['sharpe']:.2f}"):
                trades_df = pd.DataFrame(p['trades'])
                
                if not trades_df.empty:
                    display_df = trades_df[[
                        'entry_time', 'exit_time', 'direction', 
                        'entry_price', 'exit_price', 'stop_loss',
                        'position_size', 'notional', 'leverage_used',
                        'price_change_pct', 'pnl_dollars', 'pnl_pct',
                        'capital_before', 'capital_after'
                    ]].copy()
                    
                    display_df.columns = [
                        'Entry', 'Exit', 'Dir',
                        'Entry $', 'Exit $', 'Stop $',
                        'Size', 'Notional', 'Lev',
                        'Price Δ%', 'P&L $', 'P&L %',
                        'Cap Before', 'Cap After'
                    ]
                    
                    # Format numbers
                    display_df['Entry $'] = display_df['Entry $'].apply(lambda x: f"${x:,.0f}")
                    display_df['Exit $'] = display_df['Exit $'].apply(lambda x: f"${x:,.0f}")
                    display_df['Stop $'] = display_df['Stop $'].apply(lambda x: f"${x:,.0f}")
                    display_df['Size'] = display_df['Size'].apply(lambda x: f"{x:.4f}")
                    display_df['Notional'] = display_df['Notional'].apply(lambda x: f"${x:,.0f}")
                    display_df['Lev'] = display_df['Lev'].apply(lambda x: f"{x:.1f}x")
                    display_df['Price Δ%'] = display_df['Price Δ%'].apply(lambda x: f"{x:+.2f}%")
                    display_df['P&L $'] = display_df['P&L $'].apply(lambda x: f"${x:+,.2f}")
                    display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:+.2f}%")
                    display_df['Cap Before'] = display_df['Cap Before'].apply(lambda x: f"${x:,.0f}")
                    display_df['Cap After'] = display_df['Cap After'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
                    
                    st.markdown(f"""
                    **Summary**: {p['num_trades']} trades | Win Rate: {p['win_rate']:.0f}% |
                    Start: ${p['start_capital']:,.0f} → End: ${p['end_capital']:,.0f} | 
                    **ROI: {p['roi_pct']:+,.2f}%** | **Sharpe: {p['sharpe']:.2f}**
                    """)
    
    # Equity Curve
    st.markdown("---")
    st.markdown("### 📈 Equity Curve (All Periods)")
    
    if trade_logs:
        fig = go.Figure()
        colors = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6']
        
        for i, p in enumerate(trade_logs):
            trades = p['trades']
            if trades:
                equity = [p['start_capital']] + [t['capital_after'] for t in trades]
                x_vals = list(range(len(equity)))
                
                fig.add_trace(go.Scatter(
                    x=x_vals, 
                    y=equity, 
                    name=f"Period {p['period']} ({p['roi_pct']:+,.0f}%)",
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="Portfolio Value Progression (Trade-by-Trade)",
            xaxis_title="Trade #",
            yaxis_title="Portfolio Value ($)",
            yaxis_type="log",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_robustness_page()
