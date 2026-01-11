"""Metrics panel component."""
import streamlit as st

from backtest import BacktestResult, PerformanceMetrics


def render_metrics_panel(result: BacktestResult) -> None:
    """
    Render the metrics panel with key performance indicators.
    
    Displays:
    - Primary metrics row (return, drawdown, win rate, profit factor)
    - Secondary metrics row (avg R, expectancy, trades/day, hold time)
    - Cost breakdown section
    """
    metrics = result.metrics
    
    # Primary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if metrics.net_return >= 0 else "inverse"
        st.metric(
            "Net Return",
            f"{metrics.net_return:.2f}%",
            delta=f"${metrics.net_return_dollars:,.2f}",
            delta_color=delta_color,
        )
    
    with col2:
        st.metric(
            "Max Drawdown",
            f"{metrics.max_drawdown:.2f}%",
            delta=f"${metrics.max_drawdown_dollars:,.2f}",
            delta_color="inverse",
        )
    
    with col3:
        st.metric(
            "Win Rate",
            f"{metrics.win_rate:.1f}%",
            delta=f"{metrics.winning_trades}W / {metrics.losing_trades}L",
        )
    
    with col4:
        pf_display = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "∞"
        st.metric(
            "Profit Factor",
            pf_display,
            help="Gross Profit / Gross Loss",
        )
    
    # Secondary metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Avg R-Multiple",
            f"{metrics.avg_r_multiple:.2f}R",
            help="Average risk-adjusted return per trade",
        )
    
    with col6:
        st.metric(
            "Expectancy",
            f"{metrics.expectancy:.3f}%",
            help="Average profit per trade as % of capital",
        )
    
    with col7:
        st.metric(
            "Trades/Day",
            f"{metrics.trades_per_day:.1f}",
        )
    
    with col8:
        st.metric(
            "Avg Hold Time",
            f"{metrics.avg_hold_time_hours:.1f}h",
        )
    
    # Win/Loss details in expander
    with st.expander("📊 Trade Details", expanded=False):
        det_col1, det_col2 = st.columns(2)
        
        with det_col1:
            st.markdown("**Winners**")
            st.write(f"- Count: {metrics.winning_trades}")
            st.write(f"- Average: ${metrics.avg_win:,.2f}")
            st.write(f"- Largest: ${metrics.largest_win:,.2f}")
        
        with det_col2:
            st.markdown("**Losers**")
            st.write(f"- Count: {metrics.losing_trades}")
            st.write(f"- Average: ${metrics.avg_loss:,.2f}")
            st.write(f"- Largest: ${metrics.largest_loss:,.2f}")
    
    # Cost breakdown
    st.divider()
    st.subheader("💰 Cost Breakdown")
    
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
    
    with cost_col1:
        st.metric(
            "Total Fees",
            f"${metrics.total_fees:,.2f}",
        )
    
    with cost_col2:
        st.metric(
            "Total Funding",
            f"${metrics.total_funding:,.2f}",
        )
    
    with cost_col3:
        st.metric(
            "Total Slippage",
            f"${metrics.total_slippage:,.2f}",
        )
    
    with cost_col4:
        st.metric(
            "Total Costs",
            f"${metrics.total_costs:,.2f}",
        )
    
    # P&L comparison
    pnl_col1, pnl_col2 = st.columns(2)
    
    with pnl_col1:
        st.metric(
            "P&L Before Costs",
            f"${metrics.pnl_before_costs:,.2f}",
        )
    
    with pnl_col2:
        st.metric(
            "P&L After Costs",
            f"${metrics.pnl_after_costs:,.2f}",
            delta=f"-${metrics.total_costs:,.2f}",
            delta_color="inverse",
        )
