"""
Diagnostics component for strategy debugging.

Shows "Why No Trades?" analysis and entry condition breakdown.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest import BacktestResult


def render_diagnostics(result: "BacktestResult", show_all: bool = False):
    """
    Render diagnostics panel for strategy debugging.
    
    Shows which entry conditions are passing/failing most often.
    """
    if result is None:
        st.warning("No result to diagnose")
        return
    
    data = result.data
    
    st.subheader("🔍 Strategy Diagnostics")
    
    # Trade count summary
    total_trades = result.metrics.total_trades
    
    if total_trades == 0:
        st.error("⚠️ **No trades executed**")
        st.markdown("Analyzing why entry conditions never triggered...")
    else:
        st.info(f"✅ **{total_trades} trades executed** ({result.metrics.trades_per_day:.1f}/day)")
    
    # Find diagnostic columns
    diag_cols = [c for c in data.columns if c.startswith("_diag_")]
    
    if diag_cols:
        st.markdown("### Entry Condition Analysis")
        
        # Create condition pass/fail table
        condition_stats = []
        for col in diag_cols:
            name = col.replace("_diag_", "").replace("_", " ").title()
            series = data[col]
            
            if series.dtype == bool:
                true_count = series.sum()
                total = len(series.dropna())
                pass_rate = (true_count / total * 100) if total > 0 else 0
                
                condition_stats.append({
                    "Condition": name,
                    "Pass Count": true_count,
                    "Total Bars": total,
                    "Pass Rate": f"{pass_rate:.1f}%",
                    "Status": "✅" if pass_rate > 5 else "⚠️" if pass_rate > 0 else "❌"
                })
        
        if condition_stats:
            df = pd.DataFrame(condition_stats)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Identify most restrictive filter
            most_restrictive = min(condition_stats, key=lambda x: float(x["Pass Rate"].rstrip("%")))
            st.warning(f"**Most restrictive filter**: {most_restrictive['Condition']} ({most_restrictive['Pass Rate']} pass rate)")
    
    # Show indicator distributions
    if show_all:
        st.markdown("### Indicator Distributions")
        
        indicator_cols = ["adx", "choppiness", "atr", "zscore", "sma_slope", "ema_slope", "atr_slope"]
        available = [c for c in indicator_cols if c in data.columns]
        
        if available:
            stats = []
            for col in available:
                series = data[col].dropna()
                if len(series) > 0:
                    stats.append({
                        "Indicator": col.upper(),
                        "Min": f"{series.min():.2f}",
                        "Max": f"{series.max():.2f}",
                        "Mean": f"{series.mean():.2f}",
                        "Std": f"{series.std():.2f}",
                        "Current": f"{series.iloc[-1]:.2f}",
                    })
            
            if stats:
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
    
    # Entry signal summary
    if "entry_signal" in data.columns:
        entry_signals = data["entry_signal"]
        long_signals = (entry_signals == 1).sum()
        short_signals = (entry_signals == -1).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Long Signals", long_signals)
        with col2:
            st.metric("Short Signals", short_signals)
        with col3:
            st.metric("Total Signals", long_signals + short_signals)


def render_cost_breakdown(result: "BacktestResult"):
    """Render detailed cost breakdown."""
    if result is None:
        return
    
    m = result.metrics
    
    st.subheader("💰 Cost Breakdown")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Gross PnL", f"${m.pnl_before_costs:,.2f}")
    
    with cols[1]:
        st.metric("Net PnL", f"${m.pnl_after_costs:,.2f}", 
                  delta=f"-${m.total_costs:,.2f}" if m.total_costs > 0 else None,
                  delta_color="inverse")
    
    with cols[2]:
        avg_cost = m.total_costs / m.total_trades if m.total_trades > 0 else 0
        st.metric("Avg Cost/Trade", f"${avg_cost:,.2f}")
    
    with cols[3]:
        st.metric("Trades/Day", f"{m.trades_per_day:.1f}")
    
    # Detailed cost table
    if m.total_trades > 0:
        cost_data = {
            "Cost Type": ["Fees", "Slippage", "Funding", "Total"],
            "Amount": [
                f"${m.total_fees:,.2f}",
                f"${m.total_slippage:,.2f}",
                f"${m.total_funding:,.2f}",
                f"${m.total_costs:,.2f}",
            ],
            "Per Trade": [
                f"${m.total_fees/m.total_trades:,.2f}",
                f"${m.total_slippage/m.total_trades:,.2f}",
                f"${m.total_funding/m.total_trades:,.2f}",
                f"${m.total_costs/m.total_trades:,.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)


def get_diagnostic_summary(result: "BacktestResult") -> dict:
    """Get summary diagnostics for comparison table."""
    if result is None:
        return {}
    
    m = result.metrics
    avg_cost = m.total_costs / m.total_trades if m.total_trades > 0 else 0
    
    return {
        "gross_pnl": m.pnl_before_costs,
        "net_pnl": m.pnl_after_costs,
        "fees": m.total_fees,
        "slippage": m.total_slippage,
        "avg_cost_per_trade": avg_cost,
        "trades_per_day": m.trades_per_day,
    }
