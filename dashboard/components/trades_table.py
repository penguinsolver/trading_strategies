"""Trades table component with sorting and filtering."""
import streamlit as st
import pandas as pd

from backtest import BacktestResult


def render_trades_table(result: BacktestResult) -> None:
    """
    Render an interactive trades table with:
    - All trade information
    - Sorting capability
    - Filtering options
    - Export buttons
    """
    if not result.trades:
        st.info("No trades were executed during this backtest.")
        return
    
    trades_df = result.get_trades_df()
    
    # Create header with export buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📋 Trade Ledger ({len(trades_df)} trades)")
    
    with col2:
        csv = trades_df.to_csv(index=False)
        st.download_button(
            "📥 Export CSV",
            csv,
            file_name=f"trades_{result.strategy_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
    
    with col3:
        json_data = trades_df.to_json(orient="records", date_format="iso")
        st.download_button(
            "📥 Export JSON",
            json_data,
            file_name=f"trades_{result.strategy_name.lower().replace(' ', '_')}.json",
            mime="application/json",
        )
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        side_filter = st.multiselect(
            "Filter by Side",
            options=["long", "short"],
            default=["long", "short"],
        )
    
    with filter_col2:
        pnl_filter = st.radio(
            "Filter by P&L",
            options=["All", "Winners", "Losers"],
            horizontal=True,
        )
    
    with filter_col3:
        exit_filter = st.multiselect(
            "Filter by Exit Reason",
            options=trades_df["exit_reason"].unique().tolist(),
            default=trades_df["exit_reason"].unique().tolist(),
        )
    
    # Apply filters
    filtered_df = trades_df.copy()
    
    filtered_df = filtered_df[filtered_df["side"].isin(side_filter)]
    
    if pnl_filter == "Winners":
        filtered_df = filtered_df[filtered_df["pnl_net"] > 0]
    elif pnl_filter == "Losers":
        filtered_df = filtered_df[filtered_df["pnl_net"] <= 0]
    
    filtered_df = filtered_df[filtered_df["exit_reason"].isin(exit_filter)]
    
    # Format for display
    display_df = filtered_df.copy()
    
    # Format columns
    display_df["entry_time"] = pd.to_datetime(display_df["entry_time"]).dt.strftime("%m/%d %H:%M")
    display_df["exit_time"] = pd.to_datetime(display_df["exit_time"]).dt.strftime("%m/%d %H:%M")
    display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:,.2f}")
    display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:,.2f}")
    display_df["stop_price"] = display_df["stop_price"].apply(lambda x: f"${x:,.2f}")
    display_df["pnl_gross"] = display_df["pnl_gross"].apply(lambda x: f"${x:,.2f}")
    display_df["pnl_net"] = display_df["pnl_net"].apply(lambda x: f"${x:,.2f}")
    display_df["r_multiple"] = display_df["r_multiple"].apply(lambda x: f"{x:.2f}R")
    display_df["fees"] = display_df["fees"].apply(lambda x: f"${x:.2f}")
    display_df["duration_hours"] = display_df["duration_hours"].apply(lambda x: f"{x:.1f}h")
    
    # Select and rename columns for display
    display_columns = [
        "entry_time", "exit_time", "side", "entry_price", "exit_price",
        "pnl_net", "r_multiple", "exit_reason", "duration_hours", "fees"
    ]
    
    column_names = {
        "entry_time": "Entry",
        "exit_time": "Exit",
        "side": "Side",
        "entry_price": "Entry $",
        "exit_price": "Exit $",
        "pnl_net": "P&L",
        "r_multiple": "R-Mult",
        "exit_reason": "Reason",
        "duration_hours": "Duration",
        "fees": "Fees",
    }
    
    display_df = display_df[display_columns].rename(columns=column_names)
    
    # Color coding with conditional styling
    def highlight_pnl(val):
        if "$" in str(val):
            # Extract number
            num = float(str(val).replace("$", "").replace(",", ""))
            if num > 0:
                return "background-color: rgba(38, 166, 154, 0.3)"
            elif num < 0:
                return "background-color: rgba(239, 83, 80, 0.3)"
        return ""
    
    # Show table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
    )
    
    # Summary stats for filtered trades
    if len(filtered_df) > 0:
        st.caption(
            f"Showing {len(filtered_df)} of {len(trades_df)} trades | "
            f"Total P&L: ${filtered_df['pnl_net'].apply(lambda x: float(str(x).replace('$', '').replace(',', '')) if isinstance(x, str) else x).sum():,.2f}"
        )
