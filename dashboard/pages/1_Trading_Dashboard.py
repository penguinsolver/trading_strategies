"""
Trading Dashboard page - the main backtesting interface.
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
from dashboard.components import (
    render_sidebar,
    render_price_chart,
    render_equity_chart,
    render_metrics_panel,
    render_trades_table,
)


def render_trading_dashboard():
    """Render the main trading dashboard."""
    
    # Header
    st.title("📈 BTC Active Trading Lab")
    st.caption("Backtest and compare trading strategies on Hyperliquid BTC perpetual")
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Run button in main area
    run_col1, run_col2 = st.columns([1, 5])
    with run_col1:
        run_backtest = st.button("▶️ Run Backtest", type="primary", use_container_width=True)
    
    with run_col2:
        st.info(
            f"Strategy: **{settings['strategy_key'].replace('_', ' ').title()}** | "
            f"Window: **{settings['window']}** | "
            f"Interval: **{settings['interval']}** | "
            f"Capital: **${settings['capital']:,}** | "
            f"Risk: **{settings['risk_per_trade']*100:.1f}%**"
        )
    
    st.divider()
    
    # Initialize session state
    if "result" not in st.session_state:
        st.session_state.result = None
    
    # Run backtest
    if run_backtest:
        with st.spinner("Fetching data from Hyperliquid..."):
            try:
                # Fetch data
                fetcher = CandleFetcher(coin="BTC", use_cache=True)
                data = fetcher.fetch_data(
                    interval=settings["interval"],
                    window=settings["window"],
                )
                
                if data.empty:
                    st.error("No data received from Hyperliquid API. Please try again.")
                    return
                
                st.success(f"Fetched {len(data)} candles for analysis")
                
            except Exception as e:
                st.error(f"Failed to fetch data: {str(e)}")
                return
        
        with st.spinner("Running backtest..."):
            try:
                # Create strategy
                strategy_class = STRATEGIES[settings["strategy_key"]]
                strategy = strategy_class(**settings["strategy_params"])
                
                # Create cost model
                cost_model = CostModel(
                    maker_fee=settings["maker_fee"],
                    taker_fee=settings["taker_fee"],
                    slippage_bps=settings["slippage_bps"],
                )
                
                # Run backtest
                engine = BacktestEngine(cost_model=cost_model)
                result = engine.run(
                    strategy=strategy,
                    data=data,
                    initial_capital=settings["capital"],
                    risk_per_trade=settings["risk_per_trade"],
                )
                
                st.session_state.result = result
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results
    if st.session_state.result is not None:
        result = st.session_state.result
        
        # Metrics panel
        render_metrics_panel(result)
        
        st.divider()
        
        # Charts in tabs
        chart_tab, equity_tab = st.tabs(["📊 Price Chart", "📈 Equity Curve"])
        
        with chart_tab:
            price_fig = render_price_chart(result)
            st.plotly_chart(price_fig, use_container_width=True)
        
        with equity_tab:
            equity_fig = render_equity_chart(result)
            st.plotly_chart(equity_fig, use_container_width=True)
        
        st.divider()
        
        # Trades table
        render_trades_table(result)
        
    else:
        # Welcome message when no backtest has been run
        st.markdown("""
        ### 👋 Ready to backtest!
        
        Select a strategy from the sidebar, adjust parameters if needed, and click **Run Backtest** to see results.
        
        **Need help?** Check the **📚 Strategy Guide** page for explanations and recommended parameters.
        """)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Trading Dashboard - BTC Trading Lab",
        page_icon="📈",
        layout="wide",
    )
    render_trading_dashboard()
