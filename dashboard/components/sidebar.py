"""Sidebar component with strategy and parameter controls."""
import streamlit as st

from strategies import STRATEGIES
from strategies.base import ParamConfig


def render_sidebar() -> dict:
    """
    Render the sidebar with strategy selection and parameter controls.
    
    Returns:
        Dictionary with:
        - strategy_key: Selected strategy key
        - strategy_params: Strategy parameters
        - window: Selected time window
        - capital: Initial capital
        - risk_per_trade: Risk per trade percentage
    """
    st.sidebar.header("🎯 Strategy Settings")
    
    # Strategy selection - all 21 strategies
    strategy_options = {
        # Original strategies
        "trend_pullback": "📈 Trend Pullback",
        "breakout": "🚀 Breakout",
        "vwap_reversion": "🔄 VWAP Reversion",
        "ma_crossover": "📊 MA Crossover",
        # Batch 1 strategies
        "supertrend": "⚡ Supertrend",
        "donchian_turtle": "🐢 Donchian Turtle",
        "rsi2_dip": "📉 RSI-2 Dip",
        "bb_squeeze": "🎯 BB Squeeze",
        "inside_bar": "📦 Inside Bar",
        "orb": "🌅 Opening Range",
        "breakout_retest": "🔁 Breakout Retest",
        "regime_switcher": "🔀 Regime Switcher",
        # Batch 2 - Selective strategies
        "atr_channel": "📡 ATR Channel",
        "volume_breakout": "📢 Volume Breakout",
        "zscore_reversion": "📐 Z-Score Reversion",
        "chandelier_trend": "💎 Chandelier Trend",
        "avwap_pullback": "⚓ AVWAP Pullback",
        "regression_slope": "📉 Regression Slope",
        # Batch 3 - Anti-chop strategies
        "bb_mean_reversion": "🔙 BB Mean Revert",
        "prev_day_range": "📅 Prev Day Range",
        "ts_momentum": "📊 TS Momentum",
    }
    
    strategy_key = st.sidebar.selectbox(
        "Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        help="Select trading strategy to backtest",
    )
    
    # Strategy description
    strategy_class = STRATEGIES[strategy_key]
    strategy_instance = strategy_class()
    if strategy_instance.description:
        st.sidebar.caption(strategy_instance.description)
    
    st.sidebar.divider()
    
    # Time window selection
    st.sidebar.header("⏱️ Backtest Window")
    
    window = st.sidebar.radio(
        "Time Period",
        options=["24h", "7d"],
        horizontal=True,
        help="Historical data window for backtesting",
    )
    
    # Data interval selection - now includes 4h
    interval = st.sidebar.selectbox(
        "Candle Interval",
        options=["5m", "15m", "1h", "4h"],
        index=0,
        help="Candle interval for backtesting (smaller = more data points)",
    )
    
    st.sidebar.divider()
    
    # Capital and risk settings
    st.sidebar.header("💰 Position Sizing")
    
    capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Starting capital for backtest",
    )
    
    risk_per_trade = st.sidebar.slider(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Percentage of capital risked per trade",
    ) / 100
    
    st.sidebar.divider()
    
    # Strategy-specific parameters
    st.sidebar.header("⚙️ Strategy Parameters")
    
    param_configs = strategy_instance.get_param_config()
    strategy_params = {}
    
    for param in param_configs:
        strategy_params[param.name] = render_param_input(param)
    
    st.sidebar.divider()
    
    # Cost settings
    with st.sidebar.expander("📊 Cost Settings", expanded=False):
        maker_fee = st.number_input(
            "Maker Fee (%)",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
        ) / 100
        
        taker_fee = st.number_input(
            "Taker Fee (%)",
            min_value=0.0,
            max_value=0.2,
            value=0.035,
            step=0.001,
            format="%.3f",
        ) / 100
        
        slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )
    
    return {
        "strategy_key": strategy_key,
        "strategy_params": strategy_params,
        "window": window,
        "interval": interval,
        "capital": capital,
        "risk_per_trade": risk_per_trade,
        "maker_fee": maker_fee,
        "taker_fee": taker_fee,
        "slippage_bps": slippage_bps,
    }


def render_param_input(param: ParamConfig):
    """Render appropriate input widget for a parameter."""
    
    if param.param_type == "int":
        return st.sidebar.slider(
            param.label,
            min_value=int(param.min_value or 1),
            max_value=int(param.max_value or 100),
            value=int(param.default),
            step=int(param.step or 1),
            help=param.help_text,
        )
    
    elif param.param_type == "float":
        return st.sidebar.slider(
            param.label,
            min_value=float(param.min_value or 0.0),
            max_value=float(param.max_value or 10.0),
            value=float(param.default),
            step=float(param.step or 0.1),
            help=param.help_text,
        )
    
    elif param.param_type == "bool":
        return st.sidebar.checkbox(
            param.label,
            value=param.default,
            help=param.help_text,
        )
    
    elif param.param_type == "select":
        return st.sidebar.selectbox(
            param.label,
            options=param.options,
            index=param.options.index(param.default) if param.default in param.options else 0,
            help=param.help_text,
        )
    
    return param.default
