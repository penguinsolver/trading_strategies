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
    
    # Strategy selection - all 90 strategies organized by batch
    strategy_options = {
        # Original strategies (4)
        "trend_pullback": "📈 Trend Pullback",
        "breakout": "🚀 Breakout",
        "vwap_reversion": "🔄 VWAP Reversion",
        "ma_crossover": "📊 MA Crossover",
        # Batch 1 - Diverse (8)
        "supertrend": "⚡ Supertrend",
        "donchian_turtle": "🐢 Donchian Turtle",
        "rsi2_dip": "📉 RSI-2 Dip",
        "bb_squeeze": "🎯 BB Squeeze",
        "inside_bar": "📦 Inside Bar",
        "orb": "🌅 Opening Range",
        "breakout_retest": "🔁 Breakout Retest",
        "regime_switcher": "🔀 Regime Switcher",
        # Batch 2 - Selective (6)
        "atr_channel": "📡 ATR Channel",
        "volume_breakout": "📢 Volume Breakout",
        "zscore_reversion": "📐 Z-Score Reversion",
        "chandelier_trend": "💎 Chandelier Trend",
        "avwap_pullback": "⚓ AVWAP Pullback",
        "regression_slope": "📉 Regression Slope",
        # Batch 3 - Anti-chop (3)
        "bb_mean_reversion": "🔙 BB Mean Revert",
        "prev_day_range": "📅 Prev Day Range",
        "ts_momentum": "📊 TS Momentum",
        # Batch 4 - Classic Indicators (9)
        "keltner_breakout": "📊 Keltner Breakout",
        "macd_divergence": "📉 MACD Divergence",
        "parabolic_sar": "🎯 Parabolic SAR",
        "stochastic_momentum": "📈 Stochastic",
        "williams_r": "📉 Williams %R",
        "cci_momentum": "📊 CCI Momentum",
        "ichimoku_cloud": "☁️ Ichimoku Cloud",
        "elder_ray": "👁️ Elder Ray",
        "obv_divergence": "📊 OBV Divergence",
        # Batch 5-6 - Pivot & Volume (6)
        "pivot_point": "📍 Pivot Point",
        "trix_momentum": "📈 TRIX Momentum",
        "aroon_trend": "🌙 Aroon Trend",
        "force_index": "💪 Force Index",
        "mfi_reversal": "💰 MFI Reversal",
        "ad_line": "📈 A/D Line",
        # Batch 7-8 - Momentum (6)
        "ultimate_oscillator": "🎯 Ultimate Osc",
        "dmi_cross": "↔️ DMI Cross",
        "roc_momentum": "📈 ROC Momentum",
        "hull_ma": "🚀 Hull MA",
        "vortex": "🌀 Vortex",
        "chaikin_oscillator": "📊 Chaikin Osc",
        # Batch 9-10 - Final Classic (6)
        "kst": "📈 KST",
        "coppock": "📉 Coppock Curve",
        "ppo": "📊 PPO",
        "macd_zero": "📈 MACD Zero",
        "rsi_divergence": "📉 RSI Divergence",
        "smi": "📊 SMI",
        # Batch 11-12 - Optimized (12)
        "rsi_extreme": "🎯 RSI Extreme",
        "tight_ema_scalp": "⚡ Tight EMA Scalp",
        "range_breakout": "📊 Range Breakout",
        "ema_slope_momentum": "📈 EMA Slope",
        "price_action": "🕯️ Price Action",
        "momentum_burst": "💥 Momentum Burst",
        "triple_ema": "📈 Triple EMA",
        "candle_combo": "🕯️ Candle Combo",
        "vwap_bounce": "🔄 VWAP Bounce",
        "hl_breakout": "📊 HL Breakout",
        "rsi_bb_revert": "🔙 RSI+BB Revert",
        "quick_scalp": "⚡ Quick Scalp",
        # Batch 13-14 - Trend (12)
        "atr_trend_rider": "🚀 ATR Trend Rider",
        "dual_tf_momentum": "📊 Dual TF Momentum",
        "vol_contraction": "📉 Vol Contraction",
        "c2c_momentum": "📈 C2C Momentum",
        "gap_fill": "📊 Gap Fill",
        "range_revert": "🔙 Range Revert",
        "strong_trend": "💪 Strong Trend",
        "pullback_ema": "🔙 Pullback EMA",
        "vol_weighted_trend": "📊 Vol Weighted",
        "inside_bar_bo": "📦 Inside Bar BO",
        "rsi_trending": "📈 RSI Trending",
        "close_breakout": "🚀 Close Breakout",
        # Batch 15-17 - Final (18)
        "quick_rsi_scalp": "⚡ Quick RSI Scalp",
        "vol_spike": "📈 Vol Spike",
        "ema_ribbon": "🎀 EMA Ribbon",
        "bounce_low": "⬆️ Bounce Low",
        "mom_continue": "📈 Mom Continue",
        "simple_pa": "🕯️ Simple PA",
        "fast_trend_scalp": "⚡ Fast Trend Scalp",
        "aggressive_bo": "🚀 Aggressive BO",
        "micro_trend": "📈 Micro Trend",
        "quick_reversal": "🔄 Quick Reversal",
        "trend_simple": "📊 Trend Simple",
        "doji_reversal": "🕯️ Doji Reversal",
        "bar_count": "📊 Bar Count",
        "opening_move": "🌅 Opening Move",
        "fade_extreme": "🔙 Fade Extreme",
        "tight_range_break": "📊 Tight Range",
        "mom_filter": "📈 Mom Filter",
        "final_ema": "📊 Final EMA",
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
    
    window = st.sidebar.selectbox(
        "Time Period",
        options=["24h", "7d", "14d", "30d", "90d", "180d"],
        index=1,
        help="24h, 7d, 14d, 30d (1 month), 90d (3 months), 180d (6 months)",
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
