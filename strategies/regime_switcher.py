"""
Strategy: Regime Switcher

Meta strategy that selects between sub-strategies based on market regime.
- Trend regime (high ADX): Uses trend-following strategy (Donchian Turtle or Supertrend)
- Range regime (low ADX): Uses mean-reversion strategy (RSI-2 Dip or VWAP Reversion)
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import adx, atr, ema


class RegimeSwitcherStrategy(Strategy):
    """Adaptive strategy that switches based on market regime."""
    
    @property
    def name(self) -> str:
        return "Regime Switcher"
    
    @property
    def description(self) -> str:
        return "Adaptive strategy selecting trend or mean-reversion based on ADX regime detection."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="adx_period",
                label="ADX Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ADX calculation period",
            ),
            ParamConfig(
                name="adx_trend_threshold",
                label="ADX Trend Threshold",
                param_type="int",
                default=25,
                min_value=20,
                max_value=35,
                step=5,
                help_text="ADX level above which market is considered trending",
            ),
            ParamConfig(
                name="adx_range_threshold",
                label="ADX Range Threshold",
                param_type="int",
                default=20,
                min_value=15,
                max_value=25,
                step=5,
                help_text="ADX level below which market is considered ranging",
            ),
            ParamConfig(
                name="trend_mode",
                label="Trend Strategy",
                param_type="select",
                default="supertrend",
                options=["supertrend", "donchian"],
                help_text="Strategy to use in trending regime",
            ),
            ParamConfig(
                name="range_mode",
                label="Range Strategy",
                param_type="select",
                default="rsi_dip",
                options=["rsi_dip", "vwap"],
                help_text="Strategy to use in ranging regime",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ATR period for various calculations",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on regime detection."""
        # Parameters
        adx_period = self.params.get("adx_period", 14)
        adx_trend_threshold = self.params.get("adx_trend_threshold", 25)
        adx_range_threshold = self.params.get("adx_range_threshold", 20)
        trend_mode = self.params.get("trend_mode", "supertrend")
        range_mode = self.params.get("range_mode", "rsi_dip")
        atr_period = self.params.get("atr_period", 14)
        
        # Calculate ADX for regime detection
        adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], adx_period)
        data["adx"] = adx_val
        data["plus_di"] = plus_di
        data["minus_di"] = minus_di
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Determine regime
        data["trend_regime"] = data["adx"] >= adx_trend_threshold
        data["range_regime"] = data["adx"] <= adx_range_threshold
        data["neutral_regime"] = ~data["trend_regime"] & ~data["range_regime"]
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Generate signals based on regime
        # Import and use sub-strategies inline to avoid circular imports
        
        # Trend regime signals
        if trend_mode == "supertrend":
            self._generate_supertrend_signals(data, atr_period)
        else:  # donchian
            self._generate_donchian_signals(data, atr_period)
        
        # Range regime signals
        if range_mode == "rsi_dip":
            self._generate_rsi_signals(data, atr_period)
        else:  # vwap
            self._generate_vwap_signals(data, atr_period)
        
        return data
    
    def _generate_supertrend_signals(self, data: pd.DataFrame, atr_period: int):
        """Generate Supertrend signals for trend regime."""
        from indicators import supertrend
        
        st_line, st_direction = supertrend(data["high"], data["low"], data["close"], atr_period, 3.0)
        data["supertrend"] = st_line
        data["st_direction"] = st_direction
        
        # Detect flips
        prev_dir = st_direction.shift(1)
        flip_long = (st_direction == 1) & (prev_dir == -1) & data["trend_regime"]
        flip_short = (st_direction == -1) & (prev_dir == 1) & data["trend_regime"]
        
        data.loc[flip_long, "entry_signal"] = 1
        data.loc[flip_long, "stop_price"] = data.loc[flip_long, "supertrend"]
        data.loc[flip_long, "trailing_stop_atr"] = data.loc[flip_long, "atr"] * 1.5
        
        data.loc[flip_short, "entry_signal"] = -1
        data.loc[flip_short, "stop_price"] = data.loc[flip_short, "supertrend"]
        data.loc[flip_short, "trailing_stop_atr"] = data.loc[flip_short, "atr"] * 1.5
    
    def _generate_donchian_signals(self, data: pd.DataFrame, atr_period: int):
        """Generate Donchian breakout signals for trend regime."""
        from indicators import donchian_channels
        
        dc_upper, dc_lower, dc_mid = donchian_channels(data["high"], data["low"], 20)
        data["dc_upper"] = dc_upper.shift(1)
        data["dc_lower"] = dc_lower.shift(1)
        
        # Breakout signals in trend regime
        breakout_long = (
            data["trend_regime"] &
            (data["close"] > data["dc_upper"]) &
            (data["close"].shift(1) <= data["dc_upper"].shift(1))
        )
        
        breakout_short = (
            data["trend_regime"] &
            (data["close"] < data["dc_lower"]) &
            (data["close"].shift(1) >= data["dc_lower"].shift(1))
        )
        
        data.loc[breakout_long, "entry_signal"] = 1
        data.loc[breakout_long, "stop_price"] = data.loc[breakout_long, "dc_lower"]
        data.loc[breakout_long, "trailing_stop_atr"] = data.loc[breakout_long, "atr"] * 2.0
        
        data.loc[breakout_short, "entry_signal"] = -1
        data.loc[breakout_short, "stop_price"] = data.loc[breakout_short, "dc_upper"]
        data.loc[breakout_short, "trailing_stop_atr"] = data.loc[breakout_short, "atr"] * 2.0
    
    def _generate_rsi_signals(self, data: pd.DataFrame, atr_period: int):
        """Generate RSI-2 signals for range regime."""
        from indicators import rsi
        
        data["rsi"] = rsi(data["close"], 2)
        data["ema_200"] = ema(data["close"], 200)
        
        # Uptrend + RSI oversold in range regime
        long_signal = (
            data["range_regime"] &
            (data["close"] > data["ema_200"]) &
            (data["rsi"] < 10)
        )
        
        # Only take longs for safety in range regime
        data.loc[long_signal, "entry_signal"] = 1
        data.loc[long_signal, "stop_price"] = data.loc[long_signal, "close"] - (data.loc[long_signal, "atr"] * 2)
        
        # Exit when RSI recovers
        data.loc[(data["rsi"] > 50) & (data["rsi"].shift(1) <= 50), "exit_signal"] = True
    
    def _generate_vwap_signals(self, data: pd.DataFrame, atr_period: int):
        """Generate VWAP reversion signals for range regime."""
        # Simple VWAP approximation using typical price cumulative
        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
        data["vwap"] = (data["typical_price"] * data["volume"]).cumsum() / data["volume"].cumsum()
        
        # Distance from VWAP
        data["vwap_distance"] = (data["close"] - data["vwap"]) / data["atr"]
        
        # Mean reversion: far from VWAP in range regime
        long_signal = data["range_regime"] & (data["vwap_distance"] < -2)
        short_signal = data["range_regime"] & (data["vwap_distance"] > 2)
        
        data.loc[long_signal, "entry_signal"] = 1
        data.loc[long_signal, "stop_price"] = data.loc[long_signal, "close"] - (data.loc[long_signal, "atr"] * 1.5)
        
        data.loc[short_signal, "entry_signal"] = -1
        data.loc[short_signal, "stop_price"] = data.loc[short_signal, "close"] + (data.loc[short_signal, "atr"] * 1.5)
        
        # Exit at VWAP
        data.loc[abs(data["vwap_distance"]) < 0.5, "exit_signal"] = True
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "ADX", "column": "adx", "color": "purple", "style": "solid"},
        ]
