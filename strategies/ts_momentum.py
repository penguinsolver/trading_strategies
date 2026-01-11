"""
Strategy: Time-Series Momentum (Higher Timeframe)

Higher timeframe momentum strategy using SMA and slope confirmation.
Designed for 1h/4h signals to reduce noise and overtrading.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import sma, ema, atr


class TSMomentumStrategy(Strategy):
    """Time-series momentum on higher timeframe."""
    
    @property
    def name(self) -> str:
        return "TS Momentum"
    
    @property
    def description(self) -> str:
        return "Higher TF momentum: long when close > SMA(50) and slope positive. Low frequency."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="sma_period",
                label="SMA Period",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=10,
                help_text="SMA period for trend",
            ),
            ParamConfig(
                name="slope_lookback",
                label="Slope Lookback",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=5,
                help_text="Bars to measure slope over",
            ),
            ParamConfig(
                name="slope_threshold",
                label="Slope Threshold",
                param_type="float",
                default=0.0,
                min_value=-0.5,
                max_value=1.0,
                step=0.1,
                help_text="Min slope for trend (0 = any positive)",
            ),
            ParamConfig(
                name="atr_trail_mult",
                label="ATR Trail Mult",
                param_type="float",
                default=2.0,
                min_value=1.0,
                max_value=4.0,
                step=0.5,
                help_text="ATR multiplier for trailing stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series momentum signals."""
        # Parameters
        sma_period = self.params.get("sma_period", 50)
        slope_lookback = self.params.get("slope_lookback", 10)
        slope_threshold = self.params.get("slope_threshold", 0.0)
        atr_trail_mult = self.params.get("atr_trail_mult", 2.0)
        
        # Calculate SMA
        data["sma"] = sma(data["close"], sma_period)
        
        # Calculate SMA slope (change per bar, normalized)
        data["sma_slope"] = (
            (data["sma"] - data["sma"].shift(slope_lookback)) / 
            data["sma"].shift(slope_lookback) * 100
        )
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Trend conditions (use shifted to avoid lookahead)
        prev_close = data["close"].shift(1)
        prev_sma = data["sma"].shift(1)
        prev_slope = data["sma_slope"].shift(1)
        
        # Long condition: close > SMA AND slope > threshold
        long_condition = (
            (data["close"] > data["sma"]) &
            (prev_close <= prev_sma) &  # Just crossed above
            (data["sma_slope"] > slope_threshold)
        )
        
        # Short condition: close < SMA AND slope < -threshold
        short_condition = (
            (data["close"] < data["sma"]) &
            (prev_close >= prev_sma) &  # Just crossed below
            (data["sma_slope"] < -slope_threshold)
        )
        
        # Exit conditions: cross back
        exit_long = (data["close"] < data["sma"]) & (prev_close >= prev_sma)
        exit_short = (data["close"] > data["sma"]) & (prev_close <= prev_sma)
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_condition.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - (
            data.loc[long_mask, "atr"] * atr_trail_mult
        )
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_trail_mult
        
        # Short entries
        short_mask = short_condition.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + (
            data.loc[short_mask, "atr"] * atr_trail_mult
        )
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_trail_mult
        
        # Exit signals
        exit_mask = (exit_long | exit_short).fillna(False)
        data.loc[exit_mask, "exit_signal"] = True
        
        # Store diagnostics
        data["_diag_above_sma"] = data["close"] > data["sma"]
        data["_diag_slope_positive"] = data["sma_slope"] > slope_threshold
        data["_diag_slope_negative"] = data["sma_slope"] < -slope_threshold
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "SMA", "column": "sma", "color": "blue", "style": "solid"},
        ]
