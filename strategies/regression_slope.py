"""
Strategy: Regression Slope Trend

Trend strategy based on linear regression slope.
Entry: Only when slope magnitude exceeds threshold (strong trend)
Exit: Trailing stop based on ATR
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, linear_regression_slope, ema


class RegressionSlopeStrategy(Strategy):
    """Linear regression slope trend strategy."""
    
    @property
    def name(self) -> str:
        return "Regression Slope"
    
    @property
    def description(self) -> str:
        return "Trend strategy using linear regression slope. Only trades when slope magnitude confirms strong trend."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="slope_lookback",
                label="Slope Lookback",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=10,
                help_text="Lookback period for regression",
            ),
            ParamConfig(
                name="slope_threshold",
                label="Slope Threshold",
                param_type="float",
                default=0.5,
                min_value=0.2,
                max_value=1.0,
                step=0.1,
                help_text="Minimum slope magnitude for trend confirmation",
            ),
            ParamConfig(
                name="pullback_ema",
                label="Pullback EMA",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="EMA for pullback detection",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiplier for stop/trail",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regression slope trend signals."""
        # Parameters
        slope_lookback = self.params.get("slope_lookback", 50)
        slope_threshold = self.params.get("slope_threshold", 0.5)
        pullback_ema = self.params.get("pullback_ema", 20)
        atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        
        # Calculate regression slope
        data["slope"] = linear_regression_slope(data["close"], slope_lookback)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Calculate pullback EMA
        data["pb_ema"] = ema(data["close"], pullback_ema)
        
        # Trend regime: slope magnitude exceeds threshold
        data["strong_uptrend"] = data["slope"] > slope_threshold
        data["strong_downtrend"] = data["slope"] < -slope_threshold
        
        # Pullback detection: price dips to EMA in trend
        # For uptrend: price touches EMA from above and closes back above
        data["pullback_long"] = (
            data["strong_uptrend"] &
            (data["low"] <= data["pb_ema"]) &
            (data["close"] > data["pb_ema"])
        )
        
        # For downtrend: price rallies to EMA from below and closes back below
        data["pullback_short"] = (
            data["strong_downtrend"] &
            (data["high"] >= data["pb_ema"]) &
            (data["close"] < data["pb_ema"])
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["pullback_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - (
            data.loc[long_mask, "atr"] * 0.5  # Tight stop below pullback low
        )
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries
        short_mask = data["pullback_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + (
            data.loc[short_mask, "atr"] * 0.5  # Tight stop above pullback high
        )
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        # Exit when slope weakens significantly
        data["slope_weakening"] = (
            (data["slope"].abs() < slope_threshold / 2) &
            (data["slope"].shift(1).abs() >= slope_threshold / 2)
        )
        data.loc[data["slope_weakening"], "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Pullback EMA", "column": "pb_ema", "color": "blue", "style": "solid"},
        ]
