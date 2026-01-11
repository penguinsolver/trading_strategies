"""
Strategy: EMA Slope Momentum

Trade when EMA slope is strong (steep), indicating momentum.
Only enter when trend is accelerating.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


class EMASlopeMomentumStrategy(Strategy):
    """EMA slope momentum strategy."""
    
    @property
    def name(self) -> str:
        return "EMA Slope"
    
    @property
    def description(self) -> str:
        return "Trade when EMA slope is steep, indicating strong momentum."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="ema_period",
                label="EMA Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="EMA period",
            ),
            ParamConfig(
                name="slope_threshold",
                label="Slope Threshold",
                param_type="float",
                default=0.3,
                min_value=0.1,
                max_value=0.5,
                step=0.1,
                help_text="Min slope % per bar",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA slope signals."""
        ema_period = self.params.get("ema_period", 20)
        slope_threshold = self.params.get("slope_threshold", 0.3)
        
        data["ema20"] = ema(data["close"], ema_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Calculate slope as % change
        data["ema_slope"] = (data["ema20"] - data["ema20"].shift(3)) / data["ema20"].shift(3) * 100
        
        prev_slope = data["ema_slope"].shift(1)
        
        # Entry when slope exceeds threshold and price confirms
        long_cond = (
            (data["ema_slope"] > slope_threshold) &
            (prev_slope <= slope_threshold) &
            (data["close"] > data["ema20"])
        )
        
        short_cond = (
            (data["ema_slope"] < -slope_threshold) &
            (prev_slope >= -slope_threshold) &
            (data["close"] < data["ema20"])
        )
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ema20"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ema20"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit when slope flattens
        slope_flat = (data["ema_slope"].abs() < slope_threshold / 2)
        data.loc[slope_flat.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA", "column": "ema20", "color": "blue", "style": "solid"},
        ]
