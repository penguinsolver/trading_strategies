"""
Strategy: Force Index Breakout

Alexander Elder's Force Index combining price and volume.
High Force Index readings indicate strong buying/selling pressure.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Calculate Force Index."""
    force = (close - close.shift(1)) * volume
    smoothed = ema(force, period)
    return smoothed


class ForceIndexStrategy(Strategy):
    """Force Index momentum strategy."""
    
    @property
    def name(self) -> str:
        return "Force Index"
    
    @property
    def description(self) -> str:
        return "Trade Force Index zero-line crosses for volume-confirmed momentum moves."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="Force Period",
                param_type="int",
                default=13,
                min_value=7,
                max_value=21,
                step=2,
                help_text="EMA smoothing period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Force Index signals."""
        period = self.params.get("period", 13)
        
        # Calculate Force Index
        data["force"] = force_index(data["close"], data["volume"], period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_force = data["force"].shift(1)
        
        # Long: Force crosses above zero
        long_cond = (data["force"] > 0) & (prev_force <= 0)
        
        # Short: Force crosses below zero
        short_cond = (data["force"] < 0) & (prev_force >= 0)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on opposite cross
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Force Index", "column": "force", "color": "purple", "style": "solid"},
        ]
