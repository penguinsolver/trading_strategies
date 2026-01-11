"""
Strategy: Range Reversion

Trade mean reversion within defined ranges.
Only trade when price is ranging, not trending.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, adx


class RangeReversionStrategy(Strategy):
    """Range reversion strategy."""
    
    @property
    def name(self) -> str:
        return "Range Revert"
    
    @property
    def description(self) -> str:
        return "Mean reversion within ranges, only when ADX shows no trend."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="adx_max",
                label="Max ADX",
                param_type="int",
                default=25,
                min_value=15,
                max_value=35,
                step=5,
                help_text="Maximum ADX for ranging",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate range reversion signals."""
        adx_max = self.params.get("adx_max", 25)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        adx_val, _, _ = adx(data["high"], data["low"], data["close"], 14)
        data["adx"] = adx_val
        
        # Range boundaries
        data["range_high"] = data["high"].rolling(20).max()
        data["range_low"] = data["low"].rolling(20).min()
        data["range_mid"] = (data["range_high"] + data["range_low"]) / 2
        
        # Ranging condition
        is_ranging = data["adx"] < adx_max
        
        # At extremes
        at_low = data["close"] < data["range_low"] + (data["range_mid"] - data["range_low"]) * 0.25
        at_high = data["close"] > data["range_high"] - (data["range_high"] - data["range_mid"]) * 0.25
        
        long_cond = is_ranging & at_low & (data["close"] > data["open"])  # Bullish at support
        short_cond = is_ranging & at_high & (data["close"] < data["open"])  # Bearish at resistance
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "range_low"] - data.loc[long_mask, "atr"] * 0.25
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "range_high"] + data.loc[short_mask, "atr"] * 0.25
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit at mid
        data.loc[(data["close"] > data["range_mid"]).fillna(False), "exit_signal"] = True
        data.loc[(data["close"] < data["range_mid"]).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Range High", "column": "range_high", "color": "red", "style": "dashed"},
            {"name": "Range Low", "column": "range_low", "color": "green", "style": "dashed"},
        ]
