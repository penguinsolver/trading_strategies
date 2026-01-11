"""
Strategy: Volume Weighted Trend

Weight trend signals by volume for confirmation.
Higher volume = more reliable signal.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema, sma


class VolumeWeightedTrendStrategy(Strategy):
    """Volume weighted trend strategy."""
    
    @property
    def name(self) -> str:
        return "Vol Weighted Trend"
    
    @property
    def description(self) -> str:
        return "Trade EMA crosses with above-average volume confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume weighted signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema20"] = ema(data["close"], 20)
        data["vol_avg"] = sma(data["volume"], 20)
        
        # EMA cross
        above_ema = data["close"] > data["ema20"]
        below_ema = data["close"] < data["ema20"]
        prev_above = above_ema.shift(1).fillna(False)
        prev_below = below_ema.shift(1).fillna(False)
        
        cross_up = above_ema & (~prev_above)
        cross_down = below_ema & (~prev_below)
        
        # Volume confirmation
        high_vol = data["volume"] > data["vol_avg"] * 1.5
        
        long_cond = cross_up & high_vol
        short_cond = cross_down & high_vol
        
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
        
        # Exit on opposite cross
        data.loc[cross_down.fillna(False), "exit_signal"] = True
        data.loc[cross_up.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 20", "column": "ema20", "color": "blue", "style": "solid"}]
