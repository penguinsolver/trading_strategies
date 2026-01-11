"""
Strategy: Triple EMA Momentum

Triple EMA alignment for strong trend confirmation.
Only trade when fast>mid>slow (or reverse for short).
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


class TripleEMAStrategy(Strategy):
    """Triple EMA alignment strategy."""
    
    @property
    def name(self) -> str:
        return "Triple EMA"
    
    @property
    def description(self) -> str:
        return "Trade when EMAs are aligned (8>21>55 for longs) with momentum confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Triple EMA signals."""
        data["ema8"] = ema(data["close"], 8)
        data["ema21"] = ema(data["close"], 21)
        data["ema55"] = ema(data["close"], 55)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Alignment
        bullish_align = (data["ema8"] > data["ema21"]) & (data["ema21"] > data["ema55"])
        bearish_align = (data["ema8"] < data["ema21"]) & (data["ema21"] < data["ema55"])
        
        prev_bull = bullish_align.shift(1).fillna(False)
        prev_bear = bearish_align.shift(1).fillna(False)
        
        # Entry on new alignment
        long_cond = bullish_align & (~prev_bull)
        short_cond = bearish_align & (~prev_bear)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ema55"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ema55"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.0
        
        # Exit on loss of alignment
        data.loc[(~bullish_align & ~bearish_align).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 8", "column": "ema8", "color": "green", "style": "solid"},
            {"name": "EMA 21", "column": "ema21", "color": "blue", "style": "solid"},
            {"name": "EMA 55", "column": "ema55", "color": "red", "style": "solid"},
        ]
