"""
Strategy: EMA Ribbon Trend

Trade when EMA ribbon is fully aligned.
8, 13, 21, 34 EMA all in order.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class EMARibbonTrendStrategy(Strategy):
    """EMA ribbon strategy."""
    
    @property
    def name(self) -> str:
        return "EMA Ribbon"
    
    @property
    def description(self) -> str:
        return "Trade when EMA ribbon (8,13,21,34) is fully aligned."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA ribbon signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema8"] = ema(data["close"], 8)
        data["ema13"] = ema(data["close"], 13)
        data["ema21"] = ema(data["close"], 21)
        data["ema34"] = ema(data["close"], 34)
        
        # Full bullish alignment
        bull_ribbon = (
            (data["ema8"] > data["ema13"]) &
            (data["ema13"] > data["ema21"]) &
            (data["ema21"] > data["ema34"])
        )
        
        # Full bearish alignment
        bear_ribbon = (
            (data["ema8"] < data["ema13"]) &
            (data["ema13"] < data["ema21"]) &
            (data["ema21"] < data["ema34"])
        )
        
        prev_bull = bull_ribbon.shift(1).fillna(False)
        prev_bear = bear_ribbon.shift(1).fillna(False)
        
        # Entry on new alignment
        long_cond = bull_ribbon & (~prev_bull)
        short_cond = bear_ribbon & (~prev_bear)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ema34"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ema34"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.0
        
        # Exit on ribbon break
        data.loc[(~bull_ribbon & ~bear_ribbon).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 8", "column": "ema8", "color": "green", "style": "solid"},
            {"name": "EMA 34", "column": "ema34", "color": "red", "style": "solid"},
        ]
