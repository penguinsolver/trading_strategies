"""
Strategy: Close Breakout

Simple but effective: trade breakouts of previous close.
Clean signal generation.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class CloseBreakoutStrategy(Strategy):
    """Close breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Close Breakout"
    
    @property
    def description(self) -> str:
        return "Trade when close breaks above/below previous N-bar closing range."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="lookback",
                label="Lookback",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=5,
                help_text="Bars for close high/low",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate close breakout signals."""
        lookback = self.params.get("lookback", 10)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema20"] = ema(data["close"], 20)
        
        # Close high/low
        data["close_high"] = data["close"].rolling(lookback).max().shift(1)
        data["close_low"] = data["close"].rolling(lookback).min().shift(1)
        
        # Breakout
        breakout_up = data["close"] > data["close_high"]
        breakout_down = data["close"] < data["close_low"]
        
        # Trend filter
        above_ema = data["close"] > data["ema20"]
        below_ema = data["close"] < data["ema20"]
        
        long_cond = breakout_up & above_ema
        short_cond = breakout_down & below_ema
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close_low"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close_high"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on break back into range
        in_range = (data["close"] < data["close_high"]) & (data["close"] > data["close_low"])
        data.loc[in_range.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 20", "column": "ema20", "color": "blue", "style": "solid"}]
