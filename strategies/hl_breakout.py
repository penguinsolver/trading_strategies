"""
Strategy: High Low Breakout

Simple breakout of session high/low with trend filter.
Clean, classic breakout strategy.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class HighLowBreakoutStrategy(Strategy):
    """High/Low breakout strategy."""
    
    @property
    def name(self) -> str:
        return "HL Breakout"
    
    @property
    def description(self) -> str:
        return "Trade breakouts of previous N-bar high/low with EMA trend filter."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="lookback",
                label="Lookback Bars",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=5,
                help_text="Bars for high/low",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate high/low breakout signals."""
        lookback = self.params.get("lookback", 10)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema50"] = ema(data["close"], 50)
        
        # Previous high/low
        data["prev_high"] = data["high"].rolling(lookback).max().shift(1)
        data["prev_low"] = data["low"].rolling(lookback).min().shift(1)
        
        # Trend filter
        above_ema = data["close"] > data["ema50"]
        below_ema = data["close"] < data["ema50"]
        
        # Breakout
        long_cond = (data["close"] > data["prev_high"]) & above_ema
        short_cond = (data["close"] < data["prev_low"]) & below_ema
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "prev_low"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "prev_high"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on EMA cross
        ema_cross_down = (data["close"] < data["ema50"]) & (data["close"].shift(1) >= data["ema50"].shift(1))
        ema_cross_up = (data["close"] > data["ema50"]) & (data["close"].shift(1) <= data["ema50"].shift(1))
        data.loc[(ema_cross_down | ema_cross_up).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 50", "column": "ema50", "color": "blue", "style": "solid"}]
