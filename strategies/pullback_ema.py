"""
Strategy: Pullback to EMA

Trade pullbacks to EMA in established trends.
Classic trend continuation pattern.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class PullbackToEMAStrategy(Strategy):
    """Pullback to EMA strategy."""
    
    @property
    def name(self) -> str:
        return "Pullback EMA"
    
    @property
    def description(self) -> str:
        return "Trade pullbacks to 21 EMA in established trends defined by 50 EMA."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pullback signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema21"] = ema(data["close"], 21)
        data["ema50"] = ema(data["close"], 50)
        
        # Trend: EMA21 > EMA50 = uptrend
        uptrend = data["ema21"] > data["ema50"]
        downtrend = data["ema21"] < data["ema50"]
        
        # Touch EMA21 (within 0.3 ATR)
        touch_zone = data["atr"] * 0.3
        touch_ema21 = (data["low"] <= data["ema21"] + touch_zone) & (data["high"] >= data["ema21"] - touch_zone)
        
        # Bounce: bullish bar after touch
        bullish_bar = data["close"] > data["open"]
        bearish_bar = data["close"] < data["open"]
        
        long_cond = uptrend & touch_ema21 & bullish_bar
        short_cond = downtrend & touch_ema21 & bearish_bar
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ema21"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ema21"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on trend reversal
        trend_change = (uptrend & (~uptrend.shift(1).fillna(False))) | (downtrend & (~downtrend.shift(1).fillna(False)))
        data.loc[trend_change.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 21", "column": "ema21", "color": "blue", "style": "solid"},
            {"name": "EMA 50", "column": "ema50", "color": "red", "style": "solid"},
        ]
