"""
Strategy: RSI Trending

Use RSI in trending mode (50 as pivot instead of 70/30).
Trade RSI crosses above/below 50.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class RSITrendingStrategy(Strategy):
    """RSI trending strategy."""
    
    @property
    def name(self) -> str:
        return "RSI Trending"
    
    @property
    def description(self) -> str:
        return "Trade RSI crosses above/below 50 with EMA trend confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI trending signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["rsi"] = rsi(data["close"], 14)
        data["ema20"] = ema(data["close"], 20)
        
        # RSI crosses 50
        above_50 = data["rsi"] > 50
        below_50 = data["rsi"] < 50
        prev_above = above_50.shift(1).fillna(False)
        prev_below = below_50.shift(1).fillna(False)
        
        cross_up = above_50 & (~prev_above)
        cross_down = below_50 & (~prev_below)
        
        # Trend confirmation
        above_ema = data["close"] > data["ema20"]
        below_ema = data["close"] < data["ema20"]
        
        long_cond = cross_up & above_ema
        short_cond = cross_down & below_ema
        
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
        return [{"name": "RSI", "column": "rsi", "color": "purple", "style": "solid"}]
