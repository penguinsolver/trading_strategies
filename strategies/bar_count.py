"""Strategy: Bar Count Trend - Trade after N bars in same direction."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr

class BarCountTrendStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Bar Count"
    
    @property
    def description(self) -> str:
        return "Trade after 4 consecutive bars close higher/lower."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        higher = data["close"] > data["close"].shift(1)
        lower = data["close"] < data["close"].shift(1)
        
        # 4 in a row
        four_higher = higher & higher.shift(1) & higher.shift(2) & higher.shift(3)
        four_lower = lower & lower.shift(1) & lower.shift(2) & lower.shift(3)
        
        prev_four_higher = four_higher.shift(1).fillna(False)
        prev_four_lower = four_lower.shift(1).fillna(False)
        
        long_cond = four_higher & (~prev_four_higher)
        short_cond = four_lower & (~prev_four_lower)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data["low"].rolling(4).min().loc[long_mask]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data["high"].rolling(4).max().loc[short_mask]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on reversal
        data.loc[lower.fillna(False), "exit_signal"] = True
        data.loc[higher.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
