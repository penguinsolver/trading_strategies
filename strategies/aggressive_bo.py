"""Strategy: Aggressive Breakout - Trade breakouts with minimal confirmation."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr

class AggressiveBreakoutStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Aggressive BO"
    
    @property
    def description(self) -> str:
        return "Trade immediate breakouts of 10-bar high/low."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["high10"] = data["high"].rolling(10).max().shift(1)
        data["low10"] = data["low"].rolling(10).min().shift(1)
        
        long_cond = data["close"] > data["high10"]
        short_cond = data["close"] < data["low10"]
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low10"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"]
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high10"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"]
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
