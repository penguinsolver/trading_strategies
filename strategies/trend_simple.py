"""Strategy: Trend Follow Simple - Simple EMA trend following."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr, ema

class TrendFollowSimpleStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Trend Simple"
    
    @property
    def description(self) -> str:
        return "Simple trend following: stay long above 20 EMA, short below."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema20"] = ema(data["close"], 20)
        
        above = data["close"] > data["ema20"]
        below = data["close"] < data["ema20"]
        prev_above = above.shift(1).fillna(False)
        prev_below = below.shift(1).fillna(False)
        
        long_cond = above & (~prev_above)
        short_cond = below & (~prev_below)
        
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
        
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 20", "column": "ema20", "color": "blue", "style": "solid"}]
