"""Strategy: Micro Trend - Trade micro trends using 5 EMA."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr, ema

class MicroTrendStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Micro Trend"
    
    @property
    def description(self) -> str:
        return "Trade micro trends using 5 EMA direction."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 10)
        data["ema5"] = ema(data["close"], 5)
        
        rising = data["ema5"] > data["ema5"].shift(1)
        falling = data["ema5"] < data["ema5"].shift(1)
        prev_rising = rising.shift(1).fillna(False)
        prev_falling = falling.shift(1).fillna(False)
        
        long_cond = rising & (~prev_rising)
        short_cond = falling & (~prev_falling)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.75
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1  
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.75
        
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 5", "column": "ema5", "color": "blue", "style": "solid"}]
