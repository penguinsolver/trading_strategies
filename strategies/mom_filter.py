"""Strategy: Momentum Filter - Only trade high momentum."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr, ema

class MomentumFilterStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Mom Filter"
    
    @property
    def description(self) -> str:
        return "Trade EMA cross only when momentum (ROC) is strong."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema10"] = ema(data["close"], 10)
        data["ema20"] = ema(data["close"], 20)
        
        # Momentum: 5-bar return
        data["roc5"] = (data["close"] - data["close"].shift(5)) / data["close"].shift(5) * 100
        
        # EMA cross
        above = data["ema10"] > data["ema20"]
        below = data["ema10"] < data["ema20"]
        prev_above = above.shift(1).fillna(False)
        prev_below = below.shift(1).fillna(False)
        
        cross_up = above & (~prev_above)
        cross_down = below & (~prev_below)
        
        # Momentum filter: > 1% ROC
        strong_up = data["roc5"] > 1
        strong_down = data["roc5"] < -1
        
        long_cond = cross_up & strong_up
        short_cond = cross_down & strong_down
        
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
        
        data.loc[cross_down.fillna(False), "exit_signal"] = True
        data.loc[cross_up.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 10", "column": "ema10", "color": "blue", "style": "solid"}]
