"""Strategy: Quick Reversal - Fast mean reversion scalping."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr, sma

class QuickReversalStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Quick Reversal"
    
    @property
    def description(self) -> str:
        return "Fast mean reversion when price extends > 1 ATR from SMA."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["sma20"] = sma(data["close"], 20)
        
        dist = data["close"] - data["sma20"]
        extended_up = dist > data["atr"] * 1.5
        extended_down = dist < -data["atr"] * 1.5
        
        # Entry on reversal bar after extension
        reversal_down = extended_up.shift(1) & (data["close"] < data["open"])
        reversal_up = extended_down.shift(1) & (data["close"] > data["open"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = reversal_up.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.5
        
        short_mask = reversal_down.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.5
        
        # Exit at SMA
        at_sma = (data["close"] > data["sma20"] - data["atr"] * 0.25) & (data["close"] < data["sma20"] + data["atr"] * 0.25)
        data.loc[at_sma.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "SMA 20", "column": "sma20", "color": "blue", "style": "solid"}]
