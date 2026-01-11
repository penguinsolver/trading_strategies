"""Strategy: Doji Reversal - Trade doji patterns for reversals."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr

class DojiReversalStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Doji Reversal"
    
    @property
    def description(self) -> str:
        return "Trade doji patterns at extremes for reversal signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        body = abs(data["close"] - data["open"])
        candle_range = data["high"] - data["low"]
        
        # Doji: very small body relative to range
        doji = body < candle_range * 0.1
        
        # At extreme
        high5 = data["high"].rolling(5).max()
        low5 = data["low"].rolling(5).min()
        at_high = data["high"] >= high5
        at_low = data["low"] <= low5
        
        # Reversal confirmation
        next_up = data["close"].shift(-1) > data["close"]
        next_down = data["close"].shift(-1) < data["close"]
        
        # Use previous bar doji + confirmation
        long_cond = doji.shift(1) & at_low.shift(1) & (data["close"] > data["open"])
        short_cond = doji.shift(1) & at_high.shift(1) & (data["close"] < data["open"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data["low"].shift(1).loc[long_mask] - data.loc[long_mask, "atr"] * 0.25
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.75
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data["high"].shift(1).loc[short_mask] + data.loc[short_mask, "atr"] * 0.25
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.75
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
