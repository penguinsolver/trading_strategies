"""Strategy: Opening Move - Trade strong opening moves."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr

class OpeningMoveStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Opening Move"
    
    @property
    def description(self) -> str:
        return "Trade when open is far from previous close (gap)."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Opening gap
        gap = data["open"] - data["close"].shift(1)
        big_gap_up = gap > data["atr"] * 0.5
        big_gap_down = gap < -data["atr"] * 0.5
        
        # Continuation in gap direction
        continue_up = big_gap_up & (data["close"] > data["open"])
        continue_down = big_gap_down & (data["close"] < data["open"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = continue_up.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "open"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = continue_down.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "open"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
