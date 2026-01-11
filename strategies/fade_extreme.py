"""Strategy: Fade Extreme - Fade extreme moves."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr

class FadeExtremeStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Fade Extreme"
    
    @property
    def description(self) -> str:
        return "Fade moves that are > 2 ATR in a single bar."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        move = data["close"] - data["close"].shift(1)
        extreme_up = move > data["atr"] * 2
        extreme_down = move < -data["atr"] * 2
        
        # Fade the extreme (mean reversion)
        short_cond = extreme_up  # Fade up by shorting
        long_cond = extreme_down  # Fade down by going long
        
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
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
