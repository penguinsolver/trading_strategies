"""Strategy: Tight Range Break - Trade breakouts from very tight ranges."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr

class TightRangeBreakStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Tight Range"
    
    @property
    def description(self) -> str:
        return "Trade breakouts when range < 0.5 ATR (tight consolidation)."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        bar_range = data["high"] - data["low"]
        tight = bar_range < data["atr"] * 0.5
        
        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)
        prev_tight = tight.shift(1).fillna(False)
        
        # Breakout after tight bar
        breakout_up = prev_tight & (data["close"] > prev_high)
        breakout_down = prev_tight & (data["close"] < prev_low)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = breakout_up.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = prev_low.loc[long_mask]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = breakout_down.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = prev_high.loc[short_mask]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
