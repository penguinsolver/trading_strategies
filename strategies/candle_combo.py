"""
Strategy: Candle Pattern Combo

Combine multiple candle patterns for higher probability.
Enter on hammer/shooting star + confirmation.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class CandlePatternComboStrategy(Strategy):
    """Candle pattern combination strategy."""
    
    @property
    def name(self) -> str:
        return "Candle Combo"
    
    @property
    def description(self) -> str:
        return "Trade hammer/shooting star patterns with next bar confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate candle pattern signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        body = abs(data["close"] - data["open"])
        upper_wick = data["high"] - data[["close", "open"]].max(axis=1)
        lower_wick = data[["close", "open"]].min(axis=1) - data["low"]
        candle_range = data["high"] - data["low"]
        
        # Hammer: long lower wick, small body at top
        hammer = (lower_wick > body * 2) & (upper_wick < body) & (body / candle_range < 0.35)
        
        # Shooting star: long upper wick, small body at bottom
        shooting_star = (upper_wick > body * 2) & (lower_wick < body) & (body / candle_range < 0.35)
        
        # Confirmation: next bar closes in direction
        hammer_confirmed = hammer.shift(1) & (data["close"] > data["open"])
        shooting_confirmed = shooting_star.shift(1) & (data["close"] < data["open"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = hammer_confirmed.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data["low"].shift(1).loc[long_mask] - data.loc[long_mask, "atr"] * 0.25
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = shooting_confirmed.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data["high"].shift(1).loc[short_mask] + data.loc[short_mask, "atr"] * 0.25
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
