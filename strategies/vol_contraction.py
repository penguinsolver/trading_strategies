"""
Strategy: Volatility Contraction Breakout

Trade breakouts after volatility contracts (squeeze).
Low volatility often precedes explosive moves.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class VolatilityContractionStrategy(Strategy):
    """Volatility contraction breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Vol Contract"
    
    @property
    def description(self) -> str:
        return "Trade breakouts after ATR contracts to low levels."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility contraction signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["atr_avg"] = data["atr"].rolling(50).mean()
        
        # Low volatility = ATR below average
        low_vol = data["atr"] < data["atr_avg"] * 0.75
        
        # Range high/low during low vol
        data["range_high"] = data["high"].rolling(10).max()
        data["range_low"] = data["low"].rolling(10).min()
        
        # Breakout from contraction
        prev_low_vol = low_vol.shift(1).fillna(False)
        breakout_up = prev_low_vol & (data["close"] > data["range_high"].shift(1))
        breakout_down = prev_low_vol & (data["close"] < data["range_low"].shift(1))
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = breakout_up.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "range_low"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = breakout_down.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "range_high"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit when volatility expands
        high_vol = data["atr"] > data["atr_avg"] * 1.5
        data.loc[high_vol.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
