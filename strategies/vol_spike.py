"""
Strategy: Volatility Spike

Trade after volatility spikes (big bars).
Big bars often signal continuation.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class VolatilitySpikeStrategy(Strategy):
    """Volatility spike strategy."""
    
    @property
    def name(self) -> str:
        return "Vol Spike"
    
    @property
    def description(self) -> str:
        return "Trade in direction of volatility spikes (big bars > 2x ATR)."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility spike signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Bar range
        data["bar_range"] = data["high"] - data["low"]
        
        # Spike: bar > 2x ATR
        spike = data["bar_range"] > data["atr"] * 2
        
        # Direction of spike
        bullish_spike = spike & (data["close"] > data["open"])
        bearish_spike = spike & (data["close"] < data["open"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Enter after spike in direction
        long_mask = bullish_spike.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = bearish_spike.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on reversal bar
        reversal = (data["close"] > data["open"]).shift(1) != (data["close"] > data["open"])
        data.loc[reversal.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
