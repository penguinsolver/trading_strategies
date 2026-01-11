"""
Strategy: Fast Trend Scalp

Ultra-fast trend scalping using 3/8 EMA.
Very quick entries and exits.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema
from indicators.moving_averages import crossover, crossunder


class FastTrendScalpStrategy(Strategy):
    """Fast trend scalp strategy."""
    
    @property
    def name(self) -> str:
        return "Fast Trend Scalp"
    
    @property
    def description(self) -> str:
        return "Ultra-fast 3/8 EMA crossover scalping."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fast trend signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 10)
        data["ema3"] = ema(data["close"], 3)
        data["ema8"] = ema(data["close"], 8)
        
        golden = crossover(data["ema3"], data["ema8"])
        death = crossunder(data["ema3"], data["ema8"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = golden.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.75
        
        short_mask = death.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.75
        
        data.loc[death.fillna(False), "exit_signal"] = True
        data.loc[golden.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 3", "column": "ema3", "color": "green", "style": "solid"},
            {"name": "EMA 8", "column": "ema8", "color": "red", "style": "solid"},
        ]
