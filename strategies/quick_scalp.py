"""
Strategy: Quick Momentum Scalp

Very fast momentum scalp with tight exits.
Trade quick moves and exit before reversal.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class QuickMomentumScalpStrategy(Strategy):
    """Quick momentum scalp strategy."""
    
    @property
    def name(self) -> str:
        return "Quick Scalp"
    
    @property
    def description(self) -> str:
        return "Fast momentum scalping with very tight stops and quick exits."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate quick scalp signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 10)
        data["ema5"] = ema(data["close"], 5)
        data["ema10"] = ema(data["close"], 10)
        
        # Fast momentum: 2 green bars + above both EMAs
        green_bar = data["close"] > data["open"]
        prev_green = green_bar.shift(1)
        above_emas = (data["close"] > data["ema5"]) & (data["close"] > data["ema10"])
        
        red_bar = data["close"] < data["open"]
        prev_red = red_bar.shift(1)
        below_emas = (data["close"] < data["ema5"]) & (data["close"] < data["ema10"])
        
        long_cond = green_bar & prev_green & above_emas
        short_cond = red_bar & prev_red & below_emas
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"].shift(1)
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"].shift(1)
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.5
        
        # Quick exit on any reversal bar
        reversal_bar = (green_bar & prev_red) | (red_bar & prev_green)
        data.loc[reversal_bar.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 5", "column": "ema5", "color": "green", "style": "solid"},
            {"name": "EMA 10", "column": "ema10", "color": "blue", "style": "solid"},
        ]
