"""
Strategy: Simple Price Action

Ultra simple: trade based on 2 consecutive same-direction bars.
Sometimes simpler is better.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class SimplePriceActionStrategy(Strategy):
    """Simple price action strategy."""
    
    @property
    def name(self) -> str:
        return "Simple PA"
    
    @property
    def description(self) -> str:
        return "Ultra simple: trade after 2 consecutive green/red bars."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate simple price action signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Bar colors
        green = data["close"] > data["open"]
        red = data["close"] < data["open"]
        
        # 2 consecutive
        two_green = green & green.shift(1)
        two_red = red & red.shift(1)
        
        # But not 3 (avoid chasing)
        not_three_green = ~green.shift(2).fillna(False)
        not_three_red = ~red.shift(2).fillna(False)
        
        long_cond = two_green & not_three_green
        short_cond = two_red & not_three_red
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data["low"].shift(1).loc[long_mask]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data["high"].shift(1).loc[short_mask]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit on reversal bar
        data.loc[red.fillna(False), "exit_signal"] = True  # Exit longs
        data.loc[green.fillna(False), "exit_signal"] = True  # Exit shorts
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
