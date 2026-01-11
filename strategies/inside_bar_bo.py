"""
Strategy: Inside Bar Breakout

Trade breakouts from inside bars (consolidation).
Classic price action pattern.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class InsideBarBreakoutStrategy(Strategy):
    """Inside bar breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Inside Bar BO"
    
    @property
    def description(self) -> str:
        return "Trade breakouts from inside bar patterns with trend filter."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate inside bar breakout signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema50"] = ema(data["close"], 50)
        
        # Inside bar: current bar within previous bar range
        inside_bar = (data["high"] < data["high"].shift(1)) & (data["low"] > data["low"].shift(1))
        
        # Mother bar (previous bar)
        mother_high = data["high"].shift(1)
        mother_low = data["low"].shift(1)
        
        # Breakout on bar after inside bar
        had_inside = inside_bar.shift(1).fillna(False)
        breakout_up = had_inside & (data["close"] > mother_high.shift(1))
        breakout_down = had_inside & (data["close"] < mother_low.shift(1))
        
        # Trend filter
        above_ema = data["close"] > data["ema50"]
        below_ema = data["close"] < data["ema50"]
        
        long_cond = breakout_up & above_ema
        short_cond = breakout_down & below_ema
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = mother_low.shift(1).loc[long_mask]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = mother_high.shift(1).loc[short_mask]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 50", "column": "ema50", "color": "blue", "style": "solid"}]
