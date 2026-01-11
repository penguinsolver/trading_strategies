"""
Strategy: Momentum Continuation

Continue trading in direction of strong momentum.
Enter on pullback in strong trend.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class MomentumContinuationStrategy(Strategy):
    """Momentum continuation strategy."""
    
    @property
    def name(self) -> str:
        return "Mom Continue"
    
    @property
    def description(self) -> str:
        return "Continue trading in direction of strong recent momentum."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum continuation signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema10"] = ema(data["close"], 10)
        
        # Calculate momentum (10-bar return)
        data["momentum"] = (data["close"] - data["close"].shift(10)) / data["close"].shift(10) * 100
        
        # Strong momentum
        strong_up = data["momentum"] > 2  # >2% gain in 10 bars
        strong_down = data["momentum"] < -2  # <-2% loss in 10 bars
        
        # Pullback to EMA
        touched_ema = (data["low"] <= data["ema10"] * 1.002) & (data["high"] >= data["ema10"] * 0.998)
        
        long_cond = strong_up & touched_ema & (data["close"] > data["ema10"])
        short_cond = strong_down & touched_ema & (data["close"] < data["ema10"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ema10"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ema10"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on momentum reversal
        mom_reversal = (strong_up.shift(1) & strong_down) | (strong_down.shift(1) & strong_up)
        data.loc[mom_reversal.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 10", "column": "ema10", "color": "blue", "style": "solid"}]
