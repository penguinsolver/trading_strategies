"""
Strategy: Chaikin Oscillator (CHO)

Chaikin Oscillator measures momentum of A/D line.
Trade CHO zero-line crosses for volume-momentum signals.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, 
                       fast: int = 3, slow: int = 10) -> pd.Series:
    """Calculate Chaikin Oscillator."""
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.fillna(0)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # A/D Line
    ad = mfv.cumsum()
    
    # Chaikin Oscillator (fast EMA - slow EMA of A/D)
    cho = ema(ad, fast) - ema(ad, slow)
    return cho


class ChaikinOscillatorStrategy(Strategy):
    """Chaikin Oscillator strategy."""
    
    @property
    def name(self) -> str:
        return "Chaikin Osc"
    
    @property
    def description(self) -> str:
        return "Trade Chaikin Oscillator zero-line crosses for volume-weighted momentum."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast",
                label="Fast EMA",
                param_type="int",
                default=3,
                min_value=2,
                max_value=5,
                step=1,
                help_text="Fast EMA period",
            ),
            ParamConfig(
                name="slow",
                label="Slow EMA", 
                param_type="int",
                default=10,
                min_value=7,
                max_value=20,
                step=3,
                help_text="Slow EMA period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Chaikin Oscillator signals."""
        fast = self.params.get("fast", 3)
        slow = self.params.get("slow", 10)
        
        # Calculate CHO
        data["cho"] = chaikin_oscillator(data["high"], data["low"], data["close"], data["volume"], fast, slow)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_cho = data["cho"].shift(1)
        
        # Long: CHO crosses above zero
        long_cond = (data["cho"] > 0) & (prev_cho <= 0)
        
        # Short: CHO crosses below zero
        short_cond = (data["cho"] < 0) & (prev_cho >= 0)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on opposite cross
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Chaikin Osc", "column": "cho", "color": "teal", "style": "solid"},
        ]
