"""
Strategy: Ultimate Oscillator

Larry Williams' Ultimate Oscillator using 3 timeframes.
Reduces false signals by combining multiple periods.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                        p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Calculate Ultimate Oscillator."""
    prev_close = close.shift(1)
    
    # True Range
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    
    # Buying Pressure
    bp = close - np.minimum(low, prev_close)
    
    # Average calculations
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    
    # Ultimate Oscillator (weighted average)
    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    return uo


class UltimateOscillatorStrategy(Strategy):
    """Ultimate Oscillator strategy."""
    
    @property
    def name(self) -> str:
        return "Ultimate Osc"
    
    @property
    def description(self) -> str:
        return "Trade Ultimate Oscillator extremes using multi-timeframe momentum for stronger signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="overbought",
                label="Overbought",
                param_type="int",
                default=70,
                min_value=60,
                max_value=80,
                step=5,
                help_text="Overbought level",
            ),
            ParamConfig(
                name="oversold",
                label="Oversold",
                param_type="int",
                default=30,
                min_value=20,
                max_value=40,
                step=5,
                help_text="Oversold level",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Ultimate Oscillator signals."""
        overbought = self.params.get("overbought", 70)
        oversold = self.params.get("oversold", 30)
        
        # Calculate UO
        data["uo"] = ultimate_oscillator(data["high"], data["low"], data["close"])
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_uo = data["uo"].shift(1)
        
        # Long: UO crosses above oversold
        long_cond = (data["uo"] > oversold) & (prev_uo <= oversold)
        
        # Short: UO crosses below overbought
        short_cond = (data["uo"] < overbought) & (prev_uo >= overbought)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit at neutral
        exit_cond = (data["uo"] > 40) & (data["uo"] < 60)
        data.loc[exit_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Ultimate Osc", "column": "uo", "color": "purple", "style": "solid"},
        ]
