"""
Strategy: KST (Know Sure Thing) Oscillator

Martin Pring's KST combines multiple ROC periods.
Excellent for identifying major trend changes.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import sma, atr


def kst(close: pd.Series) -> tuple:
    """Calculate KST oscillator."""
    # ROC periods
    roc1 = ((close - close.shift(10)) / close.shift(10)) * 100
    roc2 = ((close - close.shift(15)) / close.shift(15)) * 100
    roc3 = ((close - close.shift(20)) / close.shift(20)) * 100
    roc4 = ((close - close.shift(30)) / close.shift(30)) * 100
    
    # Smooth and weight
    kst_val = (sma(roc1, 10) * 1 + sma(roc2, 10) * 2 + sma(roc3, 10) * 3 + sma(roc4, 15) * 4)
    signal = sma(kst_val, 9)
    
    return kst_val, signal


class KSTStrategy(Strategy):
    """KST oscillator strategy."""
    
    @property
    def name(self) -> str:
        return "KST"
    
    @property  
    def description(self) -> str:
        return "Trade KST signal line crosses for multi-timeframe momentum confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate KST signals."""
        # Calculate KST
        kst_val, signal = kst(data["close"])
        data["kst"] = kst_val
        data["kst_signal"] = signal
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_kst = kst_val.shift(1)
        prev_signal = signal.shift(1)
        
        # Long: KST crosses above signal
        long_cond = (kst_val > signal) & (prev_kst <= prev_signal)
        
        # Short: KST crosses below signal
        short_cond = (kst_val < signal) & (prev_kst >= prev_signal)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 2
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 2
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on opposite cross
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "KST", "column": "kst", "color": "blue", "style": "solid"},
            {"name": "Signal", "column": "kst_signal", "color": "orange", "style": "dashed"},
        ]
