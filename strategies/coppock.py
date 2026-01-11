"""
Strategy: Coppock Curve

Long-term momentum indicator designed for monthly charts.
On intraday, identifies major momentum shifts.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def wma(data: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def coppock_curve(close: pd.Series, wma_period: int = 10, roc1: int = 14, roc2: int = 11) -> pd.Series:
    """Calculate Coppock Curve."""
    roc_long = ((close - close.shift(roc1)) / close.shift(roc1)) * 100
    roc_short = ((close - close.shift(roc2)) / close.shift(roc2)) * 100
    
    return wma(roc_long + roc_short, wma_period)


class CoppockCurveStrategy(Strategy):
    """Coppock Curve strategy."""
    
    @property
    def name(self) -> str:
        return "Coppock"
    
    @property
    def description(self) -> str:
        return "Trade Coppock Curve zero-line crosses for major momentum shifts."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Coppock signals."""
        # Calculate Coppock
        data["coppock"] = coppock_curve(data["close"])
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_copp = data["coppock"].shift(1)
        
        # Long: Coppock crosses above zero
        long_cond = (data["coppock"] > 0) & (prev_copp <= 0)
        
        # Short: Coppock crosses below zero
        short_cond = (data["coppock"] < 0) & (prev_copp >= 0)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 2
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.5
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 2
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.5
        
        # Exit on opposite cross
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Coppock", "column": "coppock", "color": "purple", "style": "solid"},
        ]
