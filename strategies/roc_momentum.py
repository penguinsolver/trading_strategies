"""
Strategy: ROC (Rate of Change) Momentum

Simple but effective momentum using rate of change.
Pure price momentum without noise filtering.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Calculate Rate of Change."""
    return ((close - close.shift(period)) / close.shift(period)) * 100


class ROCMomentumStrategy(Strategy):
    """Rate of Change momentum strategy."""
    
    @property
    def name(self) -> str:
        return "ROC Momentum"
    
    @property
    def description(self) -> str:
        return "Trade ROC zero-line crosses for pure price momentum signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="ROC Period",
                param_type="int",
                default=12,
                min_value=5,
                max_value=20,
                step=2,
                help_text="Lookback for rate of change",
            ),
            ParamConfig(
                name="threshold",
                label="Entry Threshold",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=2.0,
                step=0.5,
                help_text="Min ROC for entry (abs value)",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ROC signals."""
        period = self.params.get("period", 12)
        threshold = self.params.get("threshold", 0.5)
        
        # Calculate ROC
        data["roc"] = roc(data["close"], period)
        data["roc_ema"] = ema(data["roc"], 5)  # Smoothed ROC
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_roc = data["roc"].shift(1)
        
        # Long: ROC crosses above threshold
        long_cond = (data["roc"] > threshold) & (prev_roc <= threshold)
        
        # Short: ROC crosses below -threshold
        short_cond = (data["roc"] < -threshold) & (prev_roc >= -threshold)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit at zero crossing
        zero_cross = (data["roc"] * prev_roc) < 0
        data.loc[zero_cross.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "ROC", "column": "roc", "color": "blue", "style": "solid"},
            {"name": "ROC EMA", "column": "roc_ema", "color": "orange", "style": "dashed"},
        ]
