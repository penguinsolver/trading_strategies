"""
Strategy: TRIX Momentum

TRIX triple-smoothed momentum indicator for trend following.
TRIX filters out noise through triple EMA smoothing.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """Calculate TRIX indicator (triple exponential smoothing)."""
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    
    trix_val = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    return trix_val


class TRIXMomentumStrategy(Strategy):
    """TRIX momentum strategy."""
    
    @property
    def name(self) -> str:
        return "TRIX Momentum"
    
    @property
    def description(self) -> str:
        return "Trade TRIX zero-line crosses with signal line confirmation for filtered momentum signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="TRIX Period",
                param_type="int",
                default=15,
                min_value=10,
                max_value=25,
                step=5,
                help_text="Triple EMA smoothing period",
            ),
            ParamConfig(
                name="signal_period",
                label="Signal Period",
                param_type="int",
                default=9,
                min_value=5,
                max_value=15,
                step=2,
                help_text="Signal line EMA period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate TRIX signals."""
        period = self.params.get("period", 15)
        signal_period = self.params.get("signal_period", 9)
        
        # Calculate TRIX
        data["trix"] = trix(data["close"], period)
        data["trix_signal"] = ema(data["trix"], signal_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_trix = data["trix"].shift(1)
        prev_signal = data["trix_signal"].shift(1)
        
        # Long: TRIX crosses above signal line
        long_cond = (
            (data["trix"] > data["trix_signal"]) &
            (prev_trix <= prev_signal)
        )
        
        # Short: TRIX crosses below signal line  
        short_cond = (
            (data["trix"] < data["trix_signal"]) &
            (prev_trix >= prev_signal)
        )
        
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
            {"name": "TRIX", "column": "trix", "color": "blue", "style": "solid"},
            {"name": "Signal", "column": "trix_signal", "color": "orange", "style": "dashed"},
        ]
