"""
Strategy: SMI (Stochastic Momentum Index)

SMI is a refined stochastic showing momentum vs just overbought/oversold.
More accurate than standard stochastic for entry timing.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def smi(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 10, d_period: int = 3, smoothing: int = 3) -> tuple:
    """Calculate Stochastic Momentum Index."""
    # Highest high and lowest low
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    
    # Distance from midpoint
    diff = close - (hh + ll) / 2
    range_ = hh - ll
    
    # Double smoothing
    diff_smooth = ema(ema(diff, d_period), d_period)
    range_smooth = ema(ema(range_, d_period), d_period)
    
    # SMI
    smi_val = 100 * (diff_smooth / (range_smooth / 2))
    signal = ema(smi_val, smoothing)
    
    return smi_val, signal


class SMIStrategy(Strategy):
    """SMI strategy."""
    
    @property
    def name(self) -> str:
        return "SMI"
    
    @property
    def description(self) -> str:
        return "Trade SMI signal crosses for refined stochastic momentum entries."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="k_period",
                label="%K Period",
                param_type="int",
                default=10,
                min_value=5,
                max_value=15,
                step=2,
                help_text="SMI lookback period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate SMI signals."""
        k_period = self.params.get("k_period", 10)
        
        # Calculate SMI
        smi_val, signal = smi(data["high"], data["low"], data["close"], k_period, 3, 3)
        data["smi"] = smi_val
        data["smi_signal"] = signal
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_smi = smi_val.shift(1)
        prev_signal = signal.shift(1)
        
        # Long: SMI crosses above signal from oversold
        long_cond = (
            (smi_val > signal) & 
            (prev_smi <= prev_signal) & 
            (prev_smi < -40)
        )
        
        # Short: SMI crosses below signal from overbought
        short_cond = (
            (smi_val < signal) & 
            (prev_smi >= prev_signal) & 
            (prev_smi > 40)
        )
        
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
        
        # Exit on opposite extreme
        data.loc[(smi_val > 40).fillna(False), "exit_signal"] = True
        data.loc[(smi_val < -40).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "SMI", "column": "smi", "color": "blue", "style": "solid"},
            {"name": "Signal", "column": "smi_signal", "color": "orange", "style": "dashed"},
        ]
