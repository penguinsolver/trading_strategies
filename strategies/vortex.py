"""
Strategy: Vortex Indicator

Trade Vortex +VI/-VI crossovers for trend detection.
Measures upward and downward price movement.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """Calculate Vortex Indicator +VI and -VI."""
    # True Range
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    
    # Vortex Movement
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))
    
    # Sum over period
    sum_tr = tr.rolling(period).sum()
    sum_vm_plus = vm_plus.rolling(period).sum()
    sum_vm_minus = vm_minus.rolling(period).sum()
    
    # Vortex Indicator
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    return vi_plus, vi_minus


class VortexStrategy(Strategy):
    """Vortex Indicator crossover strategy."""
    
    @property
    def name(self) -> str:
        return "Vortex"
    
    @property
    def description(self) -> str:
        return "Trade Vortex +VI/-VI crosses for directional trend changes."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="Vortex Period",
                param_type="int",
                default=14,
                min_value=7,
                max_value=28,
                step=7,
                help_text="Lookback period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Vortex signals."""
        period = self.params.get("period", 14)
        
        # Calculate Vortex
        vi_plus, vi_minus = vortex_indicator(data["high"], data["low"], data["close"], period)
        data["vi_plus"] = vi_plus
        data["vi_minus"] = vi_minus
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_plus = vi_plus.shift(1)
        prev_minus = vi_minus.shift(1)
        
        # Long: +VI crosses above -VI
        long_cond = (vi_plus > vi_minus) & (prev_plus <= prev_minus)
        
        # Short: -VI crosses above +VI
        short_cond = (vi_minus > vi_plus) & (prev_minus <= prev_plus)
        
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
            {"name": "+VI", "column": "vi_plus", "color": "green", "style": "solid"},
            {"name": "-VI", "column": "vi_minus", "color": "red", "style": "solid"},
        ]
