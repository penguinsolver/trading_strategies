"""
Strategy: DMI Cross (Directional Movement Index)

Trade ADX with +DI/-DI crossovers for trend entry.
Combines strength (ADX) with direction (+DI/-DI).
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import adx, atr


class DMICrossStrategy(Strategy):
    """DMI crossover strategy with ADX filter."""
    
    @property
    def name(self) -> str:
        return "DMI Cross"
    
    @property
    def description(self) -> str:
        return "Trade +DI/-DI crossovers when ADX confirms trend strength for directional moves."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="DI Period",
                param_type="int",
                default=14,
                min_value=7,
                max_value=21,
                step=2,
                help_text="DMI calculation period",
            ),
            ParamConfig(
                name="adx_min",
                label="Min ADX",
                param_type="int",
                default=20,
                min_value=15,
                max_value=30,
                step=5,
                help_text="Minimum ADX for trend",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate DMI Cross signals."""
        period = self.params.get("period", 14)
        adx_min = self.params.get("adx_min", 20)
        
        # Calculate ADX and DI
        adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], period)
        data["adx"] = adx_val
        data["plus_di"] = plus_di
        data["minus_di"] = minus_di
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_plus = plus_di.shift(1)
        prev_minus = minus_di.shift(1)
        
        # Long: +DI crosses above -DI with ADX > threshold
        long_cond = (
            (plus_di > minus_di) &
            (prev_plus <= prev_minus) &
            (adx_val > adx_min)
        )
        
        # Short: -DI crosses above +DI with ADX > threshold
        short_cond = (
            (minus_di > plus_di) &
            (prev_minus <= prev_plus) &
            (adx_val > adx_min)
        )
        
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
        
        # Exit on opposite cross or ADX weakening
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "+DI", "column": "plus_di", "color": "green", "style": "solid"},
            {"name": "-DI", "column": "minus_di", "color": "red", "style": "solid"},
        ]
