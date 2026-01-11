"""
Strategy: Supertrend Flip

A trend-following strategy that uses the Supertrend indicator.
Entry: When Supertrend flips bullish (long) or bearish (short)
Exit: When Supertrend flips back to opposite direction
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, supertrend


class SupertrendStrategy(Strategy):
    """Supertrend flip trend-following strategy."""
    
    @property
    def name(self) -> str:
        return "Supertrend"
    
    @property
    def description(self) -> str:
        return "Trend-following strategy based on Supertrend indicator flips. Enters on direction change, exits on reversal."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=10,
                min_value=7,
                max_value=21,
                step=1,
                help_text="ATR period for Supertrend calculation",
            ),
            ParamConfig(
                name="multiplier",
                label="Multiplier",
                param_type="float",
                default=3.0,
                min_value=2.0,
                max_value=5.0,
                step=0.5,
                help_text="ATR multiplier for band width",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="Stop ATR Multiple",
                param_type="float",
                default=1.5,
                min_value=1.0,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiple for trailing stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Supertrend flip signals."""
        # Parameters
        atr_period = self.params.get("atr_period", 10)
        multiplier = self.params.get("multiplier", 3.0)
        atr_stop_mult = self.params.get("atr_stop_mult", 1.5)
        
        # Calculate Supertrend
        st_line, st_direction = supertrend(
            data["high"], data["low"], data["close"],
            atr_period=atr_period,
            multiplier=multiplier
        )
        data["supertrend"] = st_line
        data["st_direction"] = st_direction
        
        # Calculate ATR for stops
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Detect flips (direction change)
        data["prev_direction"] = data["st_direction"].shift(1)
        data["flip_long"] = (data["st_direction"] == 1) & (data["prev_direction"] == -1)
        data["flip_short"] = (data["st_direction"] == -1) & (data["prev_direction"] == 1)
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Generate signals - vectorized approach where possible
        # Long entries on bullish flip
        long_mask = data["flip_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "supertrend"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries on bearish flip
        short_mask = data["flip_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "supertrend"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        # Exit signals on opposite flip (handled by new entry, but mark explicit)
        # When we flip, we're exiting the old position and entering new
        data.loc[long_mask | short_mask, "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Supertrend", "column": "supertrend", "color": "purple", "style": "solid"},
        ]
