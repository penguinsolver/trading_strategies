"""
Strategy: AD (Accumulation/Distribution) Line

Chaikin's A/D line measures buying vs selling pressure.
Trade divergences and trend confirmations.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Accumulation/Distribution Line."""
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.fillna(0)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Cumulative A/D Line
    ad = mfv.cumsum()
    return ad


class ADLineStrategy(Strategy):
    """Accumulation/Distribution line strategy."""
    
    @property
    def name(self) -> str:
        return "A/D Line"
    
    @property
    def description(self) -> str:
        return "Trade A/D line trends and EMA crosses for volume-based momentum."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast_ema",
                label="Fast EMA",
                param_type="int",
                default=3,
                min_value=2,
                max_value=10,
                step=2,
                help_text="Fast EMA of A/D line",
            ),
            ParamConfig(
                name="slow_ema",
                label="Slow EMA",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=5,
                help_text="Slow EMA of A/D line",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate A/D Line signals."""
        fast = self.params.get("fast_ema", 3)
        slow = self.params.get("slow_ema", 10)
        
        # Calculate A/D Line
        data["ad"] = ad_line(data["high"], data["low"], data["close"], data["volume"])
        data["ad_fast"] = ema(data["ad"], fast)
        data["ad_slow"] = ema(data["ad"], slow)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_fast = data["ad_fast"].shift(1)
        prev_slow = data["ad_slow"].shift(1)
        
        # Long: Fast A/D crosses above slow A/D
        long_cond = (data["ad_fast"] > data["ad_slow"]) & (prev_fast <= prev_slow)
        
        # Short: Fast A/D crosses below slow A/D
        short_cond = (data["ad_fast"] < data["ad_slow"]) & (prev_fast >= prev_slow)
        
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
            {"name": "A/D Fast", "column": "ad_fast", "color": "blue", "style": "solid"},
            {"name": "A/D Slow", "column": "ad_slow", "color": "orange", "style": "dashed"},
        ]
