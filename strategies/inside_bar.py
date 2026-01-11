"""
Strategy: Inside Bar / NR7 Breakout

Compression pattern breakout strategy.
Entry: Break of inside bar or narrowest range pattern
Exit: R-based target or trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class InsideBarStrategy(Strategy):
    """Inside bar and NR7 compression pattern breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Inside Bar"
    
    @property
    def description(self) -> str:
        return "Breakout of compression patterns (inside bar, NR7). Enters on pattern break with tight stop."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="pattern_type",
                label="Pattern Type",
                param_type="select",
                default="inside",
                options=["inside", "nr7", "both"],
                help_text="Pattern type to trade: inside bar, narrow range, or both",
            ),
            ParamConfig(
                name="nr_lookback",
                label="NR Lookback",
                param_type="int",
                default=7,
                min_value=4,
                max_value=14,
                step=1,
                help_text="Lookback period for narrowest range detection",
            ),
            ParamConfig(
                name="target_r",
                label="Target R-Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=4.0,
                step=0.5,
                help_text="R-multiple target for exits",
            ),
            ParamConfig(
                name="use_trail",
                label="Use Trailing Stop",
                param_type="bool",
                default=True,
                help_text="Trail stop after hitting 1R profit",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ATR period for trailing stop sizing",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate inside bar / NR7 breakout signals."""
        # Parameters
        pattern_type = self.params.get("pattern_type", "inside")
        nr_lookback = self.params.get("nr_lookback", 7)
        target_r = self.params.get("target_r", 2.0)
        use_trail = self.params.get("use_trail", True)
        atr_period = self.params.get("atr_period", 14)
        
        # Calculate bar range
        data["bar_range"] = data["high"] - data["low"]
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Inside bar detection: current bar's range is inside previous bar
        # High < previous high AND low > previous low
        data["inside_bar"] = (
            (data["high"] < data["high"].shift(1)) &
            (data["low"] > data["low"].shift(1))
        )
        
        # NR7 detection: narrowest range in N bars
        data["min_range_n"] = data["bar_range"].rolling(window=nr_lookback, min_periods=nr_lookback).min()
        data["nr_bar"] = data["bar_range"] == data["min_range_n"]
        
        # Determine which patterns to use
        if pattern_type == "inside":
            data["pattern"] = data["inside_bar"]
        elif pattern_type == "nr7":
            data["pattern"] = data["nr_bar"]
        else:  # both
            data["pattern"] = data["inside_bar"] | data["nr_bar"]
        
        # Store pattern high/low for breakout detection
        # We check on the bar AFTER the pattern
        data["pattern_high"] = data["high"].shift(1)
        data["pattern_low"] = data["low"].shift(1)
        data["was_pattern"] = data["pattern"].shift(1).fillna(False)
        
        # Breakout detection
        # Long: Price breaks above pattern high
        data["breakout_long"] = (
            data["was_pattern"] &
            (data["close"] > data["pattern_high"])
        )
        
        # Short: Price breaks below pattern low
        data["breakout_short"] = (
            data["was_pattern"] &
            (data["close"] < data["pattern_low"])
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["target_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["breakout_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        # Stop at pattern low
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "pattern_low"]
        # Calculate risk (entry - stop)
        long_risk = data.loc[long_mask, "close"] - data.loc[long_mask, "pattern_low"]
        # Target = entry + target_r * risk
        data.loc[long_mask, "target_price"] = data.loc[long_mask, "close"] + (target_r * long_risk)
        if use_trail:
            data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"]
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        # Stop at pattern high
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "pattern_high"]
        # Calculate risk
        short_risk = data.loc[short_mask, "pattern_high"] - data.loc[short_mask, "close"]
        # Target
        data.loc[short_mask, "target_price"] = data.loc[short_mask, "close"] - (target_r * short_risk)
        if use_trail:
            data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"]
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []  # Pattern-based, no overlay indicators
