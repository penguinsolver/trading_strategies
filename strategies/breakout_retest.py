"""
Strategy: Breakout + Retest

Higher quality breakout strategy that waits for price to retest the breakout level.
Entry: After breakout, wait for retest and reclaim of level
Exit: ATR trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, donchian_channels


class BreakoutRetestStrategy(Strategy):
    """Breakout with retest confirmation strategy."""
    
    @property
    def name(self) -> str:
        return "Breakout Retest"
    
    @property
    def description(self) -> str:
        return "Higher quality breakouts with retest confirmation. Waits for price to return to level before entering."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="level_source",
                label="Level Source",
                param_type="select",
                default="donchian",
                options=["donchian", "prior_day"],
                help_text="Source for breakout levels: Donchian channel or prior day high/low",
            ),
            ParamConfig(
                name="donchian_period",
                label="Donchian Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=30,
                step=5,
                help_text="Donchian channel period (if using donchian source)",
            ),
            ParamConfig(
                name="retest_window",
                label="Retest Window (Bars)",
                param_type="int",
                default=5,
                min_value=3,
                max_value=10,
                step=1,
                help_text="Number of bars to wait for retest after breakout",
            ),
            ParamConfig(
                name="reclaim_rule",
                label="Reclaim Rule",
                param_type="select",
                default="close",
                options=["close", "wick"],
                help_text="How to confirm reclaim: close above level or just wick touch",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ATR period for stop calculation",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=1.5,
                min_value=1.0,
                max_value=2.5,
                step=0.5,
                help_text="ATR multiple for stop loss",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout + retest signals."""
        # Parameters
        level_source = self.params.get("level_source", "donchian")
        donchian_period = self.params.get("donchian_period", 20)
        retest_window = self.params.get("retest_window", 5)
        reclaim_rule = self.params.get("reclaim_rule", "close")
        atr_period = self.params.get("atr_period", 14)
        atr_stop_mult = self.params.get("atr_stop_mult", 1.5)
        
        # Calculate levels based on source
        if level_source == "donchian":
            dc_upper, dc_lower, dc_mid = donchian_channels(data["high"], data["low"], donchian_period)
            data["level_high"] = dc_upper.shift(1)
            data["level_low"] = dc_lower.shift(1)
        else:  # prior_day
            # Resample to daily to get prior day high/low
            if isinstance(data.index, pd.DatetimeIndex):
                data["date"] = data.index.date
                # Calculate prior day high/low
                daily_high = data.groupby("date")["high"].transform("max").shift(1)
                daily_low = data.groupby("date")["low"].transform("min").shift(1)
                data["level_high"] = daily_high.ffill()
                data["level_low"] = daily_low.ffill()
            else:
                # Fallback to donchian if no datetime index
                dc_upper, dc_lower, dc_mid = donchian_channels(data["high"], data["low"], donchian_period)
                data["level_high"] = dc_upper.shift(1)
                data["level_low"] = dc_lower.shift(1)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Detect initial breakout
        data["initial_breakout_long"] = (
            (data["close"] > data["level_high"]) &
            (data["close"].shift(1) <= data["level_high"].shift(1))
        )
        
        data["initial_breakout_short"] = (
            (data["close"] < data["level_low"]) &
            (data["close"].shift(1) >= data["level_low"].shift(1))
        )
        
        # Track bars since breakout
        data["bars_since_long_breakout"] = 0
        data["bars_since_short_breakout"] = 0
        data["breakout_level_long"] = np.nan
        data["breakout_level_short"] = np.nan
        
        # Initialize tracking
        bars_since_long = 0
        bars_since_short = 0
        long_level = np.nan
        short_level = np.nan
        
        for i in range(len(data)):
            if data["initial_breakout_long"].iloc[i]:
                bars_since_long = 1
                long_level = data["level_high"].iloc[i]
            elif bars_since_long > 0:
                bars_since_long += 1
                if bars_since_long > retest_window * 2:  # Reset if too long
                    bars_since_long = 0
                    long_level = np.nan
            
            if data["initial_breakout_short"].iloc[i]:
                bars_since_short = 1
                short_level = data["level_low"].iloc[i]
            elif bars_since_short > 0:
                bars_since_short += 1
                if bars_since_short > retest_window * 2:
                    bars_since_short = 0
                    short_level = np.nan
            
            data.loc[data.index[i], "bars_since_long_breakout"] = bars_since_long
            data.loc[data.index[i], "bars_since_short_breakout"] = bars_since_short
            data.loc[data.index[i], "breakout_level_long"] = long_level
            data.loc[data.index[i], "breakout_level_short"] = short_level
        
        # Detect retest and reclaim
        if reclaim_rule == "close":
            # Long retest: price dipped below level and closed back above
            data["retest_long"] = (
                (data["bars_since_long_breakout"] > 1) &
                (data["bars_since_long_breakout"] <= retest_window) &
                (data["low"] <= data["breakout_level_long"]) &  # Touched level
                (data["close"] > data["breakout_level_long"])   # Reclaimed
            )
            
            # Short retest: price spiked above level and closed back below
            data["retest_short"] = (
                (data["bars_since_short_breakout"] > 1) &
                (data["bars_since_short_breakout"] <= retest_window) &
                (data["high"] >= data["breakout_level_short"]) &
                (data["close"] < data["breakout_level_short"])
            )
        else:  # wick touch
            # More aggressive: just need to touch level
            data["retest_long"] = (
                (data["bars_since_long_breakout"] > 1) &
                (data["bars_since_long_breakout"] <= retest_window) &
                (data["low"] <= data["breakout_level_long"])
            )
            
            data["retest_short"] = (
                (data["bars_since_short_breakout"] > 1) &
                (data["bars_since_short_breakout"] <= retest_window) &
                (data["high"] >= data["breakout_level_short"])
            )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries on retest
        long_mask = data["retest_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        # Stop below retest low
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - (
            data.loc[long_mask, "atr"] * atr_stop_mult * 0.5  # Tighter stop for retests
        )
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries on retest
        short_mask = data["retest_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + (
            data.loc[short_mask, "atr"] * atr_stop_mult * 0.5
        )
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Level High", "column": "level_high", "color": "green", "style": "dashed"},
            {"name": "Level Low", "column": "level_low", "color": "red", "style": "dashed"},
        ]
