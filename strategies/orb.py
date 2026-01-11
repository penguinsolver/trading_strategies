"""
Strategy: Opening Range Breakout (ORB)

Session-based breakout strategy.
Entry: Break of opening range high/low within defined session
Exit: End of session, R-based target, or trailing stop
"""
import pandas as pd
import numpy as np
from datetime import time

from .base import Strategy, ParamConfig
from indicators import atr


class ORBStrategy(Strategy):
    """Opening Range Breakout session-based strategy."""
    
    @property
    def name(self) -> str:
        return "Opening Range"
    
    @property
    def description(self) -> str:
        return "Session-based opening range breakout. Enters on break of opening range, exits at session end."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="session_start_utc",
                label="Session Start (UTC)",
                param_type="int",
                default=14,
                min_value=0,
                max_value=23,
                step=1,
                help_text="Session start hour in UTC (14 = US market open)",
            ),
            ParamConfig(
                name="session_end_utc",
                label="Session End (UTC)",
                param_type="int",
                default=21,
                min_value=0,
                max_value=23,
                step=1,
                help_text="Session end hour in UTC (21 = US market close)",
            ),
            ParamConfig(
                name="range_minutes",
                label="Range Duration (min)",
                param_type="int",
                default=60,
                min_value=15,
                max_value=120,
                step=15,
                help_text="Opening range duration in minutes",
            ),
            ParamConfig(
                name="exit_mode",
                label="Exit Mode",
                param_type="select",
                default="session",
                options=["session", "r_target", "trail"],
                help_text="Exit strategy: session end, R-target, or trailing stop",
            ),
            ParamConfig(
                name="target_r",
                label="Target R-Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=4.0,
                step=0.5,
                help_text="R-multiple target (if using r_target exit)",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ATR period for stop sizing",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Opening Range Breakout signals."""
        # Parameters
        session_start = self.params.get("session_start_utc", 14)
        session_end = self.params.get("session_end_utc", 21)
        range_minutes = self.params.get("range_minutes", 60)
        exit_mode = self.params.get("exit_mode", "session")
        target_r = self.params.get("target_r", 2.0)
        atr_period = self.params.get("atr_period", 14)
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Extract hour from index
        data["hour"] = data.index.hour
        data["minute"] = data.index.minute
        data["date"] = data.index.date
        
        # Identify session state
        # Handle session that crosses midnight
        if session_end > session_start:
            data["in_session"] = (data["hour"] >= session_start) & (data["hour"] < session_end)
        else:
            data["in_session"] = (data["hour"] >= session_start) | (data["hour"] < session_end)
        
        # Identify opening range period (first N minutes of session)
        # Calculate minutes into session
        data["minutes_into_session"] = np.where(
            data["in_session"],
            (data["hour"] - session_start) * 60 + data["minute"],
            -1
        )
        data.loc[data["minutes_into_session"] < 0, "minutes_into_session"] = -1
        
        data["in_opening_range"] = (data["minutes_into_session"] >= 0) & (data["minutes_into_session"] < range_minutes)
        
        # Calculate opening range high/low for each session
        # Group by date and calculate OR levels
        data["or_high"] = np.nan
        data["or_low"] = np.nan
        
        # Process each date
        for date_val in data["date"].unique():
            date_mask = data["date"] == date_val
            or_mask = date_mask & data["in_opening_range"]
            
            if or_mask.any():
                or_high = data.loc[or_mask, "high"].max()
                or_low = data.loc[or_mask, "low"].min()
                
                # Apply to all bars in session after opening range
                session_mask = date_mask & data["in_session"] & ~data["in_opening_range"]
                data.loc[session_mask, "or_high"] = or_high
                data.loc[session_mask, "or_low"] = or_low
        
        # Forward fill within each session
        data["or_high"] = data["or_high"].ffill()
        data["or_low"] = data["or_low"].ffill()
        
        # Breakout detection (only during session, after opening range)
        data["trading_window"] = data["in_session"] & ~data["in_opening_range"]
        
        # Long: Close breaks above OR high
        data["breakout_long"] = (
            data["trading_window"] &
            (data["close"] > data["or_high"]) &
            (data["close"].shift(1) <= data["or_high"])
        )
        
        # Short: Close breaks below OR low
        data["breakout_short"] = (
            data["trading_window"] &
            (data["close"] < data["or_low"]) &
            (data["close"].shift(1) >= data["or_low"])
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
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "or_low"]
        
        if exit_mode == "r_target":
            long_risk = data.loc[long_mask, "close"] - data.loc[long_mask, "or_low"]
            data.loc[long_mask, "target_price"] = data.loc[long_mask, "close"] + (target_r * long_risk)
        elif exit_mode == "trail":
            data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"]
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "or_high"]
        
        if exit_mode == "r_target":
            short_risk = data.loc[short_mask, "or_high"] - data.loc[short_mask, "close"]
            data.loc[short_mask, "target_price"] = data.loc[short_mask, "close"] - (target_r * short_risk)
        elif exit_mode == "trail":
            data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"]
        
        # Session end exit (for session exit mode)
        if exit_mode == "session":
            # Detect session end
            data["session_ending"] = data["in_session"] & ~data["in_session"].shift(-1).fillna(False)
            data.loc[data["session_ending"], "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "OR High", "column": "or_high", "color": "green", "style": "dashed"},
            {"name": "OR Low", "column": "or_low", "color": "red", "style": "dashed"},
        ]
