"""
Strategy: Previous-Day Range Breakout (Low Frequency)

Low-frequency breakout strategy trading yesterday's high/low.
Max 1 trade per day to reduce churn and costs.
"""
import pandas as pd
import numpy as np
from datetime import datetime

from .base import Strategy, ParamConfig
from indicators import atr


class PrevDayRangeStrategy(Strategy):
    """Previous day range breakout with daily trade limit."""
    
    @property
    def name(self) -> str:
        return "Prev Day Range"
    
    @property
    def description(self) -> str:
        return "Breakout of yesterday's high/low. Max 1 trade per day to reduce churn."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="daily_limit",
                label="Trades Per Day",
                param_type="int",
                default=1,
                min_value=1,
                max_value=3,
                step=1,
                help_text="Maximum trades per day (1 = very selective)",
            ),
            ParamConfig(
                name="retest_required",
                label="Retest Required",
                param_type="bool",
                default=False,
                help_text="Wait for retest of level before entry",
            ),
            ParamConfig(
                name="stop_mode",
                label="Stop Mode",
                param_type="select",
                default="range",
                options=["range", "atr"],
                help_text="Stop at opposite range or ATR-based",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Mult",
                param_type="float",
                default=1.5,
                min_value=1.0,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiplier for stop (if ATR mode)",
            ),
        ]
    
    def _get_prev_day_levels(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate previous day's high and low."""
        if not isinstance(data.index, pd.DatetimeIndex):
            # Fallback: use 96-bar lookback (24 hours at 15min)
            lookback = 96
            prev_high = data["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
            prev_low = data["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)
            return prev_high, prev_low
        
        # Group by date
        data["_date"] = data.index.date
        
        # Calculate daily high/low
        daily_high = data.groupby("_date")["high"].transform("max")
        daily_low = data.groupby("_date")["low"].transform("min")
        
        # Shift to get previous day values
        prev_high = pd.Series(index=data.index, dtype=float)
        prev_low = pd.Series(index=data.index, dtype=float)
        
        prev_date = None
        prev_h = None
        prev_l = None
        
        for date in data["_date"].unique():
            mask = data["_date"] == date
            if prev_h is not None:
                prev_high.loc[mask] = prev_h
                prev_low.loc[mask] = prev_l
            prev_h = daily_high[mask].iloc[0]
            prev_l = daily_low[mask].iloc[0]
            prev_date = date
        
        return prev_high, prev_low
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate previous day range breakout signals."""
        # Parameters
        daily_limit = self.params.get("daily_limit", 1)
        retest_required = self.params.get("retest_required", False)
        stop_mode = self.params.get("stop_mode", "range")
        atr_stop_mult = self.params.get("atr_stop_mult", 1.5)
        
        # Calculate previous day levels
        prev_high, prev_low = self._get_prev_day_levels(data.copy())
        data["prev_high"] = prev_high
        data["prev_low"] = prev_low
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Breakout detection
        prev_close = data["close"].shift(1)
        
        # Long breakout: close above previous day high
        data["breakout_long"] = (
            (data["close"] > data["prev_high"]) &
            (prev_close <= data["prev_high"].shift(1))
        )
        
        # Short breakout: close below previous day low
        data["breakout_short"] = (
            (data["close"] < data["prev_low"]) &
            (prev_close >= data["prev_low"].shift(1))
        )
        
        # Apply daily limit
        if isinstance(data.index, pd.DatetimeIndex):
            data["_date"] = data.index.date
            
            # Track trades per day
            data["_daily_trade_count"] = 0
            current_date = None
            trade_count = 0
            
            for i in range(len(data)):
                date = data["_date"].iloc[i]
                if date != current_date:
                    current_date = date
                    trade_count = 0
                
                if data["breakout_long"].iloc[i] or data["breakout_short"].iloc[i]:
                    if trade_count < daily_limit:
                        trade_count += 1
                        data.iloc[i, data.columns.get_loc("_daily_trade_count")] = trade_count
                    else:
                        # Exceeded daily limit - disable signal
                        if data["breakout_long"].iloc[i]:
                            data.iloc[i, data.columns.get_loc("breakout_long")] = False
                        if data["breakout_short"].iloc[i]:
                            data.iloc[i, data.columns.get_loc("breakout_short")] = False
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["breakout_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        
        if stop_mode == "range":
            data.loc[long_mask, "stop_price"] = data.loc[long_mask, "prev_low"]
        else:  # atr
            data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - (
                data.loc[long_mask, "atr"] * atr_stop_mult
            )
        
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        
        if stop_mode == "range":
            data.loc[short_mask, "stop_price"] = data.loc[short_mask, "prev_high"]
        else:  # atr
            data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + (
                data.loc[short_mask, "atr"] * atr_stop_mult
            )
        
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        # Store diagnostics
        data["_diag_above_prev_high"] = data["close"] > data["prev_high"]
        data["_diag_below_prev_low"] = data["close"] < data["prev_low"]
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Prev High", "column": "prev_high", "color": "green", "style": "dashed"},
            {"name": "Prev Low", "column": "prev_low", "color": "red", "style": "dashed"},
        ]
