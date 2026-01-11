"""
Strategy: Volume-Confirmed Breakout

Low-frequency breakout strategy that requires volume confirmation.
Entry: Breakout of prior day high/low or Donchian with above-average volume
Exit: ATR trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, donchian_channels, sma


class VolumeBreakoutStrategy(Strategy):
    """Volume-confirmed breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Volume Breakout"
    
    @property
    def description(self) -> str:
        return "Breakout strategy requiring volume confirmation. Only enters when volume exceeds SMA*multiplier."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="breakout_source",
                label="Breakout Source",
                param_type="select",
                default="donchian",
                options=["donchian", "prior_day"],
                help_text="Source for breakout levels",
            ),
            ParamConfig(
                name="donchian_period",
                label="Donchian Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=30,
                step=5,
                help_text="Period for Donchian channel (if used)",
            ),
            ParamConfig(
                name="vol_lookback",
                label="Volume Lookback",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="Lookback for volume SMA",
            ),
            ParamConfig(
                name="vol_mult",
                label="Volume Multiplier",
                param_type="float",
                default=1.5,
                min_value=1.2,
                max_value=2.5,
                step=0.1,
                help_text="Volume must exceed SMA * this multiplier",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiplier for trailing stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-confirmed breakout signals."""
        # Parameters
        breakout_source = self.params.get("breakout_source", "donchian")
        donchian_period = self.params.get("donchian_period", 20)
        vol_lookback = self.params.get("vol_lookback", 20)
        vol_mult = self.params.get("vol_mult", 1.5)
        atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Calculate breakout levels
        if breakout_source == "donchian":
            dc_upper, dc_lower, dc_mid = donchian_channels(data["high"], data["low"], donchian_period)
            data["level_high"] = dc_upper.shift(1)  # Use prior bar to avoid lookahead
            data["level_low"] = dc_lower.shift(1)
        else:  # prior_day
            # Calculate prior day high/low
            if isinstance(data.index, pd.DatetimeIndex):
                data["date"] = data.index.date
                daily_high = data.groupby("date")["high"].transform("max")
                daily_low = data.groupby("date")["low"].transform("min")
                # Shift to get prior day values
                data["level_high"] = daily_high.shift(1).ffill()
                data["level_low"] = daily_low.shift(1).ffill()
            else:
                # Fallback to donchian
                dc_upper, dc_lower, dc_mid = donchian_channels(data["high"], data["low"], donchian_period)
                data["level_high"] = dc_upper.shift(1)
                data["level_low"] = dc_lower.shift(1)
        
        # Volume filter
        data["vol_sma"] = sma(data["volume"], vol_lookback)
        data["vol_threshold"] = data["vol_sma"] * vol_mult
        data["vol_confirmed"] = data["volume"] > data["vol_threshold"]
        
        # Breakout detection with volume confirmation
        prev_close = data["close"].shift(1)
        
        # Long: Close breaks above level high with volume confirmation
        data["breakout_long"] = (
            (data["close"] > data["level_high"]) &
            (prev_close <= data["level_high"].shift(1)) &
            data["vol_confirmed"]
        )
        
        # Short: Close breaks below level low with volume confirmation
        data["breakout_short"] = (
            (data["close"] < data["level_low"]) &
            (prev_close >= data["level_low"].shift(1)) &
            data["vol_confirmed"]
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["breakout_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - (
            data.loc[long_mask, "atr"] * atr_stop_mult
        )
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + (
            data.loc[short_mask, "atr"] * atr_stop_mult
        )
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Level High", "column": "level_high", "color": "green", "style": "dashed"},
            {"name": "Level Low", "column": "level_low", "color": "red", "style": "dashed"},
        ]
