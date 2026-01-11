"""
Strategy: ATR Channel Breakout

Low-frequency breakout strategy with volatility expansion filter.
Entry: Close breaks beyond ATR channel when ATR% is expanding
Exit: Opposite channel or ATR trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class ATRChannelStrategy(Strategy):
    """ATR Channel breakout with volatility filter."""
    
    @property
    def name(self) -> str:
        return "ATR Channel"
    
    @property
    def description(self) -> str:
        return "Breakout of ATR-based channel with volatility expansion filter. Trades only when volatility is confirming."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="ema_period",
                label="EMA Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="EMA period for channel center",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=21,
                step=1,
                help_text="ATR calculation period",
            ),
            ParamConfig(
                name="atr_mult",
                label="ATR Multiplier",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiplier for channel width",
            ),
            ParamConfig(
                name="atr_pct_threshold",
                label="ATR% Threshold",
                param_type="float",
                default=2.0,
                min_value=1.0,
                max_value=4.0,
                step=0.5,
                help_text="Minimum ATR% for volatility expansion confirmation",
            ),
            ParamConfig(
                name="exit_mode",
                label="Exit Mode",
                param_type="select",
                default="trail",
                options=["channel", "trail"],
                help_text="Exit on opposite channel or ATR trailing stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ATR channel breakout signals."""
        # Parameters
        ema_period = self.params.get("ema_period", 20)
        atr_period = self.params.get("atr_period", 14)
        atr_mult = self.params.get("atr_mult", 2.0)
        atr_pct_threshold = self.params.get("atr_pct_threshold", 2.0)
        exit_mode = self.params.get("exit_mode", "trail")
        
        # Calculate EMA center line
        data["ema"] = ema(data["close"], ema_period)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Calculate ATR as percentage of price
        data["atr_pct"] = (data["atr"] / data["close"]) * 100
        
        # Calculate channel bands
        data["upper_channel"] = data["ema"] + (data["atr"] * atr_mult)
        data["lower_channel"] = data["ema"] - (data["atr"] * atr_mult)
        
        # Volatility filter: ATR% must exceed threshold
        data["vol_expanding"] = data["atr_pct"] > atr_pct_threshold
        
        # Breakout detection (use shifted values to avoid lookahead)
        prev_upper = data["upper_channel"].shift(1)
        prev_lower = data["lower_channel"].shift(1)
        prev_close = data["close"].shift(1)
        
        # Long: Close breaks above upper channel with volatility confirmation
        data["breakout_long"] = (
            (data["close"] > prev_upper) &
            (prev_close <= prev_upper.shift(1)) &
            data["vol_expanding"]
        )
        
        # Short: Close breaks below lower channel with volatility confirmation
        data["breakout_short"] = (
            (data["close"] < prev_lower) &
            (prev_close >= prev_lower.shift(1)) &
            data["vol_expanding"]
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["breakout_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "lower_channel"]
        
        if exit_mode == "trail":
            data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_mult
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "upper_channel"]
        
        if exit_mode == "trail":
            data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_mult
        
        # Exit signals for channel mode
        if exit_mode == "channel":
            # Long exit: price touches lower channel
            data.loc[data["low"] <= data["lower_channel"], "exit_signal"] = True
            # Short exit: price touches upper channel  
            data.loc[data["high"] >= data["upper_channel"], "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA", "column": "ema", "color": "blue", "style": "solid"},
            {"name": "Upper Channel", "column": "upper_channel", "color": "green", "style": "dashed"},
            {"name": "Lower Channel", "column": "lower_channel", "color": "red", "style": "dashed"},
        ]
