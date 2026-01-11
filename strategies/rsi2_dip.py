"""
Strategy: RSI-2 Dip Buy

Mean reversion strategy that buys dips / sells rips in the direction of the trend.
Entry: RSI(2) extreme reading in direction of trend (EMA filter)
Exit: RSI returns to neutral OR time-based exit
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, rsi, atr


class RSI2DipStrategy(Strategy):
    """RSI-2 mean reversion within trend strategy."""
    
    @property
    def name(self) -> str:
        return "RSI-2 Dip"
    
    @property
    def description(self) -> str:
        return "Mean reversion dip buying in trending markets. Uses RSI(2) extremes with EMA trend filter."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="ema_period",
                label="EMA Trend Period",
                param_type="int",
                default=200,
                min_value=100,
                max_value=300,
                step=50,
                help_text="EMA period for trend filter",
            ),
            ParamConfig(
                name="rsi_period",
                label="RSI Period",
                param_type="int",
                default=2,
                min_value=2,
                max_value=5,
                step=1,
                help_text="RSI calculation period",
            ),
            ParamConfig(
                name="rsi_oversold",
                label="RSI Oversold",
                param_type="int",
                default=10,
                min_value=5,
                max_value=25,
                step=5,
                help_text="RSI oversold threshold for long entries",
            ),
            ParamConfig(
                name="rsi_overbought",
                label="RSI Overbought",
                param_type="int",
                default=90,
                min_value=75,
                max_value=95,
                step=5,
                help_text="RSI overbought threshold for short entries",
            ),
            ParamConfig(
                name="rsi_exit",
                label="RSI Exit Level",
                param_type="int",
                default=50,
                min_value=40,
                max_value=60,
                step=5,
                help_text="RSI level to exit position",
            ),
            ParamConfig(
                name="time_exit_bars",
                label="Time Exit (Bars)",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=1,
                help_text="Maximum bars to hold position",
            ),
            ParamConfig(
                name="long_only",
                label="Long Only",
                param_type="bool",
                default=True,
                help_text="Only take long positions (dip buying)",
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
                default=2.0,
                min_value=1.0,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiple for stop loss",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-2 dip buy signals."""
        # Parameters
        ema_period = self.params.get("ema_period", 200)
        rsi_period = self.params.get("rsi_period", 2)
        rsi_oversold = self.params.get("rsi_oversold", 10)
        rsi_overbought = self.params.get("rsi_overbought", 90)
        rsi_exit = self.params.get("rsi_exit", 50)
        time_exit_bars = self.params.get("time_exit_bars", 10)
        long_only = self.params.get("long_only", True)
        atr_period = self.params.get("atr_period", 14)
        atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        
        # Calculate indicators
        data["ema_trend"] = ema(data["close"], ema_period)
        data["rsi"] = rsi(data["close"], rsi_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Trend filter
        data["uptrend"] = data["close"] > data["ema_trend"]
        data["downtrend"] = data["close"] < data["ema_trend"]
        
        # Entry conditions
        # Long: Uptrend + RSI oversold
        data["long_setup"] = data["uptrend"] & (data["rsi"] < rsi_oversold)
        
        # Short: Downtrend + RSI overbought (if not long_only)
        if not long_only:
            data["short_setup"] = data["downtrend"] & (data["rsi"] > rsi_overbought)
        else:
            data["short_setup"] = False
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Generate entry signals
        # Long entries
        long_mask = data["long_setup"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - (
            data.loc[long_mask, "atr"] * atr_stop_mult
        )
        
        # Short entries (if enabled)
        if not long_only:
            short_mask = data["short_setup"].fillna(False)
            data.loc[short_mask, "entry_signal"] = -1
            data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + (
                data.loc[short_mask, "atr"] * atr_stop_mult
            )
        
        # Exit signals based on RSI returning to exit level
        # This is handled by the backtest engine's exit_signal
        # Long exit: RSI crosses above exit level
        data["long_exit"] = (data["rsi"] > rsi_exit) & (data["rsi"].shift(1) <= rsi_exit)
        # Short exit: RSI crosses below exit level
        data["short_exit"] = (data["rsi"] < rsi_exit) & (data["rsi"].shift(1) >= rsi_exit)
        
        data.loc[data["long_exit"] | data["short_exit"], "exit_signal"] = True
        
        # Note: Time-based exit would need to be handled in backtest engine
        # For now, we rely on the RSI exit and stop loss
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA Trend", "column": "ema_trend", "color": "blue", "style": "solid"},
        ]
