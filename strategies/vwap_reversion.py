"""
Strategy C: VWAP Mean Reversion

Only trades in ranging conditions (low ADX).
Entry: Price extends from VWAP with RSI confirmation.
Target: Return to VWAP (mean reversion).
Stop: Fixed ATR-based stop.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr, vwap
from indicators.trend import rsi, adx


class VWAPReversionStrategy(Strategy):
    """VWAP mean reversion strategy for ranging markets."""
    
    @property
    def name(self) -> str:
        return "VWAP Reversion"
    
    @property
    def description(self) -> str:
        return "Mean reversion to VWAP. Active only in low-volatility ranging conditions."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="vwap_stdev_entry",
                label="VWAP Deviation (ATR mult)",
                param_type="float",
                default=1.5,
                min_value=0.5,
                max_value=3.0,
                step=0.25,
                help_text="Distance from VWAP (in ATR) to trigger entry",
            ),
            ParamConfig(
                name="rsi_period",
                label="RSI Period",
                param_type="int",
                default=14,
                min_value=5,
                max_value=30,
                step=1,
                help_text="RSI period for overbought/oversold confirmation",
            ),
            ParamConfig(
                name="rsi_oversold",
                label="RSI Oversold",
                param_type="int",
                default=35,
                min_value=20,
                max_value=45,
                step=5,
                help_text="RSI level for oversold (long entries)",
            ),
            ParamConfig(
                name="rsi_overbought",
                label="RSI Overbought",
                param_type="int",
                default=65,
                min_value=55,
                max_value=80,
                step=5,
                help_text="RSI level for overbought (short entries)",
            ),
            ParamConfig(
                name="adx_max",
                label="Max ADX (range filter)",
                param_type="int",
                default=25,
                min_value=15,
                max_value=40,
                step=5,
                help_text="Maximum ADX to confirm ranging conditions",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="Stop ATR Multiplier",
                param_type="float",
                default=1.5,
                min_value=0.5,
                max_value=3.0,
                step=0.25,
                help_text="ATR multiplier for stop distance",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP reversion signals."""
        # Parameters
        vwap_deviation = self.params.get("vwap_stdev_entry", 1.5)
        rsi_period = self.params.get("rsi_period", 14)
        rsi_oversold = self.params.get("rsi_oversold", 35)
        rsi_overbought = self.params.get("rsi_overbought", 65)
        adx_max = self.params.get("adx_max", 25)
        atr_stop_mult = self.params.get("atr_stop_mult", 1.5)
        
        # Calculate indicators
        data["vwap"] = vwap(data)
        data["rsi"] = rsi(data["close"], rsi_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # ADX for ranging filter
        adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], 14)
        data["adx"] = adx_val
        
        # Calculate distance from VWAP in ATR units
        data["vwap_distance"] = (data["close"] - data["vwap"]) / data["atr"]
        
        # Ranging condition
        data["is_ranging"] = data["adx"] < adx_max
        
        # Entry conditions
        data["long_setup"] = (
            data["is_ranging"] &
            (data["vwap_distance"] < -vwap_deviation) &  # Below VWAP
            (data["rsi"] < rsi_oversold)  # Oversold
        )
        
        data["short_setup"] = (
            data["is_ranging"] &
            (data["vwap_distance"] > vwap_deviation) &  # Above VWAP
            (data["rsi"] > rsi_overbought)  # Overbought
        )
        
        # Entry trigger: first green/red candle after setup
        data["long_trigger"] = (
            data["long_setup"].shift(1) &
            (data["close"] > data["open"])  # Green candle
        )
        
        data["short_trigger"] = (
            data["short_setup"].shift(1) &
            (data["close"] < data["open"])  # Red candle
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["target_price"] = np.nan
        
        # Generate signals
        for i in range(len(data)):
            if data["long_trigger"].iloc[i]:
                data.loc[data.index[i], "entry_signal"] = 1
                # Stop below entry by ATR
                stop = data["close"].iloc[i] - data["atr"].iloc[i] * atr_stop_mult
                data.loc[data.index[i], "stop_price"] = stop
                # Target: VWAP (mean reversion target)
                data.loc[data.index[i], "target_price"] = data["vwap"].iloc[i]
                
            elif data["short_trigger"].iloc[i]:
                data.loc[data.index[i], "entry_signal"] = -1
                # Stop above entry by ATR
                stop = data["close"].iloc[i] + data["atr"].iloc[i] * atr_stop_mult
                data.loc[data.index[i], "stop_price"] = stop
                # Target: VWAP
                data.loc[data.index[i], "target_price"] = data["vwap"].iloc[i]
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "VWAP", "column": "vwap", "color": "purple", "style": "solid"},
        ]
