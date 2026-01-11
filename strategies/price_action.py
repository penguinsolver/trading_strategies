"""
Strategy: Price Action Reversal

Pure price action: engulfing patterns at key levels.
No indicators, just candle patterns with context.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class PriceActionReversalStrategy(Strategy):
    """Price action engulfing reversal strategy."""
    
    @property
    def name(self) -> str:
        return "Price Action"
    
    @property
    def description(self) -> str:
        return "Trade bullish/bearish engulfing patterns at swing highs/lows."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="swing_lookback",
                label="Swing Lookback",
                param_type="int",
                default=5,
                min_value=3,
                max_value=10,
                step=2,
                help_text="Bars to identify swing high/low",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price action signals."""
        lookback = self.params.get("swing_lookback", 5)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Identify engulfing patterns
        bullish_engulf = (
            (data["close"] > data["open"]) &  # Current green
            (data["close"].shift(1) < data["open"].shift(1)) &  # Prev red
            (data["close"] > data["open"].shift(1)) &  # Body engulfs
            (data["open"] < data["close"].shift(1))
        )
        
        bearish_engulf = (
            (data["close"] < data["open"]) &  # Current red
            (data["close"].shift(1) > data["open"].shift(1)) &  # Prev green
            (data["close"] < data["open"].shift(1)) &  # Body engulfs
            (data["open"] > data["close"].shift(1))
        )
        
        # At swing low/high
        swing_low = data["low"] == data["low"].rolling(lookback * 2 + 1, center=True).min()
        swing_high = data["high"] == data["high"].rolling(lookback * 2 + 1, center=True).max()
        
        # Use shifted swing levels to avoid lookahead
        at_support = data["low"] <= data["low"].rolling(lookback).min().shift(1)
        at_resistance = data["high"] >= data["high"].rolling(lookback).max().shift(1)
        
        long_cond = bullish_engulf & at_support
        short_cond = bearish_engulf & at_resistance
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit after 5 bars or opposite signal
        # Simple time-based exit
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
