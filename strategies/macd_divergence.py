"""
Strategy: MACD Histogram Divergence

Trade divergences between price and MACD histogram for reversals.
Divergence is a powerful early warning of trend exhaustion.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


class MACDDivergenceStrategy(Strategy):
    """MACD histogram divergence reversal strategy."""
    
    @property
    def name(self) -> str:
        return "MACD Divergence"
    
    @property
    def description(self) -> str:
        return "Trade bullish/bearish divergences between price and MACD histogram for early reversal signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast_period",
                label="Fast EMA",
                param_type="int",
                default=12,
                min_value=8,
                max_value=20,
                step=2,
                help_text="Fast EMA period",
            ),
            ParamConfig(
                name="slow_period",
                label="Slow EMA",
                param_type="int",
                default=26,
                min_value=20,
                max_value=40,
                step=2,
                help_text="Slow EMA period",
            ),
            ParamConfig(
                name="lookback",
                label="Divergence Lookback",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=5,
                help_text="Bars to look for divergence",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD divergence signals."""
        fast = self.params.get("fast_period", 12)
        slow = self.params.get("slow_period", 26)
        lookback = self.params.get("lookback", 10)
        
        # Calculate MACD
        macd_line, signal_line, histogram = macd(data["close"], fast, slow, 9)
        data["macd"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_hist"] = histogram
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Find local lows/highs for divergence
        price_low = data["low"].rolling(lookback).min()
        price_high = data["high"].rolling(lookback).max()
        hist_low = histogram.rolling(lookback).min()
        hist_high = histogram.rolling(lookback).max()
        
        # Bullish divergence: Price makes lower low, MACD makes higher low
        prev_price_low = price_low.shift(lookback)
        prev_hist_low = hist_low.shift(lookback)
        bullish_div = (
            (data["low"] <= price_low) &  # Current low
            (data["low"] < prev_price_low) &  # Lower than previous low
            (histogram > prev_hist_low) &  # MACD higher than previous
            (histogram < 0)  # Still below zero (oversold)
        )
        
        # Bearish divergence: Price makes higher high, MACD makes lower high
        prev_price_high = price_high.shift(lookback)
        prev_hist_high = hist_high.shift(lookback)
        bearish_div = (
            (data["high"] >= price_high) &
            (data["high"] > prev_price_high) &
            (histogram < prev_hist_high) &
            (histogram > 0)
        )
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long on bullish divergence
        long_mask = bullish_div.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short on bearish divergence
        short_mask = bearish_div.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on MACD cross
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        data.loc[macd_cross_up | macd_cross_down, "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "MACD", "column": "macd", "color": "blue", "style": "solid"},
            {"name": "Signal", "column": "macd_signal", "color": "orange", "style": "dashed"},
        ]
