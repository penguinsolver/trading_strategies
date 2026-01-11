"""
Strategy: RSI Divergence

Trade RSI divergences with price for reversal signals.
Similar to MACD divergence but using RSI.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


class RSIDivergenceStrategy(Strategy):
    """RSI divergence strategy."""
    
    @property
    def name(self) -> str:
        return "RSI Divergence"
    
    @property
    def description(self) -> str:
        return "Trade RSI-price divergences for early reversal signals at extremes."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="RSI Period",
                param_type="int",
                default=14,
                min_value=7,
                max_value=21,
                step=2,
                help_text="RSI calculation period",
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
        """Generate RSI divergence signals."""
        period = self.params.get("period", 14)
        lookback = self.params.get("lookback", 10)
        
        # Calculate RSI
        data["rsi"] = rsi(data["close"], period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Find lows/highs
        price_low = data["low"].rolling(lookback).min()
        price_high = data["high"].rolling(lookback).max()
        rsi_low = data["rsi"].rolling(lookback).min()
        rsi_high = data["rsi"].rolling(lookback).max()
        
        prev_price_low = price_low.shift(lookback)
        prev_rsi_low = rsi_low.shift(lookback)
        prev_price_high = price_high.shift(lookback)
        prev_rsi_high = rsi_high.shift(lookback)
        
        # Bullish divergence: lower price low, higher RSI low
        bullish_div = (
            (data["low"] <= price_low) &
            (data["low"] < prev_price_low) &
            (data["rsi"] > prev_rsi_low) &
            (data["rsi"] < 40)
        )
        
        # Bearish divergence: higher price high, lower RSI high
        bearish_div = (
            (data["high"] >= price_high) &
            (data["high"] > prev_price_high) &
            (data["rsi"] < prev_rsi_high) &
            (data["rsi"] > 60)
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
        
        # Exit at RSI neutral
        exit_cond = (data["rsi"] > 45) & (data["rsi"] < 55)
        data.loc[exit_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "RSI", "column": "rsi", "color": "purple", "style": "solid"},
        ]
