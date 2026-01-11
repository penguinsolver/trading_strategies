"""
Strategy: OBV Divergence

On-Balance Volume divergence strategy.
OBV tracks cumulative volume flow, divergences signal reversals.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    price_change = close.diff()
    volume_direction = np.where(price_change > 0, volume, 
                                np.where(price_change < 0, -volume, 0))
    obv_values = pd.Series(volume_direction, index=close.index).cumsum()
    return obv_values


class OBVDivergenceStrategy(Strategy):
    """OBV divergence strategy."""
    
    @property
    def name(self) -> str:
        return "OBV Divergence"
    
    @property
    def description(self) -> str:
        return "Trade divergences between price and On-Balance Volume for early reversal signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="lookback",
                label="Divergence Lookback",
                param_type="int",
                default=14,
                min_value=7,
                max_value=28,
                step=7,
                help_text="Bars to detect divergence",
            ),
            ParamConfig(
                name="ema_period",
                label="OBV EMA",
                param_type="int",
                default=20,
                min_value=10,
                max_value=30,
                step=5,
                help_text="EMA to smooth OBV",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate OBV divergence signals."""
        lookback = self.params.get("lookback", 14)
        ema_period = self.params.get("ema_period", 20)
        
        # Calculate OBV
        data["obv"] = obv(data["close"], data["volume"])
        data["obv_ema"] = ema(data["obv"], ema_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Find local extremes for divergence
        price_low = data["low"].rolling(lookback).min()
        price_high = data["high"].rolling(lookback).max()
        obv_low = data["obv"].rolling(lookback).min()
        obv_high = data["obv"].rolling(lookback).max()
        
        prev_price_low = price_low.shift(lookback)
        prev_price_high = price_high.shift(lookback)
        prev_obv_low = obv_low.shift(lookback)
        prev_obv_high = obv_high.shift(lookback)
        
        # Bullish divergence: Price lower low, OBV higher low
        bullish_div = (
            (data["low"] <= price_low) &
            (data["low"] < prev_price_low) &
            (data["obv"] > prev_obv_low)
        )
        
        # Bearish divergence: Price higher high, OBV lower high
        bearish_div = (
            (data["high"] >= price_high) &
            (data["high"] > prev_price_high) &
            (data["obv"] < prev_obv_high)
        )
        
        # Confirmation: OBV above/below its EMA
        obv_bullish = data["obv"] > data["obv_ema"]
        obv_bearish = data["obv"] < data["obv_ema"]
        
        # Combined signals
        long_cond = bullish_div & obv_bullish
        short_cond = bearish_div & obv_bearish
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on OBV EMA cross
        prev_obv = data["obv"].shift(1)
        prev_obv_ema = data["obv_ema"].shift(1)
        obv_cross_down = (data["obv"] < data["obv_ema"]) & (prev_obv >= prev_obv_ema)
        obv_cross_up = (data["obv"] > data["obv_ema"]) & (prev_obv <= prev_obv_ema)
        data.loc[(obv_cross_down | obv_cross_up).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "OBV", "column": "obv", "color": "green", "style": "solid"},
            {"name": "OBV EMA", "column": "obv_ema", "color": "orange", "style": "dashed"},
        ]
