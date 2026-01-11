"""
Strategy: Bounce from Low

Mean reversion: buy the dip when price is below lower band.
Simple but can work in ranging markets.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, sma


class BounceFromLowStrategy(Strategy):
    """Bounce from low strategy."""
    
    @property
    def name(self) -> str:
        return "Bounce Low"
    
    @property
    def description(self) -> str:
        return "Mean reversion: buy when price near N-bar low, sell at mean."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="lookback",
                label="Lookback",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=10,
                help_text="Bars for high/low",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate bounce signals."""
        lookback = self.params.get("lookback", 20)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["range_high"] = data["high"].rolling(lookback).max()
        data["range_low"] = data["low"].rolling(lookback).min()
        data["range_mid"] = (data["range_high"] + data["range_low"]) / 2
        
        # Near low (within 25% of range from bottom)
        range_size = data["range_high"] - data["range_low"]
        near_low = data["close"] < data["range_low"] + range_size * 0.25
        near_high = data["close"] > data["range_high"] - range_size * 0.25
        
        # Bullish bar at low
        bullish = data["close"] > data["open"]
        bearish = data["close"] < data["open"]
        
        long_cond = near_low & bullish
        short_cond = near_high & bearish
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "range_low"] - data.loc[long_mask, "atr"] * 0.25
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "range_high"] + data.loc[short_mask, "atr"] * 0.25
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit at mid
        at_mid = (data["close"] > data["range_mid"] - data["atr"] * 0.25) & (data["close"] < data["range_mid"] + data["atr"] * 0.25)
        data.loc[at_mid.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Range High", "column": "range_high", "color": "red", "style": "dashed"},
            {"name": "Range Low", "column": "range_low", "color": "green", "style": "dashed"},
        ]
