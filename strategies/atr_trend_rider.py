"""
Strategy: ATR Trend Rider

Ride trends using ATR bands as dynamic support/resistance.
Stay in trends longer with wider trailing stops.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class ATRTrendRiderStrategy(Strategy):
    """ATR trend riding strategy."""
    
    @property
    def name(self) -> str:
        return "ATR Trend Rider"
    
    @property
    def description(self) -> str:
        return "Ride trends using ATR bands with wide trailing stops for longer holds."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="atr_mult",
                label="ATR Multiplier",
                param_type="float",
                default=2.5,
                min_value=1.5,
                max_value=4.0,
                step=0.5,
                help_text="ATR multiplier for bands",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ATR trend rider signals."""
        atr_mult = self.params.get("atr_mult", 2.5)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema20"] = ema(data["close"], 20)
        
        # ATR bands around EMA
        data["upper_band"] = data["ema20"] + data["atr"] * atr_mult
        data["lower_band"] = data["ema20"] - data["atr"] * atr_mult
        
        # Trend direction: above/below EMA
        above_ema = data["close"] > data["ema20"]
        below_ema = data["close"] < data["ema20"]
        prev_above = above_ema.shift(1).fillna(False)
        prev_below = below_ema.shift(1).fillna(False)
        
        # Entry on EMA cross
        long_cond = above_ema & (~prev_above)
        short_cond = below_ema & (~prev_below)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "lower_band"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "upper_band"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.5
        
        # Exit only on band break (very wide)
        data.loc[(data["close"] < data["lower_band"]).fillna(False), "exit_signal"] = True
        data.loc[(data["close"] > data["upper_band"]).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 20", "column": "ema20", "color": "blue", "style": "solid"},
            {"name": "Upper", "column": "upper_band", "color": "red", "style": "dashed"},
            {"name": "Lower", "column": "lower_band", "color": "green", "style": "dashed"},
        ]
