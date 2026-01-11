"""
Strategy: VWAP Bounce

Trade bounces off VWAP as support/resistance.
VWAP is a key institutional level.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, vwap


class VWAPBounceStrategy(Strategy):
    """VWAP bounce strategy."""
    
    @property
    def name(self) -> str:
        return "VWAP Bounce"
    
    @property
    def description(self) -> str:
        return "Trade bounces off VWAP as support/resistance with tight stops."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="tolerance",
                label="Touch Tolerance (ATR)",
                param_type="float",
                default=0.25,
                min_value=0.1,
                max_value=0.5,
                step=0.1,
                help_text="How close to VWAP to trigger",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP bounce signals."""
        tolerance = self.params.get("tolerance", 0.25)
        
        data["vwap"] = vwap(data)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        touch_zone = data["atr"] * tolerance
        
        # Near VWAP
        near_vwap = (data["low"] <= data["vwap"] + touch_zone) & (data["high"] >= data["vwap"] - touch_zone)
        
        # Bounce up: touched VWAP from above, closed above
        bounce_up = near_vwap & (data["close"] > data["vwap"]) & (data["close"] > data["open"])
        
        # Bounce down: touched VWAP from below, closed below
        bounce_down = near_vwap & (data["close"] < data["vwap"]) & (data["close"] < data["open"])
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = bounce_up.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "vwap"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = bounce_down.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "vwap"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit on break through VWAP
        break_down = data["close"] < data["vwap"] - data["atr"]
        break_up = data["close"] > data["vwap"] + data["atr"]
        data.loc[(break_down | break_up).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "VWAP", "column": "vwap", "color": "purple", "style": "solid"}]
