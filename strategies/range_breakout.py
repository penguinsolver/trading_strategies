"""
Strategy: Range Breakout Momentum

Trade breakouts from consolidation ranges with momentum confirmation.
Waits for price to break and hold above range highs.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class RangeBreakoutMomentumStrategy(Strategy):
    """Breakout from consolidation with momentum."""
    
    @property
    def name(self) -> str:
        return "Range Breakout"
    
    @property
    def description(self) -> str:
        return "Trade breakouts from tight consolidation ranges with momentum confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="range_period",
                label="Range Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=40,
                step=5,
                help_text="Bars to measure range",
            ),
            ParamConfig(
                name="breakout_mult",
                label="Breakout Threshold",
                param_type="float",
                default=0.5,
                min_value=0.25,
                max_value=1.0,
                step=0.25,
                help_text="ATR multiplier for breakout threshold",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate range breakout signals."""
        range_period = self.params.get("range_period", 20)
        breakout_mult = self.params.get("breakout_mult", 0.5)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Calculate range
        data["range_high"] = data["high"].rolling(range_period).max()
        data["range_low"] = data["low"].rolling(range_period).min()
        data["range_mid"] = (data["range_high"] + data["range_low"]) / 2
        
        # Breakout threshold
        threshold = data["atr"] * breakout_mult
        
        # Shift to avoid lookahead
        prev_high = data["range_high"].shift(1)
        prev_low = data["range_low"].shift(1)
        
        # Breakout with hold: close above/below range for 2 bars
        breakout_up = (data["close"] > prev_high + threshold) & (data["close"].shift(1) > prev_high.shift(1))
        breakout_down = (data["close"] < prev_low - threshold) & (data["close"].shift(1) < prev_low.shift(1))
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = breakout_up.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "range_mid"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = breakout_down.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "range_mid"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on reversal back into range
        back_in = (data["close"] < prev_high) & (data["close"] > prev_low)
        data.loc[back_in.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Range High", "column": "range_high", "color": "red", "style": "dashed"},
            {"name": "Range Low", "column": "range_low", "color": "green", "style": "dashed"},
        ]
