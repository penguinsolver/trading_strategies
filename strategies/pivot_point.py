"""
Strategy: Pivot Point Bounce

Trade bounces off classic pivot point levels (S1, S2, R1, R2).
Pivot points are widely watched support/resistance levels.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class PivotPointStrategy(Strategy):
    """Pivot point bounce strategy."""
    
    @property
    def name(self) -> str:
        return "Pivot Bounce"
    
    @property
    def description(self) -> str:
        return "Trade bounces off classic pivot S/R levels with momentum confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="tolerance",
                label="Touch Tolerance (%)",
                param_type="float",
                default=0.1,
                min_value=0.05,
                max_value=0.3,
                step=0.05,
                help_text="% within pivot level to trigger",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pivot point signals."""
        tolerance = self.params.get("tolerance", 0.1) / 100
        
        # Calculate daily pivots (use prior day's OHLC)
        data["date"] = data.index.date
        daily = data.groupby("date").agg({
            "high": "max",
            "low": "min", 
            "close": "last"
        }).shift(1)
        
        # Map back to original data
        data["prev_high"] = data["date"].map(daily["high"])
        data["prev_low"] = data["date"].map(daily["low"])
        data["prev_close"] = data["date"].map(daily["close"])
        
        # Classic pivot formula
        data["pivot"] = (data["prev_high"] + data["prev_low"] + data["prev_close"]) / 3
        data["r1"] = 2 * data["pivot"] - data["prev_low"]
        data["s1"] = 2 * data["pivot"] - data["prev_high"]
        data["r2"] = data["pivot"] + (data["prev_high"] - data["prev_low"])
        data["s2"] = data["pivot"] - (data["prev_high"] - data["prev_low"])
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Bounce conditions
        near_s1 = (data["low"] <= data["s1"] * (1 + tolerance)) & (data["low"] >= data["s1"] * (1 - tolerance))
        near_s2 = (data["low"] <= data["s2"] * (1 + tolerance)) & (data["low"] >= data["s2"] * (1 - tolerance))
        near_r1 = (data["high"] >= data["r1"] * (1 - tolerance)) & (data["high"] <= data["r1"] * (1 + tolerance))
        near_r2 = (data["high"] >= data["r2"] * (1 - tolerance)) & (data["high"] <= data["r2"] * (1 + tolerance))
        
        # Bullish bounce off support (close above open for confirmation)
        bullish_candle = data["close"] > data["open"]
        bearish_candle = data["close"] < data["open"]
        
        long_cond = (near_s1 | near_s2) & bullish_candle
        short_cond = (near_r1 | near_r2) & bearish_candle
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit at pivot
        data.loc[data["close"] > data["pivot"], "exit_signal"] = True  # Exit shorts
        data.loc[data["close"] < data["pivot"], "exit_signal"] = True  # Exit longs
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Pivot", "column": "pivot", "color": "blue", "style": "solid"},
            {"name": "R1", "column": "r1", "color": "red", "style": "dashed"},
            {"name": "S1", "column": "s1", "color": "green", "style": "dashed"},
        ]
