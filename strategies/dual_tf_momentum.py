"""
Strategy: Dual Timeframe Momentum

Use higher timeframe trend with lower timeframe entry.
Improved by using rolling calculations to simulate higher TF.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class DualTimeframeMomentumStrategy(Strategy):
    """Dual timeframe momentum strategy."""
    
    @property
    def name(self) -> str:
        return "Dual TF Momentum"
    
    @property
    def description(self) -> str:
        return "Use higher timeframe trend direction with lower timeframe momentum entries."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate dual TF signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Simulate higher TF with longer EMA (4x = like 1h on 15m)
        data["htf_ema"] = ema(data["close"], 80)  # Higher timeframe trend
        data["ltf_ema"] = ema(data["close"], 20)  # Lower timeframe
        
        # HTF trend
        htf_up = data["close"] > data["htf_ema"]
        htf_down = data["close"] < data["htf_ema"]
        
        # LTF momentum
        ltf_cross_up = (data["close"] > data["ltf_ema"]) & (data["close"].shift(1) <= data["ltf_ema"].shift(1))
        ltf_cross_down = (data["close"] < data["ltf_ema"]) & (data["close"].shift(1) >= data["ltf_ema"].shift(1))
        
        # Entry: LTF cross in direction of HTF trend
        long_cond = ltf_cross_up & htf_up
        short_cond = ltf_cross_down & htf_down
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ltf_ema"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ltf_ema"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.0
        
        # Exit on HTF trend change
        htf_change = (htf_up & (~htf_up.shift(1).fillna(False))) | (htf_down & (~htf_down.shift(1).fillna(False)))
        data.loc[htf_change.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "HTF EMA", "column": "htf_ema", "color": "red", "style": "solid"},
            {"name": "LTF EMA", "column": "ltf_ema", "color": "blue", "style": "solid"},
        ]
