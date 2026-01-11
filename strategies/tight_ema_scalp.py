"""
Strategy: Tight EMA Scalper

Fast EMA cross with very tight stops for scalping.
Lower win rate but higher reward when right.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr
from indicators.moving_averages import crossover, crossunder


class TightEMAScalperStrategy(Strategy):
    """Fast EMA scalping with aggressive entries."""
    
    @property
    def name(self) -> str:
        return "Tight EMA Scalp"
    
    @property
    def description(self) -> str:
        return "Fast 5/13 EMA cross with tight ATR stops for quick scalping profits."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast",
                label="Fast EMA",
                param_type="int",
                default=5,
                min_value=3,
                max_value=8,
                step=1,
                help_text="Very fast EMA",
            ),
            ParamConfig(
                name="slow",
                label="Slow EMA",
                param_type="int",
                default=13,
                min_value=10,
                max_value=20,
                step=2,
                help_text="Slow EMA",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate tight EMA scalp signals."""
        fast = self.params.get("fast", 5)
        slow = self.params.get("slow", 13)
        
        data["ema_fast"] = ema(data["close"], fast)
        data["ema_slow"] = ema(data["close"], slow)
        data["atr"] = atr(data["high"], data["low"], data["close"], 10)  # Faster ATR
        
        golden = crossover(data["ema_fast"], data["ema_slow"])
        death = crossunder(data["ema_fast"], data["ema_slow"])
        
        # Momentum filter: close must be moving in direction
        bullish_momentum = data["close"] > data["close"].shift(2)
        bearish_momentum = data["close"] < data["close"].shift(2)
        
        long_cond = golden & bullish_momentum
        short_cond = death & bearish_momentum
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Very tight stops (0.75 ATR)
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 0.75
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 0.75
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit on opposite cross
        data.loc[death.fillna(False), "exit_signal"] = True
        data.loc[golden.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Fast EMA", "column": "ema_fast", "color": "green", "style": "solid"},
            {"name": "Slow EMA", "column": "ema_slow", "color": "red", "style": "solid"},
        ]
