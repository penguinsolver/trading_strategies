"""
Strategy: Close to Close Momentum

Simple but effective: trade based on close-to-close returns.
If last N closes are all up, momentum is strong.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class CloseToCloseMomentumStrategy(Strategy):
    """Close-to-close momentum strategy."""
    
    @property
    def name(self) -> str:
        return "C2C Momentum"
    
    @property
    def description(self) -> str:
        return "Trade when close-to-close returns show consistent direction (3+ bars)."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="streak",
                label="Streak Length",
                param_type="int",
                default=3,
                min_value=2,
                max_value=5,
                step=1,
                help_text="Consecutive closes in same direction",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate close-to-close signals."""
        streak = self.params.get("streak", 3)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Close changes
        data["up"] = data["close"] > data["close"].shift(1)
        data["down"] = data["close"] < data["close"].shift(1)
        
        # Count consecutive
        up_streak = data["up"].rolling(streak).sum() == streak
        down_streak = data["down"].rolling(streak).sum() == streak
        
        # Entry on streak completion
        prev_up_streak = up_streak.shift(1).fillna(False)
        prev_down_streak = down_streak.shift(1).fillna(False)
        
        long_cond = up_streak & (~prev_up_streak)
        short_cond = down_streak & (~prev_down_streak)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"].rolling(streak).min()
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"].rolling(streak).max()
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit on reversal bar
        data.loc[data["down"].fillna(False), "exit_signal"] = True  # Exit longs
        data.loc[data["up"].fillna(False), "exit_signal"] = True   # Exit shorts
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
