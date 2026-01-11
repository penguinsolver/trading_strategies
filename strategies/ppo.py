"""
Strategy: PPO (Percentage Price Oscillator)

Similar to MACD but shown as percentage, making it easier to compare.
Trade PPO signal line crosses.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def ppo(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate PPO and signal."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    ppo_val = ((ema_fast - ema_slow) / ema_slow) * 100
    signal_line = ema(ppo_val, signal)
    histogram = ppo_val - signal_line
    
    return ppo_val, signal_line, histogram


class PPOStrategy(Strategy):
    """PPO strategy."""
    
    @property
    def name(self) -> str:
        return "PPO"
    
    @property
    def description(self) -> str:
        return "Trade PPO signal line crosses for percentage-normalized momentum."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast",
                label="Fast Period",
                param_type="int",
                default=12,
                min_value=8,
                max_value=15,
                step=1,
                help_text="Fast EMA period",
            ),
            ParamConfig(
                name="slow",
                label="Slow Period",
                param_type="int",
                default=26,
                min_value=20,
                max_value=35,
                step=2,
                help_text="Slow EMA period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate PPO signals."""
        fast = self.params.get("fast", 12)
        slow = self.params.get("slow", 26)
        
        # Calculate PPO
        ppo_val, signal_line, hist = ppo(data["close"], fast, slow, 9)
        data["ppo"] = ppo_val
        data["ppo_signal"] = signal_line
        data["ppo_hist"] = hist
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_ppo = ppo_val.shift(1)
        prev_signal = signal_line.shift(1)
        
        # Long: PPO crosses above signal
        long_cond = (ppo_val > signal_line) & (prev_ppo <= prev_signal)
        
        # Short: PPO crosses below signal
        short_cond = (ppo_val < signal_line) & (prev_ppo >= prev_signal)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on opposite cross
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "PPO", "column": "ppo", "color": "blue", "style": "solid"},
            {"name": "Signal", "column": "ppo_signal", "color": "orange", "style": "dashed"},
        ]
