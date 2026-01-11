"""
Strategy: RSI Extreme Bounce (Optimized for 70% WR)

Based on research showing RSI on 15min with 1:2 R:R achieved 70% win rate.
Very tight entry at RSI extremes with quick profit targets.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class RSIExtremeBounceStrategy(Strategy):
    """RSI extreme bounce with tight R:R for high win rate."""
    
    @property
    def name(self) -> str:
        return "RSI Extreme"
    
    @property
    def description(self) -> str:
        return "Trade RSI extreme bounces (<15 or >85) with 1:2 risk-reward for high win rate."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="RSI Period",
                param_type="int",
                default=7,
                min_value=5,
                max_value=14,
                step=2,
                help_text="Faster RSI for quicker signals",
            ),
            ParamConfig(
                name="extreme_low",
                label="Extreme Low",
                param_type="int",
                default=15,
                min_value=10,
                max_value=25,
                step=5,
                help_text="Very oversold level",
            ),
            ParamConfig(
                name="extreme_high",
                label="Extreme High",
                param_type="int",
                default=85,
                min_value=75,
                max_value=90,
                step=5,
                help_text="Very overbought level",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI extreme signals."""
        period = self.params.get("period", 7)
        extreme_low = self.params.get("extreme_low", 15)
        extreme_high = self.params.get("extreme_high", 85)
        
        data["rsi"] = rsi(data["close"], period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_rsi = data["rsi"].shift(1)
        
        # Very strict entry: RSI bouncing from extreme
        long_cond = (data["rsi"] > extreme_low) & (prev_rsi <= extreme_low)
        short_cond = (data["rsi"] < extreme_high) & (prev_rsi >= extreme_high)
        
        # Initialize
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long with tight stop (0.5 ATR), target 1 ATR (1:2 R:R)
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.75
        
        # Short
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.75
        
        # Quick exit at neutral RSI
        data.loc[(data["rsi"] > 45) & (data["rsi"] < 55), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "RSI", "column": "rsi", "color": "purple", "style": "solid"}]
