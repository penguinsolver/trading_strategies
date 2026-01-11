"""
Strategy: Quick RSI Scalp

Very fast RSI scalping with tight entries/exits.
Enter on RSI extreme, exit quickly.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def rsi(close: pd.Series, period: int = 5) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class QuickRSIScalpStrategy(Strategy):
    """Quick RSI scalp strategy."""
    
    @property
    def name(self) -> str:
        return "Quick RSI Scalp"
    
    @property
    def description(self) -> str:
        return "Very fast RSI(5) scalping with <20/>80 entry and quick 50 exit."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate quick RSI signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 10)
        data["rsi5"] = rsi(data["close"], 5)
        
        # Very extreme readings
        oversold = data["rsi5"] < 20
        overbought = data["rsi5"] > 80
        
        # Entry on extreme
        long_cond = oversold & (data["close"] > data["close"].shift(1))  # Bounce
        short_cond = overbought & (data["close"] < data["close"].shift(1))  # Rejection
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.5
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.5
        
        # Quick exit at RSI 50
        data.loc[(data["rsi5"] > 45) & (data["rsi5"] < 55), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "RSI 5", "column": "rsi5", "color": "purple", "style": "solid"}]
