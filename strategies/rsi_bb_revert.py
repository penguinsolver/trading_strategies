"""
Strategy: Mean Reversion RSI+BB

Mean reversion using both RSI and Bollinger Band extremes.
Double confirmation for higher probability.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, sma


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class MeanReversionRSIBBStrategy(Strategy):
    """RSI + BB mean reversion strategy."""
    
    @property
    def name(self) -> str:
        return "RSI+BB Revert"
    
    @property
    def description(self) -> str:
        return "Mean reversion when both RSI extreme AND price at Bollinger Band."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI+BB signals."""
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["rsi"] = rsi(data["close"], 14)
        
        # Bollinger Bands
        data["bb_mid"] = sma(data["close"], 20)
        data["bb_std"] = data["close"].rolling(20).std()
        data["bb_upper"] = data["bb_mid"] + 2 * data["bb_std"]
        data["bb_lower"] = data["bb_mid"] - 2 * data["bb_std"]
        
        # Double confirmation
        oversold = (data["rsi"] < 30) & (data["close"] < data["bb_lower"])
        overbought = (data["rsi"] > 70) & (data["close"] > data["bb_upper"])
        
        # Previous states
        prev_oversold = oversold.shift(1)
        prev_overbought = overbought.shift(1)
        
        # Entry on exit from extreme
        long_cond = prev_oversold & (data["rsi"] > 30)
        short_cond = prev_overbought & (data["rsi"] < 70)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "bb_lower"] - data.loc[long_mask, "atr"] * 0.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "bb_upper"] + data.loc[short_mask, "atr"] * 0.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit at BB mid
        data.loc[(data["close"] > data["bb_mid"]).fillna(False), "exit_signal"] = True
        data.loc[(data["close"] < data["bb_mid"]).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "BB Upper", "column": "bb_upper", "color": "red", "style": "dashed"},
            {"name": "BB Lower", "column": "bb_lower", "color": "green", "style": "dashed"},
        ]
