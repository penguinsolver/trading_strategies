"""
Strategy: Hull Moving Average Trend

Hull MA provides faster response than EMA while reducing noise.
Trade Hull MA direction changes for smooth trend signals.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def wma(data: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_ma(data: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average (HMA)."""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    return hma


class HullMAStrategy(Strategy):
    """Hull Moving Average trend strategy."""
    
    @property
    def name(self) -> str:
        return "Hull MA"
    
    @property
    def description(self) -> str:
        return "Trade Hull MA direction changes for faster trend signals with less lag."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="HMA Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="Hull MA period",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Hull MA signals."""
        period = self.params.get("period", 20)
        
        # Calculate Hull MA
        data["hma"] = hull_ma(data["close"], period)
        data["hma_prev"] = data["hma"].shift(1)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # HMA direction
        hma_rising = data["hma"] > data["hma_prev"]
        hma_falling = data["hma"] < data["hma_prev"]
        prev_rising = data["hma_prev"] > data["hma"].shift(2)
        
        # Long: HMA turns up
        long_cond = hma_rising & ~prev_rising
        
        # Short: HMA turns down
        short_cond = hma_falling & prev_rising
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 2
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 2
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on direction change
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Hull MA", "column": "hma", "color": "purple", "style": "solid"},
        ]
