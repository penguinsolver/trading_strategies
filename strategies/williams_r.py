"""
Strategy: Williams %R Reversal

Trade extreme Williams %R readings for mean reversion.
Williams %R is similar to stochastic but inverted scale.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


class WilliamsRStrategy(Strategy):
    """Williams %R reversal strategy."""
    
    @property
    def name(self) -> str:
        return "Williams %R"
    
    @property
    def description(self) -> str:
        return "Trade Williams %R extremes with trend filter for high-probability reversals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="Period",
                param_type="int",
                default=14,
                min_value=7,
                max_value=21,
                step=2,
                help_text="Williams %R lookback",
            ),
            ParamConfig(
                name="overbought",
                label="Overbought",
                param_type="int",
                default=-20,
                min_value=-30,
                max_value=-10,
                step=5,
                help_text="Overbought level (negative)",
            ),
            ParamConfig(
                name="oversold",
                label="Oversold",
                param_type="int",
                default=-80,
                min_value=-90,
                max_value=-70,
                step=5,
                help_text="Oversold level (negative)",
            ),
            ParamConfig(
                name="ema_filter",
                label="EMA Filter",
                param_type="int",
                default=50,
                min_value=0,
                max_value=100,
                step=10,
                help_text="Trend EMA (0=disabled)",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Williams %R signals."""
        period = self.params.get("period", 14)
        overbought = self.params.get("overbought", -20)
        oversold = self.params.get("oversold", -80)
        ema_period = self.params.get("ema_filter", 50)
        
        # Calculate Williams %R
        data["williams_r"] = williams_r(data["high"], data["low"], data["close"], period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # EMA trend filter
        if ema_period > 0:
            data["ema_trend"] = ema(data["close"], ema_period)
            above_ema = data["close"] > data["ema_trend"]
            below_ema = data["close"] < data["ema_trend"]
        else:
            above_ema = pd.Series(True, index=data.index)
            below_ema = pd.Series(True, index=data.index)
        
        prev_wr = data["williams_r"].shift(1)
        
        # Long: %R rises from oversold (only in uptrend)
        long_cond = (
            (data["williams_r"] > oversold) &
            (prev_wr <= oversold) &
            above_ema
        )
        
        # Short: %R falls from overbought (only in downtrend)
        short_cond = (
            (data["williams_r"] < overbought) &
            (prev_wr >= overbought) &
            below_ema
        )
        
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
        
        # Exit at neutral zone
        exit_cond = (data["williams_r"] > -50) & (data["williams_r"] < -50)
        data.loc[exit_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Williams %R", "column": "williams_r", "color": "purple", "style": "solid"},
        ]
