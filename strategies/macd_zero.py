"""
Strategy: MACD Zero Cross

Trade MACD line zero crossings for momentum confirmation.
More selective than signal line crosses.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


class MACDZeroCrossStrategy(Strategy):
    """MACD zero-line cross strategy."""
    
    @property
    def name(self) -> str:
        return "MACD Zero"
    
    @property
    def description(self) -> str:
        return "Trade MACD line zero crosses for strong momentum confirmation with trend filter."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast",
                label="Fast EMA",
                param_type="int",
                default=12,
                min_value=8,
                max_value=15,
                step=1,
                help_text="Fast EMA period",
            ),
            ParamConfig(
                name="slow",
                label="Slow EMA",
                param_type="int",
                default=26,
                min_value=20,
                max_value=35,
                step=2,
                help_text="Slow EMA period",
            ),
            ParamConfig(
                name="trend_ema",
                label="Trend EMA",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=10,
                help_text="Trend filter EMA (0=disabled)",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD zero cross signals."""
        fast = self.params.get("fast", 12)
        slow = self.params.get("slow", 26)
        trend_period = self.params.get("trend_ema", 50)
        
        # Calculate MACD
        ema_fast = ema(data["close"], fast)
        ema_slow = ema(data["close"], slow)
        data["macd"] = ema_fast - ema_slow
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Trend filter
        if trend_period > 0:
            data["trend_ema"] = ema(data["close"], trend_period)
            above_trend = data["close"] > data["trend_ema"]
            below_trend = data["close"] < data["trend_ema"]
        else:
            above_trend = pd.Series(True, index=data.index)
            below_trend = pd.Series(True, index=data.index)
        
        prev_macd = data["macd"].shift(1)
        
        # Long: MACD crosses above zero AND above trend EMA
        long_cond = (data["macd"] > 0) & (prev_macd <= 0) & above_trend
        
        # Short: MACD crosses below zero AND below trend EMA
        short_cond = (data["macd"] < 0) & (prev_macd >= 0) & below_trend
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 2
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.5
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 2
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.5
        
        # Exit on opposite cross
        data.loc[short_cond.fillna(False), "exit_signal"] = True
        data.loc[long_cond.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "MACD", "column": "macd", "color": "blue", "style": "solid"},
        ]
