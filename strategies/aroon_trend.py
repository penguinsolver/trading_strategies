"""
Strategy: Aroon Trend

Aroon indicator for trend identification and strength.
Aroon measures time since recent highs/lows.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> tuple:
    """Calculate Aroon Up and Down."""
    # Days since highest high
    def bars_since_high(x):
        return period - x.argmax()
    
    def bars_since_low(x):
        return period - x.argmin()
    
    aroon_up = high.rolling(period + 1).apply(bars_since_high, raw=True) / period * 100
    aroon_down = low.rolling(period + 1).apply(bars_since_low, raw=True) / period * 100
    
    return aroon_up, aroon_down


class AroonTrendStrategy(Strategy):
    """Aroon trend identification strategy."""
    
    @property
    def name(self) -> str:
        return "Aroon Trend"
    
    @property
    def description(self) -> str:
        return "Trade Aroon crosses for early trend detection. Aroon Up/Down crosses signal trend changes."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="Aroon Period",
                param_type="int",
                default=25,
                min_value=14,
                max_value=50,
                step=5,
                help_text="Lookback for high/low measurement",
            ),
            ParamConfig(
                name="threshold",
                label="Strength Threshold",
                param_type="int",
                default=70,
                min_value=50,
                max_value=90,
                step=10,
                help_text="Min Aroon level for strong trend",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Aroon signals."""
        period = self.params.get("period", 25)
        threshold = self.params.get("threshold", 70)
        
        # Calculate Aroon
        up, down = aroon(data["high"], data["low"], period)
        data["aroon_up"] = up
        data["aroon_down"] = down
        data["aroon_osc"] = up - down
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_up = up.shift(1)
        prev_down = down.shift(1)
        
        # Long: Aroon Up crosses above Aroon Down AND Up > threshold
        long_cond = (
            (up > down) &
            (prev_up <= prev_down) &
            (up > threshold)
        )
        
        # Short: Aroon Down crosses above Aroon Up AND Down > threshold
        short_cond = (
            (down > up) &
            (prev_down <= prev_up) &
            (down > threshold)
        )
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit when trend weakens (oscillator near zero)
        data.loc[(data["aroon_osc"].abs() < 30).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Aroon Up", "column": "aroon_up", "color": "green", "style": "solid"},
            {"name": "Aroon Down", "column": "aroon_down", "color": "red", "style": "solid"},
        ]
