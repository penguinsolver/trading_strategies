"""
Strategy: Elder Ray (Bull/Bear Power)

Alexander Elder's Bull and Bear Power indicators.
Measures buying/selling pressure relative to EMA.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr


def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> tuple:
    """Calculate Bull and Bear Power."""
    ema_line = ema(close, period)
    bull_power = high - ema_line
    bear_power = low - ema_line
    return bull_power, bear_power, ema_line


class ElderRayStrategy(Strategy):
    """Elder Ray bull/bear power strategy."""
    
    @property
    def name(self) -> str:
        return "Elder Ray"
    
    @property
    def description(self) -> str:
        return "Trade Elder Ray bull/bear power divergences with EMA trend confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="EMA Period",
                param_type="int",
                default=13,
                min_value=10,
                max_value=20,
                step=2,
                help_text="EMA period for Elder Ray",
            ),
            ParamConfig(
                name="power_threshold",
                label="Power Threshold",
                param_type="float",
                default=0.0,
                min_value=-0.5,
                max_value=0.5,
                step=0.1,
                help_text="Min power change for signal",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Elder Ray signals."""
        period = self.params.get("period", 13)
        
        # Calculate Elder Ray
        bull_power, bear_power, ema_line = elder_ray(data["high"], data["low"], data["close"], period)
        data["bull_power"] = bull_power
        data["bear_power"] = bear_power
        data["elder_ema"] = ema_line
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_bull = bull_power.shift(1)
        prev_bear = bear_power.shift(1)
        prev_ema = ema_line.shift(1)
        
        # Long: EMA rising + bear power rising from negative (bears losing control)
        long_cond = (
            (ema_line > prev_ema) &  # Uptrend
            (bear_power < 0) &  # Bears still negative
            (bear_power > prev_bear)  # But rising (less selling)
        )
        
        # Short: EMA falling + bull power falling from positive (bulls losing control)
        short_cond = (
            (ema_line < prev_ema) &  # Downtrend
            (bull_power > 0) &  # Bulls still positive
            (bull_power < prev_bull)  # But falling (less buying)
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
        
        # Exit when power reverses
        exit_long = (bear_power < prev_bear) | (ema_line < prev_ema)
        exit_short = (bull_power > prev_bull) | (ema_line > prev_ema)
        data.loc[(exit_long | exit_short).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Elder EMA", "column": "elder_ema", "color": "blue", "style": "solid"},
        ]
