"""
Strategy: CCI Momentum

Commodity Channel Index for momentum and trend trading.
CCI measures price deviation from average, good for breakouts.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, sma


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, period)
    mean_deviation = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    
    cci_val = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci_val


class CCIMomentumStrategy(Strategy):
    """CCI momentum strategy."""
    
    @property
    def name(self) -> str:
        return "CCI Momentum"
    
    @property
    def description(self) -> str:
        return "Trade CCI breakouts above/below 100/-100 for momentum entries with trend confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=40,
                step=5,
                help_text="CCI lookback period",
            ),
            ParamConfig(
                name="entry_level",
                label="Entry Level",
                param_type="int",
                default=100,
                min_value=50,
                max_value=150,
                step=25,
                help_text="CCI level for entry (±)",
            ),
            ParamConfig(
                name="exit_level",
                label="Exit Level",
                param_type="int",
                default=0,
                min_value=-50,
                max_value=50,
                step=25,
                help_text="CCI level for exit",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate CCI signals."""
        period = self.params.get("period", 20)
        entry_level = self.params.get("entry_level", 100)
        exit_level = self.params.get("exit_level", 0)
        
        # Calculate CCI
        data["cci"] = cci(data["high"], data["low"], data["close"], period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_cci = data["cci"].shift(1)
        
        # Long: CCI breaks above +entry_level
        long_cond = (
            (data["cci"] > entry_level) &
            (prev_cci <= entry_level)
        )
        
        # Short: CCI breaks below -entry_level
        short_cond = (
            (data["cci"] < -entry_level) &
            (prev_cci >= -entry_level)
        )
        
        # Exit conditions
        exit_long = (data["cci"] < exit_level) & (prev_cci >= exit_level)
        exit_short = (data["cci"] > exit_level) & (prev_cci <= exit_level)
        
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
        
        # Exits
        data.loc[(exit_long | exit_short).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "CCI", "column": "cci", "color": "teal", "style": "solid"},
        ]
