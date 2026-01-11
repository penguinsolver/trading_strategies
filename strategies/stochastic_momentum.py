"""
Strategy: Stochastic Momentum (80/20)

Classic stochastic oscillator for overbought/oversold momentum trading.
Uses %K and %D crossovers at extreme levels.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, sma


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic %K and %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = sma(k, d_period)
    
    return k, d


class StochasticMomentumStrategy(Strategy):
    """Stochastic oscillator momentum strategy."""
    
    @property
    def name(self) -> str:
        return "Stochastic 80/20"
    
    @property
    def description(self) -> str:
        return "Trade stochastic crossovers at overbought (>80) and oversold (<20) levels."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="k_period",
                label="%K Period",
                param_type="int",
                default=14,
                min_value=5,
                max_value=21,
                step=2,
                help_text="Stochastic %K lookback",
            ),
            ParamConfig(
                name="d_period",
                label="%D Period",
                param_type="int",
                default=3,
                min_value=2,
                max_value=5,
                step=1,
                help_text="Stochastic %D smoothing",
            ),
            ParamConfig(
                name="overbought",
                label="Overbought",
                param_type="int",
                default=80,
                min_value=70,
                max_value=90,
                step=5,
                help_text="Overbought threshold",
            ),
            ParamConfig(
                name="oversold",
                label="Oversold",
                param_type="int",
                default=20,
                min_value=10,
                max_value=30,
                step=5,
                help_text="Oversold threshold",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate stochastic signals."""
        k_period = self.params.get("k_period", 14)
        d_period = self.params.get("d_period", 3)
        overbought = self.params.get("overbought", 80)
        oversold = self.params.get("oversold", 20)
        
        # Calculate stochastic
        k, d = stochastic(data["high"], data["low"], data["close"], k_period, d_period)
        data["stoch_k"] = k
        data["stoch_d"] = d
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Crossovers with shift
        prev_k = k.shift(1)
        prev_d = d.shift(1)
        
        # Long: %K crosses above %D in oversold zone
        long_cond = (
            (k > d) &
            (prev_k <= prev_d) &
            (prev_k < oversold)
        )
        
        # Short: %K crosses below %D in overbought zone
        short_cond = (
            (k < d) &
            (prev_k >= prev_d) &
            (prev_k > overbought)
        )
        
        # Exit conditions
        exit_long = (k > overbought) | ((k < d) & (prev_k >= prev_d))
        exit_short = (k < oversold) | ((k > d) & (prev_k <= prev_d))
        
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
        
        # Exits
        data.loc[exit_long.fillna(False), "exit_signal"] = True
        data.loc[exit_short.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Stoch %K", "column": "stoch_k", "color": "blue", "style": "solid"},
            {"name": "Stoch %D", "column": "stoch_d", "color": "orange", "style": "dashed"},
        ]
