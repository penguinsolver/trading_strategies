"""
Strategy D: Simple MA Crossover (Baseline)

A simple strategy to verify the backtest engine works correctly.
Entry: Fast EMA crosses above/below slow EMA.
Stop: ATR-based.
Exit: Opposite crossover.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr
from indicators.moving_averages import crossover, crossunder


class MACrossoverStrategy(Strategy):
    """Simple MA crossover strategy for baseline comparison."""
    
    @property
    def name(self) -> str:
        return "MA Crossover"
    
    @property
    def description(self) -> str:
        return "Simple EMA crossover baseline. Useful for sanity-checking the backtest engine."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="fast_period",
                label="Fast EMA Period",
                param_type="int",
                default=10,
                min_value=3,
                max_value=50,
                step=1,
                help_text="Fast EMA period",
            ),
            ParamConfig(
                name="slow_period",
                label="Slow EMA Period",
                param_type="int",
                default=30,
                min_value=10,
                max_value=100,
                step=5,
                help_text="Slow EMA period",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="Stop ATR Multiplier",
                param_type="float",
                default=1.5,
                min_value=0.5,
                max_value=4.0,
                step=0.5,
                help_text="ATR multiplier for stop distance",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=5,
                max_value=30,
                step=1,
                help_text="ATR period for stop calculation",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals."""
        # Parameters
        fast_period = self.params.get("fast_period", 10)
        slow_period = self.params.get("slow_period", 30)
        atr_stop_mult = self.params.get("atr_stop_mult", 1.5)
        atr_period = self.params.get("atr_period", 14)
        
        # Calculate indicators
        data["ema_fast"] = ema(data["close"], fast_period)
        data["ema_slow"] = ema(data["close"], slow_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Crossover signals
        data["golden_cross"] = crossover(data["ema_fast"], data["ema_slow"])
        data["death_cross"] = crossunder(data["ema_fast"], data["ema_slow"])
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        
        # Track current position for exit signals
        in_long = False
        in_short = False
        
        for i in range(len(data)):
            # Entry signals
            if data["golden_cross"].iloc[i] and not in_long:
                data.loc[data.index[i], "entry_signal"] = 1
                # Stop below by ATR
                stop = data["close"].iloc[i] - data["atr"].iloc[i] * atr_stop_mult
                data.loc[data.index[i], "stop_price"] = stop
                in_long = True
                in_short = False
                
            elif data["death_cross"].iloc[i] and not in_short:
                data.loc[data.index[i], "entry_signal"] = -1
                # Stop above by ATR
                stop = data["close"].iloc[i] + data["atr"].iloc[i] * atr_stop_mult
                data.loc[data.index[i], "stop_price"] = stop
                in_short = True
                in_long = False
            
            # Exit signals (opposite crossover)
            if in_long and data["death_cross"].iloc[i]:
                data.loc[data.index[i], "exit_signal"] = True
                in_long = False
            elif in_short and data["golden_cross"].iloc[i]:
                data.loc[data.index[i], "exit_signal"] = True
                in_short = False
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Fast EMA", "column": "ema_fast", "color": "green", "style": "solid"},
            {"name": "Slow EMA", "column": "ema_slow", "color": "red", "style": "solid"},
        ]
