"""
Strategy: Chandelier Exit Trend

Trend-following strategy using Chandelier Exit for dynamic stop management.
Entry: Trend confirmation (EMA slope direction) with Supertrend or breakout
Exit: Chandelier trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema, chandelier_stop, ema_slope, supertrend


class ChandelierTrendStrategy(Strategy):
    """Chandelier Exit trend-following strategy."""
    
    @property
    def name(self) -> str:
        return "Chandelier Trend"
    
    @property
    def description(self) -> str:
        return "Trend-following with Chandelier Exit trailing stop. Filters entries with EMA slope direction."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="ema_period",
                label="EMA Period",
                param_type="int",
                default=200,
                min_value=100,
                max_value=300,
                step=50,
                help_text="EMA period for trend direction",
            ),
            ParamConfig(
                name="chandelier_n",
                label="Chandelier Lookback",
                param_type="int",
                default=22,
                min_value=15,
                max_value=30,
                step=1,
                help_text="Lookback for highest high / lowest low",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=21,
                step=1,
                help_text="ATR calculation period",
            ),
            ParamConfig(
                name="chandelier_mult",
                label="Chandelier Multiplier",
                param_type="float",
                default=3.0,
                min_value=2.0,
                max_value=4.0,
                step=0.5,
                help_text="ATR multiplier for Chandelier stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Chandelier Exit trend signals."""
        # Parameters
        ema_period = self.params.get("ema_period", 200)
        chandelier_n = self.params.get("chandelier_n", 22)
        atr_period = self.params.get("atr_period", 14)
        chandelier_mult = self.params.get("chandelier_mult", 3.0)
        
        # Calculate EMA and slope for trend direction
        data["ema"] = ema(data["close"], ema_period)
        data["ema_slope"] = ema_slope(data["close"], ema_period, 10)
        
        # Calculate Chandelier stops
        long_stop, short_stop = chandelier_stop(
            data["high"], data["low"], data["close"],
            n=chandelier_n, atr_period=atr_period, mult=chandelier_mult
        )
        data["chandelier_long"] = long_stop
        data["chandelier_short"] = short_stop
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Calculate Supertrend for entry signals
        st_line, st_direction = supertrend(
            data["high"], data["low"], data["close"],
            atr_period=10, multiplier=2.0
        )
        data["st_direction"] = st_direction
        
        # Detect Supertrend flips
        prev_dir = data["st_direction"].shift(1)
        
        # Entry conditions: Supertrend flip in direction of EMA slope
        # Long: EMA slope positive AND Supertrend flips bullish
        data["long_signal"] = (
            (data["ema_slope"] > 0) &
            (data["st_direction"] == 1) &
            (prev_dir == -1)
        )
        
        # Short: EMA slope negative AND Supertrend flips bearish
        data["short_signal"] = (
            (data["ema_slope"] < 0) &
            (data["st_direction"] == -1) &
            (prev_dir == 1)
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["long_signal"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "chandelier_long"]
        # Use Chandelier as trailing stop - convert to ATR distance
        chandelier_distance = data.loc[long_mask, "close"] - data.loc[long_mask, "chandelier_long"]
        data.loc[long_mask, "trailing_stop_atr"] = chandelier_distance
        
        # Short entries
        short_mask = data["short_signal"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "chandelier_short"]
        chandelier_distance = data.loc[short_mask, "chandelier_short"] - data.loc[short_mask, "close"]
        data.loc[short_mask, "trailing_stop_atr"] = chandelier_distance
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA", "column": "ema", "color": "blue", "style": "solid"},
            {"name": "Chandelier Long", "column": "chandelier_long", "color": "green", "style": "dashed"},
            {"name": "Chandelier Short", "column": "chandelier_short", "color": "red", "style": "dashed"},
        ]
