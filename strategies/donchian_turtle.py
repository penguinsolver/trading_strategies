"""
Strategy: Donchian Turtle Breakout

Classic Turtle Trading breakout strategy.
Entry: Close breaks above Donchian High(N) for long / below Donchian Low(N) for short
Exit: Donchian Mid(exit_N) cross OR ATR trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, donchian_channels


class DonchianTurtleStrategy(Strategy):
    """Classic Turtle Trading breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Donchian Turtle"
    
    @property
    def description(self) -> str:
        return "Classic Turtle breakout. Enter on N-period high/low break, exit on shorter-period mid or ATR trail."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="entry_n",
                label="Entry Period (N)",
                param_type="int",
                default=20,
                min_value=10,
                max_value=55,
                step=5,
                help_text="Donchian channel period for entry signals",
            ),
            ParamConfig(
                name="exit_n",
                label="Exit Period",
                param_type="int",
                default=10,
                min_value=5,
                max_value=20,
                step=1,
                help_text="Donchian channel period for exit signals",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ATR period for stops",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiple for initial and trailing stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Donchian Turtle breakout signals."""
        # Parameters
        entry_n = self.params.get("entry_n", 20)
        exit_n = self.params.get("exit_n", 10)
        atr_period = self.params.get("atr_period", 14)
        atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        
        # Calculate Entry Donchian Channels (use shifted data to avoid lookahead)
        entry_upper, entry_lower, entry_mid = donchian_channels(
            data["high"], data["low"], entry_n
        )
        data["dc_entry_upper"] = entry_upper.shift(1)  # Use previous bar's channel
        data["dc_entry_lower"] = entry_lower.shift(1)
        
        # Calculate Exit Donchian Channels
        exit_upper, exit_lower, exit_mid = donchian_channels(
            data["high"], data["low"], exit_n
        )
        data["dc_exit_mid"] = exit_mid.shift(1)
        data["dc_exit_lower"] = exit_lower.shift(1)
        data["dc_exit_upper"] = exit_upper.shift(1)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Breakout detection
        # Long: Close breaks above entry upper channel
        data["breakout_long"] = (
            (data["close"] > data["dc_entry_upper"]) &
            (data["close"].shift(1) <= data["dc_entry_upper"].shift(1))
        )
        
        # Short: Close breaks below entry lower channel
        data["breakout_short"] = (
            (data["close"] < data["dc_entry_lower"]) &
            (data["close"].shift(1) >= data["dc_entry_lower"].shift(1))
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["breakout_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        # Stop below entry by ATR multiple
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - (
            data.loc[long_mask, "atr"] * atr_stop_mult
        )
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + (
            data.loc[short_mask, "atr"] * atr_stop_mult
        )
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        # Exit signals on mid-channel cross (for opposite position)
        # Long exit: close crosses below exit mid
        # Short exit: close crosses above exit mid
        # These are handled by the backtest engine's trailing stop mostly,
        # but we can add exit signals for mid-channel exits
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Entry Upper", "column": "dc_entry_upper", "color": "green", "style": "solid"},
            {"name": "Entry Lower", "column": "dc_entry_lower", "color": "red", "style": "solid"},
            {"name": "Exit Mid", "column": "dc_exit_mid", "color": "gray", "style": "dashed"},
        ]
