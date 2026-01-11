"""
Strategy B: Donchian Breakout with Volatility Filter

Entry: Break above/below Donchian channel
Filter: Only enter when volatility is expanding (ATR ratio)
Stop: Opposite Donchian band
Target: Trail with Donchian mid or 2R
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, donchian_channels


class BreakoutStrategy(Strategy):
    """Donchian channel breakout with volatility filter."""
    
    @property
    def name(self) -> str:
        return "Breakout"
    
    @property
    def description(self) -> str:
        return "Donchian channel breakout strategy. Enters on range expansion with volatility confirmation."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="donchian_period",
                label="Donchian Period",
                param_type="int",
                default=20,
                min_value=5,
                max_value=50,
                step=1,
                help_text="Lookback period for Donchian channels",
            ),
            ParamConfig(
                name="atr_fast",
                label="Fast ATR Period",
                param_type="int",
                default=20,
                min_value=5,
                max_value=30,
                step=1,
                help_text="Short-term ATR for volatility filter",
            ),
            ParamConfig(
                name="atr_slow",
                label="Slow ATR Period",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=5,
                help_text="Long-term ATR for volatility baseline",
            ),
            ParamConfig(
                name="volatility_threshold",
                label="Volatility Threshold",
                param_type="float",
                default=1.2,
                min_value=0.8,
                max_value=2.5,
                step=0.1,
                help_text="ATR ratio threshold (fast/slow) for volatility expansion",
            ),
            ParamConfig(
                name="use_trail",
                label="Use Trailing Exit",
                param_type="bool",
                default=True,
                help_text="Trail stop using Donchian mid vs exit on opposite band touch",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals."""
        # Parameters
        donchian_period = self.params.get("donchian_period", 20)
        atr_fast_period = self.params.get("atr_fast", 20)
        atr_slow_period = self.params.get("atr_slow", 50)
        volatility_threshold = self.params.get("volatility_threshold", 1.2)
        
        # Calculate Donchian Channels
        upper, lower, mid = donchian_channels(data["high"], data["low"], donchian_period)
        data["dc_upper"] = upper
        data["dc_lower"] = lower
        data["dc_mid"] = mid
        
        # Calculate ATR for volatility filter
        data["atr_fast"] = atr(data["high"], data["low"], data["close"], atr_fast_period)
        data["atr_slow"] = atr(data["high"], data["low"], data["close"], atr_slow_period)
        data["atr_ratio"] = data["atr_fast"] / data["atr_slow"]
        
        # Volatility expansion filter
        data["volatility_ok"] = data["atr_ratio"] >= volatility_threshold
        
        # Breakout detection
        data["breakout_long"] = (
            (data["close"] > data["dc_upper"].shift(1)) &  # Close above upper channel
            (data["close"].shift(1) <= data["dc_upper"].shift(2)) &  # Previous wasn't
            data["volatility_ok"]
        )
        
        data["breakout_short"] = (
            (data["close"] < data["dc_lower"].shift(1)) &  # Close below lower channel
            (data["close"].shift(1) >= data["dc_lower"].shift(2)) &  # Previous wasn't
            data["volatility_ok"]
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Generate signals
        for i in range(len(data)):
            if data["breakout_long"].iloc[i]:
                data.loc[data.index[i], "entry_signal"] = 1
                # Stop at lower Donchian band
                data.loc[data.index[i], "stop_price"] = data["dc_lower"].iloc[i]
                # Use ATR for trailing
                data.loc[data.index[i], "trailing_stop_atr"] = data["atr_fast"].iloc[i] * 2
                
            elif data["breakout_short"].iloc[i]:
                data.loc[data.index[i], "entry_signal"] = -1
                # Stop at upper Donchian band
                data.loc[data.index[i], "stop_price"] = data["dc_upper"].iloc[i]
                # Use ATR for trailing
                data.loc[data.index[i], "trailing_stop_atr"] = data["atr_fast"].iloc[i] * 2
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "DC Upper", "column": "dc_upper", "color": "green", "style": "solid"},
            {"name": "DC Lower", "column": "dc_lower", "color": "red", "style": "solid"},
            {"name": "DC Mid", "column": "dc_mid", "color": "gray", "style": "dashed"},
        ]
