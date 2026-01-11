"""
Strategy: Bollinger Band Squeeze Breakout

Volatility compression-to-expansion strategy.
Entry: Break above/below Bollinger Bands after squeeze condition
Exit: Mid-band cross or ATR trailing stop
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, bollinger_bands, bollinger_bandwidth, bollinger_bandwidth_percentile


class BBSqueezeStrategy(Strategy):
    """Bollinger Band squeeze breakout strategy."""
    
    @property
    def name(self) -> str:
        return "BB Squeeze"
    
    @property
    def description(self) -> str:
        return "Volatility squeeze breakout. Enters on band break after compression, trails with ATR stop."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="bb_period",
                label="BB Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=30,
                step=5,
                help_text="Bollinger Bands calculation period",
            ),
            ParamConfig(
                name="bb_std",
                label="BB Std Dev",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="Bollinger Bands standard deviation multiplier",
            ),
            ParamConfig(
                name="squeeze_lookback",
                label="Squeeze Lookback",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=10,
                help_text="Lookback period for bandwidth percentile",
            ),
            ParamConfig(
                name="squeeze_percentile",
                label="Squeeze Percentile",
                param_type="int",
                default=20,
                min_value=5,
                max_value=40,
                step=5,
                help_text="Bandwidth percentile threshold for squeeze detection",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=10,
                max_value=20,
                step=1,
                help_text="ATR period for trailing stop",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiple for trailing stop",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger squeeze breakout signals."""
        # Parameters
        bb_period = self.params.get("bb_period", 20)
        bb_std = self.params.get("bb_std", 2.0)
        squeeze_lookback = self.params.get("squeeze_lookback", 50)
        squeeze_percentile = self.params.get("squeeze_percentile", 20)
        atr_period = self.params.get("atr_period", 14)
        atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        
        # Calculate Bollinger Bands
        bb_upper, bb_lower, bb_mid = bollinger_bands(data["close"], bb_period, bb_std)
        data["bb_upper"] = bb_upper
        data["bb_lower"] = bb_lower
        data["bb_mid"] = bb_mid
        
        # Calculate bandwidth and percentile
        data["bandwidth"] = bollinger_bandwidth(data["close"], bb_period, bb_std)
        data["bw_percentile"] = bollinger_bandwidth_percentile(data["bandwidth"], squeeze_lookback)
        
        # Detect squeeze condition (low bandwidth)
        data["in_squeeze"] = data["bw_percentile"] < squeeze_percentile
        
        # Was in squeeze on previous bar
        data["was_squeeze"] = data["in_squeeze"].shift(1).fillna(False)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Breakout detection - close breaks band after squeeze
        # Long: Was in squeeze AND close breaks above upper band
        data["breakout_long"] = (
            data["was_squeeze"] &
            (data["close"] > data["bb_upper"]) &
            (data["close"].shift(1) <= data["bb_upper"].shift(1))
        )
        
        # Short: Was in squeeze AND close breaks below lower band
        data["breakout_short"] = (
            data["was_squeeze"] &
            (data["close"] < data["bb_lower"]) &
            (data["close"].shift(1) >= data["bb_lower"].shift(1))
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["breakout_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "bb_mid"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries
        short_mask = data["breakout_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "bb_mid"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        # Exit on mid-band cross (price returns to middle)
        # Long exit: close crosses below mid
        # Short exit: close crosses above mid
        data["mid_cross_down"] = (data["close"] < data["bb_mid"]) & (data["close"].shift(1) >= data["bb_mid"].shift(1))
        data["mid_cross_up"] = (data["close"] > data["bb_mid"]) & (data["close"].shift(1) <= data["bb_mid"].shift(1))
        
        data.loc[data["mid_cross_down"] | data["mid_cross_up"], "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "BB Upper", "column": "bb_upper", "color": "blue", "style": "solid"},
            {"name": "BB Lower", "column": "bb_lower", "color": "blue", "style": "solid"},
            {"name": "BB Mid", "column": "bb_mid", "color": "gray", "style": "dashed"},
        ]
