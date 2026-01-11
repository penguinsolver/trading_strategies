"""
Strategy A: BTC Trend Pullback

Multi-timeframe strategy:
- Higher timeframe (1h) for trend direction via EMA
- Lower timeframe (5m) for pullback entries

Entry Logic:
1. Higher TF: Price above EMA(50) = bullish bias
2. Lower TF: Price pulls back to slow EMA(20)
3. Trigger: Price closes back above fast EMA(10) after touching slow EMA

Exit Logic:
- Stop: Below recent swing low (for longs)
- Partial: +1R close 50%
- Remainder: Trailing stop via ATR
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr
from indicators.trend import recent_swing_low, recent_swing_high


class TrendPullbackStrategy(Strategy):
    """Multi-timeframe trend pullback strategy."""
    
    @property
    def name(self) -> str:
        return "Trend Pullback"
    
    @property
    def description(self) -> str:
        return "Multi-TF trend following with pullback entries. Uses higher TF for bias, lower TF for entry timing."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="htf_ema_period",
                label="Higher TF EMA Period",
                param_type="int",
                default=50,
                min_value=10,
                max_value=200,
                step=5,
                help_text="EMA period for trend filter on higher timeframe",
            ),
            ParamConfig(
                name="ltf_slow_ema",
                label="Lower TF Slow EMA",
                param_type="int",
                default=20,
                min_value=5,
                max_value=50,
                step=1,
                help_text="Slow EMA for pullback detection",
            ),
            ParamConfig(
                name="ltf_fast_ema",
                label="Lower TF Fast EMA",
                param_type="int",
                default=10,
                min_value=3,
                max_value=30,
                step=1,
                help_text="Fast EMA for entry trigger",
            ),
            ParamConfig(
                name="atr_period",
                label="ATR Period",
                param_type="int",
                default=14,
                min_value=5,
                max_value=30,
                step=1,
                help_text="ATR period for trailing stop",
            ),
            ParamConfig(
                name="atr_multiplier",
                label="ATR Trail Multiplier",
                param_type="float",
                default=2.0,
                min_value=0.5,
                max_value=5.0,
                step=0.5,
                help_text="ATR multiplier for trailing stop distance",
            ),
            ParamConfig(
                name="swing_lookback",
                label="Swing Lookback",
                param_type="int",
                default=10,
                min_value=3,
                max_value=30,
                step=1,
                help_text="Bars to look back for swing high/low stops",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend pullback signals."""
        # Extract parameters
        htf_ema_period = self.params.get("htf_ema_period", 50)
        ltf_slow_ema = self.params.get("ltf_slow_ema", 20)
        ltf_fast_ema = self.params.get("ltf_fast_ema", 10)
        atr_period = self.params.get("atr_period", 14)
        atr_multiplier = self.params.get("atr_multiplier", 2.0)
        swing_lookback = self.params.get("swing_lookback", 10)
        
        # Calculate indicators
        data["htf_ema"] = ema(data["close"], htf_ema_period)
        data["slow_ema"] = ema(data["close"], ltf_slow_ema)
        data["fast_ema"] = ema(data["close"], ltf_fast_ema)
        data["atr"] = atr(data["high"], data["low"], data["close"], atr_period)
        
        # Swing levels for stops
        data["swing_low"] = recent_swing_low(data["low"], swing_lookback)
        data["swing_high"] = recent_swing_high(data["high"], swing_lookback)
        
        # Trend bias from higher TF EMA
        data["trend_bullish"] = data["close"] > data["htf_ema"]
        data["trend_bearish"] = data["close"] < data["htf_ema"]
        
        # Pullback detection
        # Price touched slow EMA
        data["touched_slow"] = (
            (data["low"] <= data["slow_ema"]) | 
            (data["low"].shift(1) <= data["slow_ema"].shift(1)) |
            (data["low"].shift(2) <= data["slow_ema"].shift(2))
        )
        data["touched_slow_short"] = (
            (data["high"] >= data["slow_ema"]) | 
            (data["high"].shift(1) >= data["slow_ema"].shift(1)) |
            (data["high"].shift(2) >= data["slow_ema"].shift(2))
        )
        
        # Entry trigger: close above fast EMA after pullback
        data["long_trigger"] = (
            data["trend_bullish"] & 
            data["touched_slow"] &
            (data["close"] > data["fast_ema"]) &
            (data["close"].shift(1) <= data["fast_ema"].shift(1))
        )
        
        data["short_trigger"] = (
            data["trend_bearish"] & 
            data["touched_slow_short"] &
            (data["close"] < data["fast_ema"]) &
            (data["close"].shift(1) >= data["fast_ema"].shift(1))
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["target_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Generate signals
        for i in range(len(data)):
            if data["long_trigger"].iloc[i]:
                data.loc[data.index[i], "entry_signal"] = 1
                # Stop below swing low
                stop = data["swing_low"].iloc[i] - data["atr"].iloc[i] * 0.5
                data.loc[data.index[i], "stop_price"] = stop
                # Trailing stop distance
                data.loc[data.index[i], "trailing_stop_atr"] = data["atr"].iloc[i] * atr_multiplier
                
            elif data["short_trigger"].iloc[i]:
                data.loc[data.index[i], "entry_signal"] = -1
                # Stop above swing high
                stop = data["swing_high"].iloc[i] + data["atr"].iloc[i] * 0.5
                data.loc[data.index[i], "stop_price"] = stop
                # Trailing stop distance
                data.loc[data.index[i], "trailing_stop_atr"] = data["atr"].iloc[i] * atr_multiplier
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "HTF EMA", "column": "htf_ema", "color": "orange", "style": "solid"},
            {"name": "Slow EMA", "column": "slow_ema", "color": "blue", "style": "solid"},
            {"name": "Fast EMA", "column": "fast_ema", "color": "green", "style": "dashed"},
        ]
