"""
Strategy: Parabolic SAR Trend Following

Uses Parabolic SAR for trend direction and trailing stops.
SAR provides built-in stop levels that accelerate with trend.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR."""
    length = len(high)
    sar = np.zeros(length)
    ep = np.zeros(length)  # Extreme point
    af = np.zeros(length)  # Acceleration factor
    trend = np.zeros(length)  # 1 = uptrend, -1 = downtrend
    
    # Initialize
    sar[0] = low.iloc[0]
    ep[0] = high.iloc[0]
    af[0] = af_start
    trend[0] = 1
    
    for i in range(1, length):
        # Previous values
        prev_sar = sar[i-1]
        prev_ep = ep[i-1]
        prev_af = af[i-1]
        prev_trend = trend[i-1]
        
        if prev_trend == 1:  # Uptrend
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            # SAR cannot be above prior two lows
            sar[i] = min(sar[i], low.iloc[i-1], low.iloc[max(0, i-2)])
            
            if high.iloc[i] > prev_ep:
                ep[i] = high.iloc[i]
                af[i] = min(prev_af + af_step, af_max)
            else:
                ep[i] = prev_ep
                af[i] = prev_af
            
            if low.iloc[i] < sar[i]:  # Trend reversal
                trend[i] = -1
                sar[i] = prev_ep
                ep[i] = low.iloc[i]
                af[i] = af_start
            else:
                trend[i] = 1
        else:  # Downtrend
            sar[i] = prev_sar - prev_af * (prev_sar - prev_ep)
            # SAR cannot be below prior two highs
            sar[i] = max(sar[i], high.iloc[i-1], high.iloc[max(0, i-2)])
            
            if low.iloc[i] < prev_ep:
                ep[i] = low.iloc[i]
                af[i] = min(prev_af + af_step, af_max)
            else:
                ep[i] = prev_ep
                af[i] = prev_af
            
            if high.iloc[i] > sar[i]:  # Trend reversal
                trend[i] = 1
                sar[i] = prev_ep
                ep[i] = high.iloc[i]
                af[i] = af_start
            else:
                trend[i] = -1
    
    return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)


class ParabolicSARStrategy(Strategy):
    """Parabolic SAR trend following strategy."""
    
    @property
    def name(self) -> str:
        return "Parabolic SAR"
    
    @property
    def description(self) -> str:
        return "Follow trends using Parabolic SAR with built-in acceleration and reversal detection."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="af_start",
                label="AF Start",
                param_type="float",
                default=0.02,
                min_value=0.01,
                max_value=0.05,
                step=0.01,
                help_text="Starting acceleration factor",
            ),
            ParamConfig(
                name="af_max",
                label="AF Max",
                param_type="float",
                default=0.2,
                min_value=0.1,
                max_value=0.3,
                step=0.05,
                help_text="Maximum acceleration factor",
            ),
            ParamConfig(
                name="ema_filter",
                label="EMA Filter",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=10,
                help_text="EMA period for trend filter (0=disabled)",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Parabolic SAR signals."""
        af_start = self.params.get("af_start", 0.02)
        af_max = self.params.get("af_max", 0.2)
        ema_filter = self.params.get("ema_filter", 50)
        
        # Calculate SAR
        sar, trend = parabolic_sar(data["high"], data["low"], af_start, 0.02, af_max)
        data["sar"] = sar
        data["sar_trend"] = trend
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # EMA trend filter
        if ema_filter > 0:
            data["ema_filter"] = ema(data["close"], ema_filter)
            above_ema = data["close"] > data["ema_filter"]
            below_ema = data["close"] < data["ema_filter"]
        else:
            above_ema = pd.Series(True, index=data.index)
            below_ema = pd.Series(True, index=data.index)
        
        # Trend changes with shift
        prev_trend = data["sar_trend"].shift(1)
        
        # Long: SAR flips to uptrend and above EMA
        long_cond = (
            (data["sar_trend"] == 1) &
            (prev_trend == -1) &
            above_ema
        )
        
        # Short: SAR flips to downtrend and below EMA
        short_cond = (
            (data["sar_trend"] == -1) &
            (prev_trend == 1) &
            below_ema
        )
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries - use SAR as stop
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "sar"]
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "sar"]
        
        # Exit on SAR flip
        sar_flip_down = (data["sar_trend"] == -1) & (prev_trend == 1)
        sar_flip_up = (data["sar_trend"] == 1) & (prev_trend == -1)
        data.loc[sar_flip_down | sar_flip_up, "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "SAR", "column": "sar", "color": "purple", "style": "dotted"},
        ]
