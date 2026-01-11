"""
Strategy Wrappers

Utility wrappers that can modify any strategy's behavior:
1. Time-of-day filter: Only trade during specific hours
2. Volatility targeting: Scale position size based on volatility
"""
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from indicators import atr

if TYPE_CHECKING:
    from .base import Strategy


def apply_time_filter(
    data: pd.DataFrame,
    start_hour_utc: int = 13,
    end_hour_utc: int = 17,
) -> pd.DataFrame:
    """
    Apply time-of-day filter to entry signals.
    
    Only allows entries during specified UTC hours.
    Default is London+NY overlap (13:00-17:00 UTC).
    
    Args:
        data: DataFrame with entry_signal column
        start_hour_utc: First hour of trading window (inclusive)
        end_hour_utc: Last hour of trading window (exclusive)
        
    Returns:
        DataFrame with filtered entry signals
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data  # Can't filter without datetime index
    
    # Get hour from index
    hours = data.index.hour
    
    # Create mask for valid trading hours
    if start_hour_utc <= end_hour_utc:
        valid_hours = (hours >= start_hour_utc) & (hours < end_hour_utc)
    else:
        # Handle overnight window (e.g., 22:00 to 06:00)
        valid_hours = (hours >= start_hour_utc) | (hours < end_hour_utc)
    
    # Zero out entry signals outside valid hours
    if "entry_signal" in data.columns:
        data.loc[~valid_hours, "entry_signal"] = 0
    
    return data


def apply_volatility_sizing(
    data: pd.DataFrame,
    base_risk: float = 0.01,
    min_risk: float = 0.005,
    max_risk: float = 0.02,
    atr_baseline_pct: float = 2.0,
) -> pd.DataFrame:
    """
    Apply volatility targeting to position sizing.
    
    Scales risk down when volatility (ATR%) is high, up when low.
    Keeps risk within min/max bounds.
    
    Args:
        data: DataFrame with entry signals and ATR
        base_risk: Base risk per trade (e.g., 0.01 = 1%)
        min_risk: Minimum risk allowed
        max_risk: Maximum risk allowed
        atr_baseline_pct: Baseline ATR% for normal sizing
        
    Returns:
        DataFrame with adjusted_risk column
    """
    # Calculate ATR% if not present
    if "atr" not in data.columns:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
    
    data["atr_pct"] = (data["atr"] / data["close"]) * 100
    
    # Calculate scaling factor: higher vol = lower risk
    # scaling = baseline / current (capped)
    scaling = atr_baseline_pct / data["atr_pct"].replace(0, atr_baseline_pct)
    
    # Calculate adjusted risk
    data["adjusted_risk"] = base_risk * scaling
    
    # Clamp to min/max
    data["adjusted_risk"] = data["adjusted_risk"].clip(min_risk, max_risk)
    
    return data


class TimeFilteredStrategy:
    """
    Wrapper that applies time-of-day filtering to any strategy.
    
    Usage:
        base_strategy = SupertrendStrategy()
        filtered = TimeFilteredStrategy(base_strategy, start_hour=13, end_hour=17)
        result = filtered.generate_signals(data)
    """
    
    def __init__(
        self,
        strategy: "Strategy",
        start_hour_utc: int = 13,
        end_hour_utc: int = 17,
    ):
        self.strategy = strategy
        self.start_hour_utc = start_hour_utc
        self.end_hour_utc = end_hour_utc
    
    @property
    def name(self) -> str:
        return f"{self.strategy.name} (Time Filtered)"
    
    @property
    def description(self) -> str:
        return f"{self.strategy.description} Filtered to {self.start_hour_utc}:00-{self.end_hour_utc}:00 UTC."
    
    def get_param_config(self):
        return self.strategy.get_param_config()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Generate base signals
        result = self.strategy.generate_signals(data)
        # Apply time filter
        return apply_time_filter(result, self.start_hour_utc, self.end_hour_utc)
    
    def get_indicator_info(self):
        return self.strategy.get_indicator_info()


class VolatilityTargetedStrategy:
    """
    Wrapper that applies volatility-based sizing to any strategy.
    
    Usage:
        base_strategy = SupertrendStrategy()
        vol_sized = VolatilityTargetedStrategy(base_strategy)
        # Use result.adjusted_risk for position sizing
    """
    
    def __init__(
        self,
        strategy: "Strategy",
        base_risk: float = 0.01,
        min_risk: float = 0.005,
        max_risk: float = 0.02,
    ):
        self.strategy = strategy
        self.base_risk = base_risk
        self.min_risk = min_risk
        self.max_risk = max_risk
    
    @property
    def name(self) -> str:
        return f"{self.strategy.name} (Vol Targeted)"
    
    @property
    def description(self) -> str:
        return f"{self.strategy.description} With volatility-targeted position sizing."
    
    def get_param_config(self):
        return self.strategy.get_param_config()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Generate base signals
        result = self.strategy.generate_signals(data)
        # Apply volatility sizing
        return apply_volatility_sizing(
            result,
            self.base_risk,
            self.min_risk,
            self.max_risk,
        )
    
    def get_indicator_info(self):
        return self.strategy.get_indicator_info()
