"""
Aggressive Multi-Signal Strategy.

Generates more signals by combining multiple indicators to find every possible opportunity.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AggressiveConfig:
    """Config for aggressive strategy."""
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    fast_ma: int = 5
    slow_ma: int = 15
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_multiplier: float = 1.5


class AggressiveMultiSignal:
    """
    Aggressive strategy that generates signals from multiple indicators.
    
    Signals from:
    1. MA Crossover
    2. RSI Extremes
    3. Bollinger Band Touches
    4. Breakouts
    
    Takes any signal that appears.
    """
    
    def __init__(self, config: Optional[AggressiveConfig] = None):
        self.config = config or AggressiveConfig()
        self.name = "AGGRESSIVE_MULTI_SIGNAL"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        c = self.config
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # 1. MA Crossover
        fast_ma = close.ewm(span=c.fast_ma, adjust=False).mean()
        slow_ma = close.ewm(span=c.slow_ma, adjust=False).mean()
        ma_cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        ma_cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # 2. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(c.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(c.rsi_period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        rsi_oversold = (rsi < c.rsi_oversold) & (rsi.shift(1) >= c.rsi_oversold)
        rsi_overbought = (rsi > c.rsi_overbought) & (rsi.shift(1) <= c.rsi_overbought)
        
        # 3. Bollinger Bands
        bb_middle = close.rolling(c.bb_period).mean()
        bb_std = close.rolling(c.bb_period).std()
        bb_upper = bb_middle + c.bb_std * bb_std
        bb_lower = bb_middle - c.bb_std * bb_std
        
        bb_touch_lower = close <= bb_lower
        bb_touch_upper = close >= bb_upper
        
        # 4. Breakout
        upper = high.rolling(10).max().shift(1)
        lower = low.rolling(10).min().shift(1)
        breakout_up = close > upper
        breakout_down = close < lower
        
        # ATR for stops
        atr = (high - low).rolling(c.atr_period).mean()
        
        # Combine signals - any signal triggers
        data["entry_signal"] = 0
        
        # Long signals
        long_mask = ma_cross_up | rsi_oversold | bb_touch_lower | breakout_up
        
        # Short signals
        short_mask = ma_cross_down | rsi_overbought | bb_touch_upper | breakout_down
        
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        # Prioritize breakouts
        data.loc[breakout_up, "entry_signal"] = 1
        data.loc[breakout_down, "entry_signal"] = -1
        
        # Stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - c.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + c.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class CompoundingStrategy:
    """
    Strategy that simulates compounding by tracking running equity.
    
    This is a meta-strategy that wraps another strategy.
    """
    
    def __init__(self, base_strategy, compound_factor: float = 1.0):
        self.base_strategy = base_strategy
        self.compound_factor = compound_factor
        self.name = f"COMPOUND_{base_strategy.name}"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Just pass through - compounding happens in backtest sizing
        return self.base_strategy.generate_signals(data)


class TrendBurstStrategy:
    """
    Catches trend bursts - strong moves after consolidation.
    """
    
    def __init__(self, lookback: int = 10, atr_mult: float = 1.5, vol_mult: float = 1.5):
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.vol_mult = vol_mult
        self.name = "TREND_BURST"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # ATR for volatility
        atr = (high - low).rolling(14).mean()
        avg_atr = atr.rolling(self.lookback).mean()
        
        # Detect consolidation then expansion
        volatility_low = atr < avg_atr * 0.8  # Low volatility
        volatility_burst = atr > avg_atr * self.vol_mult  # Sudden expansion
        
        # Direction of burst
        trend_up = close > close.shift(self.lookback)
        trend_down = close < close.shift(self.lookback)
        
        # Consolidation then burst
        was_consolidating = volatility_low.shift(1).rolling(3).sum() >= 2
        
        data["entry_signal"] = 0
        
        long_burst = volatility_burst & trend_up & was_consolidating
        short_burst = volatility_burst & trend_down & was_consolidating
        
        data.loc[long_burst, "entry_signal"] = 1
        data.loc[short_burst, "entry_signal"] = -1
        
        # Stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_mult * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_mult * atr,
                np.nan
            )
        )
        
        return data
