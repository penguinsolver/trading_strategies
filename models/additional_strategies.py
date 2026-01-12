"""
Additional Trading Strategies - Designed to beat MA Crossover.

Includes:
1. Momentum Strategy - Pure price momentum
2. ADX Trend Filter - Only trade in strong trends  
3. Breakout Strategy - Trade range breakouts
4. Optimized MA Crossover - Tuned parameters
5. Dual Momentum - Absolute + relative momentum
"""
import pandas as pd
import numpy as np
from typing import Optional


class MomentumStrategy:
    """
    Pure Momentum Strategy.
    
    Goes long when price momentum is strongly positive,
    short when momentum is strongly negative.
    """
    
    def __init__(
        self,
        momentum_period: int = 20,
        momentum_threshold: float = 0.03,  # 3% momentum required
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @property
    def name(self) -> str:
        return "MOMENTUM"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        
        # Calculate momentum (rate of change)
        momentum = close.pct_change(self.momentum_period)
        
        # Calculate ATR for stops
        atr = (data["high"] - data["low"]).rolling(self.atr_period).mean()
        
        data["momentum"] = momentum
        data["entry_signal"] = 0
        
        # Long when momentum > threshold
        data.loc[momentum > self.momentum_threshold, "entry_signal"] = 1
        
        # Short when momentum < -threshold
        data.loc[momentum < -self.momentum_threshold, "entry_signal"] = -1
        
        # Add stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class ADXTrendStrategy:
    """
    ADX Trend Strength Strategy.
    
    Only trades in direction of trend when ADX indicates strong trend.
    Uses EMA crossover for direction, ADX for confirmation.
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25,
        fast_ema: int = 10,
        slow_ema: int = 30,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @property
    def name(self) -> str:
        return "ADX_TREND"
    
    def _calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX indicator."""
        high, low, close = data["high"], data["low"], data["close"]
        period = self.adx_period
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0
        
        # Smooth
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        
        # Calculate EMAs
        fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        
        # Calculate ADX
        adx, plus_di, minus_di = self._calculate_adx(data)
        
        # Calculate ATR
        atr = (data["high"] - data["low"]).rolling(self.atr_period).mean()
        
        data["adx"] = adx
        data["trend_up"] = fast > slow
        data["entry_signal"] = 0
        
        # Strong trend filter
        strong_trend = adx > self.adx_threshold
        
        # Long: uptrend + strong ADX + plus_di > minus_di
        long_mask = (fast > slow) & strong_trend & (plus_di > minus_di)
        
        # Short: downtrend + strong ADX + minus_di > plus_di
        short_mask = (fast < slow) & strong_trend & (minus_di > plus_di)
        
        # Only signal on crossovers
        trend_cross = (fast > slow) != (fast.shift(1) > slow.shift(1))
        
        data.loc[long_mask & trend_cross, "entry_signal"] = 1
        data.loc[short_mask & trend_cross, "entry_signal"] = -1
        
        # Add stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class BreakoutStrategy:
    """
    Range Breakout Strategy.
    
    Trades breakouts from defined price ranges (Donchian channels).
    """
    
    def __init__(
        self,
        lookback: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @property
    def name(self) -> str:
        return "BREAKOUT"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # Donchian channels
        upper = high.rolling(self.lookback).max()
        lower = low.rolling(self.lookback).min()
        
        # ATR for stops
        atr = (high - low).rolling(self.atr_period).mean()
        
        data["upper_band"] = upper
        data["lower_band"] = lower
        data["entry_signal"] = 0
        
        # Long: break above upper band
        long_break = close > upper.shift(1)
        
        # Short: break below lower band
        short_break = close < lower.shift(1)
        
        data.loc[long_break, "entry_signal"] = 1
        data.loc[short_break, "entry_signal"] = -1
        
        # Add stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class OptimizedMACrossover:
    """
    Optimized MA Crossover.
    
    Uses optimized EMA periods and adds trend confirmation.
    """
    
    def __init__(
        self,
        fast_period: int = 8,  # Faster than default
        slow_period: int = 21,  # Classic period
        trend_period: int = 50,  # Trend filter
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @property
    def name(self) -> str:
        return "OPTIMIZED_MA"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        
        # Calculate EMAs
        fast = close.ewm(span=self.fast_period, adjust=False).mean()
        slow = close.ewm(span=self.slow_period, adjust=False).mean()
        trend = close.ewm(span=self.trend_period, adjust=False).mean()
        
        # ATR for stops
        atr = (data["high"] - data["low"]).rolling(self.atr_period).mean()
        
        data["fast_ema"] = fast
        data["slow_ema"] = slow
        data["trend_ema"] = trend
        data["entry_signal"] = 0
        
        # Crossover detection
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        
        # Trend filter: only trade in direction of long-term trend
        uptrend = close > trend
        downtrend = close < trend
        
        # Long: cross up + in uptrend
        data.loc[cross_up & uptrend, "entry_signal"] = 1
        
        # Short: cross down + in downtrend
        data.loc[cross_down & downtrend, "entry_signal"] = -1
        
        # Add stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class DualMomentumStrategy:
    """
    Dual Momentum Strategy.
    
    Combines absolute momentum (is price trending up/down?)
    with relative momentum (comparing different lookback periods).
    """
    
    def __init__(
        self,
        short_period: int = 10,
        long_period: int = 30,
        threshold: float = 0.0,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @property
    def name(self) -> str:
        return "DUAL_MOMENTUM"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        
        # Short-term momentum
        short_mom = close.pct_change(self.short_period)
        
        # Long-term momentum
        long_mom = close.pct_change(self.long_period)
        
        # ATR for stops
        atr = (data["high"] - data["low"]).rolling(self.atr_period).mean()
        
        data["short_momentum"] = short_mom
        data["long_momentum"] = long_mom
        data["entry_signal"] = 0
        
        # Long: both momentums positive and aligned
        long_mask = (short_mom > self.threshold) & (long_mom > self.threshold)
        
        # Short: both momentums negative and aligned
        short_mask = (short_mom < -self.threshold) & (long_mom < -self.threshold)
        
        # Only signal when momentum flips
        prev_short = short_mom.shift(1)
        momentum_flip = (short_mom > 0) != (prev_short > 0)
        
        data.loc[long_mask & momentum_flip, "entry_signal"] = 1
        data.loc[short_mask & momentum_flip, "entry_signal"] = -1
        
        # Add stops
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class TrendFollowingSystem:
    """
    Complete Trend Following System.
    
    Combines multiple indicators for high-conviction trend trades:
    - EMA trend direction
    - ADX trend strength
    - Price above/below key levels
    """
    
    def __init__(
        self,
        ema_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.5,
    ):
        self.ema_period = ema_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @property
    def name(self) -> str:
        return "TREND_SYSTEM"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # EMA
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        
        # Simple ADX calculation
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr_val = tr.rolling(self.adx_period).mean()
        
        # Directional movement
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        
        plus_di = 100 * plus_dm.rolling(self.adx_period).mean() / (atr_val + 1e-10)
        minus_di = 100 * minus_dm.rolling(self.adx_period).mean() / (atr_val + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()
        
        # Price position relative to recent range
        recent_high = high.rolling(20).max()
        recent_low = low.rolling(20).min()
        price_position = (close - recent_low) / (recent_high - recent_low + 1e-10)
        
        data["ema"] = ema
        data["adx"] = adx
        data["entry_signal"] = 0
        
        # Long conditions:
        # - Price above EMA
        # - ADX above threshold (trending)
        # - Price in upper half of range
        # - Plus DI > Minus DI
        long_mask = (
            (close > ema) & 
            (adx > self.adx_threshold) & 
            (price_position > 0.6) &
            (plus_di > minus_di)
        )
        
        # Short conditions:
        # - Price below EMA
        # - ADX above threshold (trending)
        # - Price in lower half of range
        # - Minus DI > Plus DI
        short_mask = (
            (close < ema) & 
            (adx > self.adx_threshold) & 
            (price_position < 0.4) &
            (minus_di > plus_di)
        )
        
        # Only new signals
        prev_long = (close.shift(1) > ema.shift(1))
        prev_short = (close.shift(1) < ema.shift(1))
        
        data.loc[long_mask & ~prev_long, "entry_signal"] = 1
        data.loc[short_mask & ~prev_short, "entry_signal"] = -1
        
        # Stops
        atr = (high - low).rolling(self.atr_period).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data
