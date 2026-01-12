"""
Regime Filter - Filters trading signals based on market regime.

Uses ADX and volatility to detect trending vs ranging markets.
"""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    
    # ADX settings
    adx_period: int = 14
    adx_trend_threshold: float = 25.0  # ADX > this = trending
    adx_range_threshold: float = 20.0  # ADX < this = ranging
    
    # Volatility settings
    atr_period: int = 14
    volatility_lookback: int = 20
    volatility_expanding_threshold: float = 1.2  # Current ATR / avg ATR > this = expanding
    
    # Which regimes to allow trading
    allow_trending: bool = True
    allow_ranging: bool = True
    allow_expanding_vol: bool = True


class RegimeFilter:
    """
    Filter that detects market regime and controls trading.
    
    Supports:
    - Trending regime (ADX > threshold)
    - Ranging regime (ADX < threshold)
    - Volatility expansion detection
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
    
    def calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
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
        
        # When both are positive, keep the larger
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0
        
        # Smooth with EMA
        period = self.config.adx_period
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def calculate_volatility_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Calculate current volatility vs average volatility."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_period).mean()
        
        # Average ATR over lookback
        avg_atr = atr.rolling(self.config.volatility_lookback).mean()
        
        # Ratio
        return atr / (avg_atr + 1e-10)
    
    def detect_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime at each bar.
        
        Returns DataFrame with regime columns added:
        - regime: "trending", "ranging", or "neutral"
        - adx: ADX value
        - vol_ratio: Volatility ratio
        - allow_trade: Whether trading is allowed
        """
        data = data.copy()
        
        # Calculate indicators
        adx = self.calculate_adx(data)
        vol_ratio = self.calculate_volatility_ratio(data)
        
        # Detect regime
        regime = pd.Series("neutral", index=data.index)
        regime[adx > self.config.adx_trend_threshold] = "trending"
        regime[adx < self.config.adx_range_threshold] = "ranging"
        
        # Detect volatility expansion
        vol_expanding = vol_ratio > self.config.volatility_expanding_threshold
        
        # Determine if trading is allowed
        allow_trade = pd.Series(False, index=data.index)
        
        if self.config.allow_trending:
            allow_trade |= (regime == "trending")
        
        if self.config.allow_ranging:
            allow_trade |= (regime == "ranging")
        
        if self.config.allow_expanding_vol:
            allow_trade |= vol_expanding
        
        # Add to data
        data["regime"] = regime
        data["adx"] = adx
        data["vol_ratio"] = vol_ratio
        data["vol_expanding"] = vol_expanding
        data["allow_trade"] = allow_trade
        
        return data
    
    def filter_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter existing entry signals based on regime.
        
        Expects data to have "entry_signal" column.
        Zeroes out signals where regime doesn't allow trading.
        """
        if "entry_signal" not in data.columns:
            return data
        
        # Detect regime if not already done
        if "allow_trade" not in data.columns:
            data = self.detect_regime(data)
        
        # Filter signals
        data["entry_signal_unfiltered"] = data["entry_signal"]
        data.loc[~data["allow_trade"], "entry_signal"] = 0
        
        return data


class RegimeAwareStrategy:
    """
    Wrapper that applies regime filtering to any strategy.
    """
    
    def __init__(
        self,
        strategy: "Strategy",
        regime_filter: Optional[RegimeFilter] = None,
    ):
        self._strategy = strategy
        self.regime_filter = regime_filter or RegimeFilter()
    
    @property
    def name(self) -> str:
        return f"Regime_{self._strategy.name}"
    
    @property
    def description(self) -> str:
        return f"{self._strategy.description} (with regime filter)"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals, then filter by regime."""
        # First, detect regime
        data = self.regime_filter.detect_regime(data)
        
        # Generate signals from underlying strategy
        data = self._strategy.generate_signals(data)
        
        # Filter signals based on regime
        data = self.regime_filter.filter_signals(data)
        
        return data


class TrendOnlyFilter(RegimeFilter):
    """Shortcut filter that only allows trading in trending markets."""
    
    def __init__(self, adx_threshold: float = 25.0):
        config = RegimeConfig(
            adx_trend_threshold=adx_threshold,
            allow_trending=True,
            allow_ranging=False,
            allow_expanding_vol=False,
        )
        super().__init__(config)


class RangeOnlyFilter(RegimeFilter):
    """Shortcut filter that only allows trading in ranging markets."""
    
    def __init__(self, adx_threshold: float = 20.0):
        config = RegimeConfig(
            adx_range_threshold=adx_threshold,
            allow_trending=False,
            allow_ranging=True,
            allow_expanding_vol=False,
        )
        super().__init__(config)
