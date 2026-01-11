"""
Volatility indicators (ATR, Donchian channels, etc.).
"""
import pandas as pd
import numpy as np


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate True Range.
    
    TR = max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close)
    )
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
        
    Returns:
        ATR series (uses Wilder's smoothing / EMA)
    """
    tr = true_range(high, low, close)
    # Wilder's smoothing is equivalent to EMA with alpha = 1/period
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channels.
    
    Args:
        high: High prices
        low: Low prices
        period: Lookback period
        
    Returns:
        Tuple of (upper, lower, mid) channels
    """
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2
    return upper, lower, mid


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        series: Price series (typically close)
        period: SMA period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper, lower, middle) bands
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, lower, middle


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: EMA period for middle line
        atr_period: ATR period
        atr_mult: ATR multiplier for bands
        
    Returns:
        Tuple of (upper, lower, middle) bands
    """
    middle = close.ewm(span=ema_period, adjust=False).mean()
    atr_val = atr(high, low, close, atr_period)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return upper, lower, middle


def bollinger_bandwidth(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """
    Calculate Bollinger Bandwidth.
    
    Bandwidth measures volatility as the percentage difference between
    upper and lower bands. Low bandwidth indicates a squeeze (low volatility).
    
    Args:
        series: Price series (typically close)
        period: SMA period
        std_dev: Standard deviation multiplier
        
    Returns:
        Bandwidth as (upper - lower) / middle * 100
    """
    upper, lower, middle = bollinger_bands(series, period, std_dev)
    return ((upper - lower) / middle) * 100


def bollinger_bandwidth_percentile(
    bandwidth: pd.Series,
    lookback: int = 50,
) -> pd.Series:
    """
    Calculate the percentile rank of current bandwidth over a lookback period.
    
    A low percentile (e.g., < 20) indicates bandwidth is historically low,
    signaling a potential squeeze condition.
    
    Args:
        bandwidth: Bollinger Bandwidth series
        lookback: Lookback period for percentile calculation
        
    Returns:
        Percentile rank (0-100) of current bandwidth
    """
    def rolling_percentile(x):
        if len(x) < 2:
            return 50.0
        current = x.iloc[-1]
        return (x.iloc[:-1] < current).sum() / (len(x) - 1) * 100
    
    return bandwidth.rolling(window=lookback, min_periods=lookback).apply(
        rolling_percentile, raw=False
    )

