"""
Moving average indicators.
"""
import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        SMA series
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        EMA series (uses span parameter for calculation)
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Weighted Moving Average.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        WMA series (linear weights)
    """
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda x: np.sum(weights * x) / weights.sum(),
        raw=True
    )


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Detect bullish crossover (fast crosses above slow).
    
    Returns:
        Boolean series, True when crossover occurs
    """
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Detect bearish crossunder (fast crosses below slow).
    
    Returns:
        Boolean series, True when crossunder occurs
    """
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))
