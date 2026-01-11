"""
VWAP (Volume Weighted Average Price) indicator.
"""
import pandas as pd
import numpy as np


def vwap(df: pd.DataFrame, anchor: str = "session") -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        anchor: Reset point - "session" (daily) or "none" (cumulative)
        
    Returns:
        VWAP series
    """
    # Typical price
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    
    if anchor == "none":
        # Cumulative VWAP (no reset)
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol
    
    # Session-anchored VWAP (reset daily)
    # Detect session boundaries
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df.index).date
    
    session_groups = pd.Series(dates, index=df.index)
    
    # Calculate VWAP per session
    tp_vol = typical_price * df["volume"]
    
    result = pd.Series(index=df.index, dtype=float)
    
    for session in session_groups.unique():
        mask = session_groups == session
        session_tp_vol = tp_vol[mask].cumsum()
        session_vol = df["volume"][mask].cumsum()
        result[mask] = session_tp_vol / session_vol
    
    return result


def vwap_bands(df: pd.DataFrame, stdev_mult: float = 1.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate VWAP with standard deviation bands.
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        stdev_mult: Standard deviation multiplier for bands
        
    Returns:
        Tuple of (vwap, upper_band, lower_band)
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    
    # Session detection
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df.index).date
    
    session_groups = pd.Series(dates, index=df.index)
    
    vwap_line = pd.Series(index=df.index, dtype=float)
    upper = pd.Series(index=df.index, dtype=float)
    lower = pd.Series(index=df.index, dtype=float)
    
    for session in session_groups.unique():
        mask = session_groups == session
        session_df = df[mask]
        session_tp = typical_price[mask]
        session_vol = session_df["volume"]
        
        # Cumulative values for this session
        tp_vol_cum = (session_tp * session_vol).cumsum()
        vol_cum = session_vol.cumsum()
        session_vwap = tp_vol_cum / vol_cum
        
        # Variance calculation
        tp_squared_vol_cum = ((session_tp ** 2) * session_vol).cumsum()
        variance = (tp_squared_vol_cum / vol_cum) - (session_vwap ** 2)
        stdev = np.sqrt(variance.clip(lower=0))
        
        vwap_line[mask] = session_vwap
        upper[mask] = session_vwap + stdev_mult * stdev
        lower[mask] = session_vwap - stdev_mult * stdev
    
    return vwap_line, upper, lower
