"""
Trend and momentum indicators.
"""
import pandas as pd
import numpy as np

from .moving_averages import ema
from .volatility import atr


def trend_direction(close: pd.Series, ma_period: int = 50) -> pd.Series:
    """
    Determine trend direction based on price vs moving average.
    
    Args:
        close: Close prices
        ma_period: Moving average period
        
    Returns:
        Series with values: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    ma = ema(close, ma_period)
    
    # Price above MA = bullish, below = bearish
    direction = pd.Series(0, index=close.index)
    direction[close > ma] = 1
    direction[close < ma] = -1
    
    return direction


def is_ranging(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_period: int = 14,
    lookback: int = 20,
    threshold: float = 0.5,
) -> pd.Series:
    """
    Detect range-bound conditions.
    
    A market is considered ranging when the price range is small
    relative to ATR-expected movement.
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        atr_period: ATR calculation period
        lookback: Period to measure range
        threshold: Range/ATR ratio threshold (lower = ranging)
        
    Returns:
        Boolean series (True = ranging)
    """
    atr_val = atr(high, low, close, atr_period)
    
    # Calculate range over lookback period
    rolling_high = high.rolling(lookback).max()
    rolling_low = low.rolling(lookback).min()
    price_range = rolling_high - rolling_low
    
    # Expected range based on ATR
    expected_range = atr_val * np.sqrt(lookback)  # Scale by sqrt for volatility
    
    # Ranging when actual range is small vs expected
    ratio = price_range / expected_range
    return ratio < threshold


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        close: Close prices
        period: RSI period
        
    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = losses.abs()
    
    # Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gains / avg_losses
    rsi_val = 100 - (100 / (1 + rs))
    
    return rsi_val


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
        
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Smooth with Wilder's method
    atr_val = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr_val
    minus_di = 100 * smooth_minus_dm / atr_val
    
    # Calculate DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx_val, plus_di, minus_di


def swing_high(high: pd.Series, left_bars: int = 2, right_bars: int = 2) -> pd.Series:
    """
    Identify swing highs.
    
    A swing high is a bar whose high is higher than the highs
    of `left_bars` bars to the left and `right_bars` bars to the right.
    
    Returns:
        Series with swing high price where detected, NaN otherwise
    """
    result = pd.Series(np.nan, index=high.index)
    
    for i in range(left_bars, len(high) - right_bars):
        is_swing = True
        current_high = high.iloc[i]
        
        # Check left bars
        for j in range(1, left_bars + 1):
            if high.iloc[i - j] >= current_high:
                is_swing = False
                break
        
        # Check right bars
        if is_swing:
            for j in range(1, right_bars + 1):
                if high.iloc[i + j] >= current_high:
                    is_swing = False
                    break
        
        if is_swing:
            result.iloc[i] = current_high
    
    return result


def swing_low(low: pd.Series, left_bars: int = 2, right_bars: int = 2) -> pd.Series:
    """
    Identify swing lows.
    
    A swing low is a bar whose low is lower than the lows
    of `left_bars` bars to the left and `right_bars` bars to the right.
    
    Returns:
        Series with swing low price where detected, NaN otherwise
    """
    result = pd.Series(np.nan, index=low.index)
    
    for i in range(left_bars, len(low) - right_bars):
        is_swing = True
        current_low = low.iloc[i]
        
        # Check left bars
        for j in range(1, left_bars + 1):
            if low.iloc[i - j] <= current_low:
                is_swing = False
                break
        
        # Check right bars
        if is_swing:
            for j in range(1, right_bars + 1):
                if low.iloc[i + j] <= current_low:
                    is_swing = False
                    break
        
        if is_swing:
            result.iloc[i] = current_low
    
    return result


def recent_swing_low(low: pd.Series, lookback: int = 10) -> pd.Series:
    """
    Get the most recent swing low within lookback period.
    
    Simplified version that finds the lowest low in the lookback period.
    """
    return low.rolling(window=lookback, min_periods=1).min()


def recent_swing_high(high: pd.Series, lookback: int = 10) -> pd.Series:
    """
    Get the most recent swing high within lookback period.
    
    Simplified version that finds the highest high in the lookback period.
    """
    return high.rolling(window=lookback, min_periods=1).max()


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend indicator.
    
    Supertrend is a trend-following indicator that uses ATR to create
    dynamic support/resistance bands. When price closes above/below
    the band, it flips direction.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_period: ATR calculation period
        multiplier: ATR multiplier for band width
        
    Returns:
        Tuple of (supertrend_line, direction) where:
        - supertrend_line: The Supertrend level
        - direction: 1 for bullish (price above), -1 for bearish (price below)
    """
    # Calculate ATR
    atr_val = atr(high, low, close, atr_period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr_val
    basic_lower = hl2 - multiplier * atr_val
    
    # Initialize output series
    n = len(close)
    final_upper = pd.Series(np.nan, index=close.index)
    final_lower = pd.Series(np.nan, index=close.index)
    supertrend_line = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)  # 1 = bullish, -1 = bearish
    
    # First valid index
    start_idx = atr_period
    if start_idx >= n:
        return supertrend_line, direction
    
    # Initialize first values
    final_upper.iloc[start_idx] = basic_upper.iloc[start_idx]
    final_lower.iloc[start_idx] = basic_lower.iloc[start_idx]
    
    # Calculate final bands with locking logic
    for i in range(start_idx + 1, n):
        # Final Upper Band: lock to lower value if prev close was above
        if close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i - 1])
        
        # Final Lower Band: lock to higher value if prev close was below
        if close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i - 1])
    
    # Determine Supertrend and direction
    for i in range(start_idx, n):
        if i == start_idx:
            # Initialize based on close vs bands
            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend_line.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend_line.iloc[i] = final_upper.iloc[i]
        else:
            prev_dir = direction.iloc[i - 1]
            
            if prev_dir == 1:  # Was bullish
                if close.iloc[i] < final_lower.iloc[i]:
                    # Flip to bearish
                    direction.iloc[i] = -1
                    supertrend_line.iloc[i] = final_upper.iloc[i]
                else:
                    # Stay bullish
                    direction.iloc[i] = 1
                    supertrend_line.iloc[i] = final_lower.iloc[i]
            else:  # Was bearish
                if close.iloc[i] > final_upper.iloc[i]:
                    # Flip to bullish
                    direction.iloc[i] = 1
                    supertrend_line.iloc[i] = final_lower.iloc[i]
                else:
                    # Stay bearish
                    direction.iloc[i] = -1
                    supertrend_line.iloc[i] = final_upper.iloc[i]
    
    return supertrend_line, direction


def chandelier_stop(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 22,
    atr_period: int = 14,
    mult: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Chandelier Exit stop levels.
    
    The Chandelier Exit is a volatility-based trailing stop that hangs
    from the highest high (for longs) or lowest low (for shorts).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        n: Lookback period for highest high / lowest low
        atr_period: ATR calculation period
        mult: ATR multiplier for stop distance
        
    Returns:
        Tuple of (long_stop, short_stop):
        - long_stop: Highest high(n) - ATR * mult (stop for long positions)
        - short_stop: Lowest low(n) + ATR * mult (stop for short positions)
    """
    # Calculate ATR
    atr_val = atr(high, low, close, atr_period)
    
    # Highest high and lowest low over lookback period
    highest_high = high.rolling(window=n, min_periods=n).max()
    lowest_low = low.rolling(window=n, min_periods=n).min()
    
    # Calculate stops
    long_stop = highest_high - (atr_val * mult)
    short_stop = lowest_low + (atr_val * mult)
    
    return long_stop, short_stop


def linear_regression_slope(
    series: pd.Series,
    period: int = 50,
) -> pd.Series:
    """
    Calculate rolling linear regression slope.
    
    Uses OLS to fit a line to the last 'period' values and returns the slope.
    Positive slope indicates uptrend, negative indicates downtrend.
    The slope is normalized by the mean price to make it comparable across assets.
    
    Args:
        series: Price series (typically close)
        period: Lookback period for regression
        
    Returns:
        Series of normalized slopes (percentage change per bar)
    """
    def calc_slope(window):
        if len(window) < period:
            return np.nan
        # X values: 0, 1, 2, ..., period-1
        x = np.arange(len(window))
        y = window.values
        
        # OLS: slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize by mean price to get percentage slope
        if y_mean != 0:
            slope = (slope / y_mean) * 100
        
        return slope
    
    return series.rolling(window=period, min_periods=period).apply(calc_slope, raw=False)


def ema_slope(
    series: pd.Series,
    ema_period: int = 200,
    slope_period: int = 10,
) -> pd.Series:
    """
    Calculate EMA slope direction.
    
    Returns the rate of change of EMA over slope_period bars,
    normalized by ATR for comparability.
    
    Args:
        series: Price series (typically close)
        ema_period: EMA calculation period
        slope_period: Period over which to measure slope
        
    Returns:
        Slope of EMA (positive = rising, negative = falling)
    """
    ema_val = ema(series, ema_period)
    slope = ema_val.diff(slope_period) / slope_period
    
    # Normalize by EMA value
    return (slope / ema_val) * 100


def zscore(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate rolling Z-score.
    
    Z-score measures how many standard deviations the current value
    is from the rolling mean.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        Z-score series
    """
    mean = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    
    # Avoid division by zero
    std = std.replace(0, np.nan)
    
    return (series - mean) / std


def choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Choppiness Index.
    
    The Choppiness Index is a volatility indicator that measures whether
    the market is trending or ranging (choppy).
    
    - Values near 100 indicate choppy, sideways market
    - Values near 0 indicate strong trending market
    - Typical thresholds: >61.8 = choppy, <38.2 = trending
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (typically 14)
        
    Returns:
        Choppiness Index series (0-100 scale)
    """
    # Calculate True Range for each bar
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = true_range.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    
    # Range
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price_range = price_range.replace(0, np.nan)
    
    # Choppiness Index formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop


def atr_slope(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    slope_period: int = 5,
) -> pd.Series:
    """
    Calculate ATR slope - indicates if volatility is expanding or contracting.
    
    Positive slope = volatility expanding
    Negative slope = volatility contracting
    Near zero = volatility flat
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_period: ATR calculation period
        slope_period: Period over which to measure slope
        
    Returns:
        ATR slope normalized by ATR value (percentage change)
    """
    atr_val = atr(high, low, close, atr_period)
    slope = atr_val.diff(slope_period) / slope_period
    
    # Normalize by ATR value
    return (slope / atr_val) * 100
