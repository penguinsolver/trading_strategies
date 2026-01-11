"""Tests for indicator functions."""
import pytest
import pandas as pd
import numpy as np

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators.moving_averages import sma, ema, crossover, crossunder
from indicators.volatility import atr, true_range, donchian_channels
from indicators.trend import rsi, trend_direction


class TestMovingAverages:
    """Tests for moving average indicators."""
    
    def test_sma_calculation(self):
        """Test SMA calculates correctly."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sma(series, period=3)
        
        # First 2 values should be NaN (not enough data)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        
        # SMA(3) at index 2 should be (1+2+3)/3 = 2
        assert result.iloc[2] == pytest.approx(2.0)
        
        # SMA(3) at index 3 should be (2+3+4)/3 = 3
        assert result.iloc[3] == pytest.approx(3.0)
    
    def test_ema_calculation(self):
        """Test EMA calculates correctly."""
        series = pd.Series([10, 12, 11, 13, 14, 12, 15, 16, 14, 17])
        result = ema(series, period=3)
        
        # First values should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        
        # EMA should be calculated from period onwards
        assert not pd.isna(result.iloc[2])
        
        # EMA should react to price changes
        assert result.iloc[-1] != result.iloc[-2]
    
    def test_crossover_detection(self):
        """Test crossover detection."""
        fast = pd.Series([1, 2, 3, 4, 5])
        slow = pd.Series([2, 2, 2, 2, 2])
        
        result = crossover(fast, slow)
        
        # Crossover should occur when fast goes from <= slow to > slow
        assert result.iloc[2] == True  # 3 > 2 and 2 <= 2
        assert result.iloc[3] == False  # Already above
    
    def test_crossunder_detection(self):
        """Test crossunder detection."""
        fast = pd.Series([5, 4, 3, 2, 1])
        slow = pd.Series([3, 3, 3, 3, 3])
        
        result = crossunder(fast, slow)
        
        # Crossunder at index 3 (2 < 3 and 3 >= 3)
        assert result.iloc[3] == True


class TestVolatility:
    """Tests for volatility indicators."""
    
    def test_true_range(self):
        """Test true range calculation."""
        high = pd.Series([105, 110, 108, 112])
        low = pd.Series([95, 100, 102, 105])
        close = pd.Series([100, 105, 106, 110])
        
        tr = true_range(high, low, close)
        
        # First TR is just high - low = 105 - 95 = 10
        assert tr.iloc[0] == pytest.approx(10.0)
        
        # Second TR should consider previous close
        # max(110-100, |110-100|, |100-100|) = 10
        assert tr.iloc[1] == pytest.approx(10.0)
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        np.random.seed(42)
        n = 50
        high = pd.Series(100 + np.random.randn(n).cumsum() + 2)
        low = high - 3 - np.random.rand(n)
        close = (high + low) / 2
        
        result = atr(high, low, close, period=14)
        
        # First 13 values should be NaN
        assert pd.isna(result.iloc[0])
        
        # ATR should be positive
        valid_atr = result.dropna()
        assert (valid_atr > 0).all()
    
    def test_donchian_channels(self):
        """Test Donchian channel calculation."""
        high = pd.Series([10, 12, 11, 15, 14, 13, 16, 15, 14, 17])
        low = pd.Series([8, 9, 10, 12, 11, 10, 13, 12, 11, 14])
        
        upper, lower, mid = donchian_channels(high, low, period=5)
        
        # At index 4 (5 bars), upper should be max of first 5 highs
        assert upper.iloc[4] == pytest.approx(15.0)  # max(10,12,11,15,14)
        
        # Lower should be min of first 5 lows
        assert lower.iloc[4] == pytest.approx(8.0)  # min(8,9,10,12,11)
        
        # Mid should be average of upper and lower
        assert mid.iloc[4] == pytest.approx((15.0 + 8.0) / 2)


class TestTrend:
    """Tests for trend indicators."""
    
    def test_rsi_boundaries(self):
        """Test RSI stays within 0-100."""
        np.random.seed(42)
        close = pd.Series(100 + np.random.randn(100).cumsum())
        
        result = rsi(close, period=14)
        valid_rsi = result.dropna()
        
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_rsi_extreme_moves(self):
        """Test RSI responds to extreme moves."""
        # Strong uptrend
        uptrend = pd.Series([100 + i for i in range(30)])
        rsi_up = rsi(uptrend, period=14)
        
        # Should be overbought
        assert rsi_up.iloc[-1] > 70
        
        # Strong downtrend
        downtrend = pd.Series([100 - i for i in range(30)])
        rsi_down = rsi(downtrend, period=14)
        
        # Should be oversold
        assert rsi_down.iloc[-1] < 30
    
    def test_trend_direction(self):
        """Test trend direction detection."""
        # Uptrend: price consistently above its average
        close = pd.Series([100 + i * 2 for i in range(60)])
        direction = trend_direction(close, ma_period=20)
        
        # Later values should show bullish
        assert direction.iloc[-1] == 1
        
        # Downtrend
        close_down = pd.Series([200 - i * 2 for i in range(60)])
        direction_down = trend_direction(close_down, ma_period=20)
        
        # Should show bearish
        assert direction_down.iloc[-1] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
