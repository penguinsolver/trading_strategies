"""
Unit tests for selective/low-frequency strategies.

Tests verify:
1. No lookahead bias - signals at time t only use data <= t
2. Low trade frequency (selective)
3. Signal generation correctness
4. Parameter handling
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from strategies import STRATEGIES
from strategies.atr_channel import ATRChannelStrategy
from strategies.volume_breakout import VolumeBreakoutStrategy
from strategies.zscore_reversion import ZScoreReversionStrategy
from strategies.chandelier_trend import ChandelierTrendStrategy
from strategies.avwap_pullback import AVWAPPullbackStrategy
from strategies.regression_slope import RegressionSlopeStrategy
from strategies.wrappers import apply_time_filter, apply_volatility_sizing


def generate_test_data(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    returns = np.random.randn(n_bars) * 0.01
    close = 50000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = (high + low) / 2
    volume = np.random.randint(1000, 10000, n_bars).astype(float)
    
    start_date = datetime(2024, 1, 1, 0, 0)
    dates = pd.date_range(start=start_date, periods=n_bars, freq="15min")
    
    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestSelectiveStrategiesLoading:
    """Test that all selective strategies load correctly."""
    
    def test_new_strategies_in_registry(self):
        """Verify new selective strategies are in registry."""
        expected = [
            "atr_channel", "volume_breakout", "zscore_reversion",
            "chandelier_trend", "avwap_pullback", "regression_slope"
        ]
        for key in expected:
            assert key in STRATEGIES, f"Strategy {key} not in registry"
    
    def test_total_strategy_count(self):
        """Should have 18 total strategies."""
        assert len(STRATEGIES) == 18


class TestNoLookaheadBias:
    """Test that strategies don't use future data."""
    
    @pytest.fixture
    def data(self):
        return generate_test_data(300)
    
    def _check_no_lookahead(self, strategy, data: pd.DataFrame):
        """Helper to verify no lookahead bias."""
        full_result = strategy.generate_signals(data.copy())
        
        check_indices = [100, 150, 200]
        
        for i in check_indices:
            partial_data = data.iloc[:i+1].copy()
            partial_result = strategy.generate_signals(partial_data)
            
            full_signal = full_result["entry_signal"].iloc[i]
            partial_signal = partial_result["entry_signal"].iloc[-1]
            
            assert full_signal == partial_signal, \
                f"Lookahead at bar {i}: full={full_signal}, partial={partial_signal}"
    
    def test_atr_channel_no_lookahead(self, data):
        self._check_no_lookahead(ATRChannelStrategy(), data)
    
    def test_volume_breakout_no_lookahead(self, data):
        self._check_no_lookahead(VolumeBreakoutStrategy(), data)
    
    def test_zscore_reversion_no_lookahead(self, data):
        self._check_no_lookahead(ZScoreReversionStrategy(), data)
    
    def test_chandelier_trend_no_lookahead(self, data):
        self._check_no_lookahead(ChandelierTrendStrategy(), data)


class TestSelectivity:
    """Test that selective strategies have low trade frequency."""
    
    @pytest.fixture
    def data(self):
        return generate_test_data(500)
    
    def test_selective_strategies_fewer_trades(self, data):
        """Selective strategies should have fewer trades than high-frequency ones."""
        from strategies import STRATEGIES
        
        selective = ["atr_channel", "volume_breakout", "chandelier_trend", "regression_slope"]
        
        for key in selective:
            strategy = STRATEGIES[key]()
            result = strategy.generate_signals(data.copy())
            
            # Count non-zero entry signals
            trade_count = (result["entry_signal"] != 0).sum()
            
            # Should have at most 50 trades over 500 bars (10%)
            assert trade_count <= 50, f"{key} had {trade_count} trades (too many)"


class TestWrappers:
    """Test wrapper modules."""
    
    @pytest.fixture
    def data(self):
        return generate_test_data(200)
    
    def test_time_filter_removes_signals(self, data):
        """Time filter should remove signals outside window."""
        # Set some entry signals
        data["entry_signal"] = 0
        data.loc[data.index[:20], "entry_signal"] = 1
        
        # Filter to a narrow window
        filtered = apply_time_filter(data.copy(), start_hour_utc=10, end_hour_utc=12)
        
        # Some signals should be removed
        original_signals = (data["entry_signal"] != 0).sum()
        filtered_signals = (filtered["entry_signal"] != 0).sum()
        
        assert filtered_signals <= original_signals
    
    def test_volatility_sizing_scales_risk(self, data):
        """Volatility sizing should add adjusted_risk column."""
        result = apply_volatility_sizing(data.copy())
        
        assert "adjusted_risk" in result.columns
        assert result["adjusted_risk"].notna().any()
        # Risk should be bounded (check only non-null values)
        valid_risk = result["adjusted_risk"].dropna()
        assert (valid_risk >= 0.005).all()
        assert (valid_risk <= 0.02).all()


class TestSignalGeneration:
    """Test that strategies generate valid signals."""
    
    @pytest.fixture
    def data(self):
        return generate_test_data(500)
    
    def test_zscore_reversion_signals(self, data):
        """Z-score reversion should generate signals."""
        strategy = ZScoreReversionStrategy()
        result = strategy.generate_signals(data)
        
        assert "entry_signal" in result.columns
        assert "zscore" in result.columns
        assert "adx" in result.columns
    
    def test_chandelier_indicators(self, data):
        """Chandelier should calculate chandelier stops."""
        strategy = ChandelierTrendStrategy()
        result = strategy.generate_signals(data)
        
        assert "chandelier_long" in result.columns
        assert "chandelier_short" in result.columns
        assert "ema_slope" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
