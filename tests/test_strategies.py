"""Tests for trading strategies."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import STRATEGIES
from strategies.trend_pullback import TrendPullbackStrategy
from strategies.breakout import BreakoutStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.ma_crossover import MACrossoverStrategy


def create_sample_data(n_bars: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.randn(n_bars) * 0.002  # 0.2% volatility
    prices = 50000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "open": prices,
        "high": prices * (1 + np.random.rand(n_bars) * 0.005),
        "low": prices * (1 - np.random.rand(n_bars) * 0.005),
        "close": prices * (1 + np.random.randn(n_bars) * 0.002),
        "volume": np.random.rand(n_bars) * 1000 + 100,
    })
    
    # Ensure OHLC consistency
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    
    # Add datetime index
    data.index = pd.date_range(
        start="2024-01-01",
        periods=n_bars,
        freq="5min",
    )
    
    return data


class TestStrategyRegistry:
    """Tests for strategy registry."""
    
    def test_all_strategies_registered(self):
        """Verify all strategies are in the registry."""
        assert "trend_pullback" in STRATEGIES
        assert "breakout" in STRATEGIES
        assert "vwap_reversion" in STRATEGIES
        assert "ma_crossover" in STRATEGIES
    
    def test_strategy_instantiation(self):
        """Test strategies can be instantiated."""
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            assert strategy.name is not None
            assert len(strategy.get_param_config()) > 0


class TestTrendPullbackStrategy:
    """Tests for Trend Pullback strategy."""
    
    def test_signal_generation(self):
        """Test signal generation doesn't crash."""
        data = create_sample_data()
        strategy = TrendPullbackStrategy()
        
        result = strategy.generate_signals(data)
        
        # Should have signal columns
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns
        assert "stop_price" in result.columns
    
    def test_indicator_columns(self):
        """Test indicator columns are added."""
        data = create_sample_data()
        strategy = TrendPullbackStrategy()
        
        result = strategy.generate_signals(data)
        
        assert "htf_ema" in result.columns
        assert "slow_ema" in result.columns
        assert "fast_ema" in result.columns
        assert "atr" in result.columns
    
    def test_custom_parameters(self):
        """Test strategy accepts custom parameters."""
        strategy = TrendPullbackStrategy(
            htf_ema_period=100,
            ltf_slow_ema=30,
        )
        
        assert strategy.params["htf_ema_period"] == 100
        assert strategy.params["ltf_slow_ema"] == 30


class TestBreakoutStrategy:
    """Tests for Breakout strategy."""
    
    def test_signal_generation(self):
        """Test signal generation."""
        data = create_sample_data()
        strategy = BreakoutStrategy()
        
        result = strategy.generate_signals(data)
        
        assert "entry_signal" in result.columns
        assert "dc_upper" in result.columns
        assert "dc_lower" in result.columns
    
    def test_volatility_filter(self):
        """Test volatility filter columns are present."""
        data = create_sample_data()
        strategy = BreakoutStrategy()
        
        result = strategy.generate_signals(data)
        
        assert "atr_fast" in result.columns
        assert "atr_slow" in result.columns
        assert "volatility_ok" in result.columns


class TestVWAPReversionStrategy:
    """Tests for VWAP Reversion strategy."""
    
    def test_signal_generation(self):
        """Test signal generation."""
        data = create_sample_data()
        strategy = VWAPReversionStrategy()
        
        result = strategy.generate_signals(data)
        
        assert "entry_signal" in result.columns
        assert "vwap" in result.columns
        assert "rsi" in result.columns
    
    def test_ranging_filter(self):
        """Test ranging condition filter."""
        data = create_sample_data()
        strategy = VWAPReversionStrategy()
        
        result = strategy.generate_signals(data)
        
        assert "adx" in result.columns
        assert "is_ranging" in result.columns


class TestMACrossoverStrategy:
    """Tests for MA Crossover baseline strategy."""
    
    def test_signal_generation(self):
        """Test signal generation."""
        data = create_sample_data()
        strategy = MACrossoverStrategy()
        
        result = strategy.generate_signals(data)
        
        assert "entry_signal" in result.columns
        assert "ema_fast" in result.columns
        assert "ema_slow" in result.columns
    
    def test_crossover_signals(self):
        """Test that crossover signals are generated."""
        data = create_sample_data(500)  # More data for crossovers
        strategy = MACrossoverStrategy(fast_period=5, slow_period=20)
        
        result = strategy.generate_signals(data)
        
        # Should have some signals (long or short)
        has_signals = (result["entry_signal"] != 0).any()
        assert has_signals, "Strategy should generate at least one signal"
    
    def test_exit_signals(self):
        """Test exit signals are generated on opposite crossover."""
        data = create_sample_data(500)
        strategy = MACrossoverStrategy()
        
        result = strategy.generate_signals(data)
        
        # Check that exit signals can occur
        assert "exit_signal" in result.columns


class TestStrategyInterface:
    """Tests for strategy interface compliance."""
    
    @pytest.mark.parametrize("strategy_key", list(STRATEGIES.keys()))
    def test_param_config(self, strategy_key):
        """Test all strategies return valid param configs."""
        strategy_class = STRATEGIES[strategy_key]
        strategy = strategy_class()
        
        params = strategy.get_param_config()
        
        assert isinstance(params, list)
        for param in params:
            assert hasattr(param, "name")
            assert hasattr(param, "label")
            assert hasattr(param, "param_type")
            assert hasattr(param, "default")
    
    @pytest.mark.parametrize("strategy_key", list(STRATEGIES.keys()))
    def test_indicator_info(self, strategy_key):
        """Test all strategies return indicator info."""
        strategy_class = STRATEGIES[strategy_key]
        strategy = strategy_class()
        
        info = strategy.get_indicator_info()
        
        assert isinstance(info, list)
        for item in info:
            assert "name" in item
            assert "column" in item
            assert "color" in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
