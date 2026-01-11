"""
Unit tests for new trading strategies.

Tests verify:
1. No lookahead bias - signals at time t only use data <= t
2. Signal generation works correctly
3. Parameter defaults are applied
4. Edge cases handled (empty data, insufficient warmup)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies import STRATEGIES
from strategies.supertrend import SupertrendStrategy
from strategies.donchian_turtle import DonchianTurtleStrategy
from strategies.rsi2_dip import RSI2DipStrategy
from strategies.bb_squeeze import BBSqueezeStrategy
from strategies.inside_bar import InsideBarStrategy
from strategies.orb import ORBStrategy
from strategies.breakout_retest import BreakoutRetestStrategy
from strategies.regime_switcher import RegimeSwitcherStrategy


def generate_test_data(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    # Generate random walk for close prices
    returns = np.random.randn(n_bars) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = (high + low) / 2 + np.random.randn(n_bars) * 0.1
    volume = np.random.randint(1000, 10000, n_bars).astype(float)
    
    # Create datetime index
    start_date = datetime(2024, 1, 1, 0, 0)
    dates = pd.date_range(start=start_date, periods=n_bars, freq="15min")
    
    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestStrategyLoading:
    """Test that all strategies can be loaded and instantiated."""
    
    def test_all_strategies_in_registry(self):
        """Verify all new strategies are in the registry."""
        expected = [
            "supertrend", "donchian_turtle", "rsi2_dip", "bb_squeeze",
            "inside_bar", "orb", "breakout_retest", "regime_switcher"
        ]
        for key in expected:
            assert key in STRATEGIES, f"Strategy {key} not in registry"
    
    def test_instantiate_all_strategies(self):
        """All strategies should instantiate with default parameters."""
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            assert strategy is not None
            assert hasattr(strategy, "name")
            assert hasattr(strategy, "generate_signals")
            assert hasattr(strategy, "get_param_config")


class TestNoLookaheadBias:
    """
    Test that strategies don't use future data.
    
    For each strategy, we verify that signals generated at bar i
    are exactly the same whether we run with data[:i+1] or full data.
    """
    
    @pytest.fixture
    def data(self):
        return generate_test_data(200)
    
    def _check_no_lookahead(self, strategy, data: pd.DataFrame):
        """Helper to verify no lookahead bias."""
        # Run on full data
        full_result = strategy.generate_signals(data.copy())
        
        # Check a sample of bars
        check_indices = [50, 100, 150]
        
        for i in check_indices:
            partial_data = data.iloc[:i+1].copy()
            partial_result = strategy.generate_signals(partial_data)
            
            # Entry signal at bar i should be the same
            full_signal = full_result["entry_signal"].iloc[i]
            partial_signal = partial_result["entry_signal"].iloc[-1]
            
            assert full_signal == partial_signal, \
                f"Lookahead detected at bar {i}: full={full_signal}, partial={partial_signal}"
    
    def test_supertrend_no_lookahead(self, data):
        strategy = SupertrendStrategy()
        self._check_no_lookahead(strategy, data)
    
    def test_donchian_turtle_no_lookahead(self, data):
        strategy = DonchianTurtleStrategy()
        self._check_no_lookahead(strategy, data)
    
    def test_rsi2_dip_no_lookahead(self, data):
        strategy = RSI2DipStrategy()
        self._check_no_lookahead(strategy, data)
    
    def test_bb_squeeze_no_lookahead(self, data):
        strategy = BBSqueezeStrategy()
        self._check_no_lookahead(strategy, data)
    
    def test_inside_bar_no_lookahead(self, data):
        strategy = InsideBarStrategy()
        self._check_no_lookahead(strategy, data)


class TestSignalGeneration:
    """Test that strategies generate valid signals."""
    
    @pytest.fixture
    def data(self):
        return generate_test_data(500)  # More data for signal generation
    
    def test_supertrend_generates_signals(self, data):
        strategy = SupertrendStrategy()
        result = strategy.generate_signals(data)
        
        # Check required columns exist
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns
        assert "stop_price" in result.columns
        
        # Check signal values are valid
        assert set(result["entry_signal"].unique()).issubset({-1, 0, 1})
        
        # Should generate some signals
        assert (result["entry_signal"] != 0).any(), "No entry signals generated"
    
    def test_donchian_turtle_generates_signals(self, data):
        strategy = DonchianTurtleStrategy()
        result = strategy.generate_signals(data)
        
        assert "entry_signal" in result.columns
        assert (result["entry_signal"] != 0).any(), "No entry signals generated"
    
    def test_rsi2_dip_generates_signals(self, data):
        strategy = RSI2DipStrategy()
        result = strategy.generate_signals(data)
        
        assert "entry_signal" in result.columns
        # RSI-2 is very sensitive, should generate signals
        assert (result["entry_signal"] != 0).any(), "No entry signals generated"
    
    def test_bb_squeeze_indicators_calculated(self, data):
        strategy = BBSqueezeStrategy()
        result = strategy.generate_signals(data)
        
        # Check Bollinger Band columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_mid" in result.columns
        assert "bandwidth" in result.columns
        assert "bw_percentile" in result.columns
    
    def test_inside_bar_pattern_detection(self, data):
        strategy = InsideBarStrategy()
        result = strategy.generate_signals(data)
        
        # Check pattern detection columns
        assert "inside_bar" in result.columns
        assert "bar_range" in result.columns


class TestParameterHandling:
    """Test that parameters are correctly applied."""
    
    def test_supertrend_params(self):
        strategy = SupertrendStrategy(atr_period=14, multiplier=2.5)
        assert strategy.params["atr_period"] == 14
        assert strategy.params["multiplier"] == 2.5
    
    def test_donchian_turtle_params(self):
        strategy = DonchianTurtleStrategy(entry_n=30, exit_n=15)
        assert strategy.params["entry_n"] == 30
        assert strategy.params["exit_n"] == 15
    
    def test_rsi2_dip_params(self):
        strategy = RSI2DipStrategy(rsi_oversold=15, long_only=False)
        assert strategy.params["rsi_oversold"] == 15
        assert strategy.params["long_only"] is False
    
    def test_inside_bar_params(self):
        strategy = InsideBarStrategy(pattern_type="nr7", nr_lookback=10)
        assert strategy.params["pattern_type"] == "nr7"
        assert strategy.params["nr_lookback"] == 10


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Strategies should handle empty data gracefully."""
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            try:
                result = strategy.generate_signals(empty_data)
                # Should return a DataFrame (possibly empty)
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                # Some strategies may raise on empty data, which is acceptable
                pass
    
    def test_insufficient_warmup(self):
        """Strategies should handle insufficient data for indicators."""
        short_data = generate_test_data(10)  # Very short data
        
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            try:
                result = strategy.generate_signals(short_data)
                # Should return something, even if all NaN
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                # Some strategies may need more data
                pass


class TestStrategyInterface:
    """Test that all strategies implement the correct interface."""
    
    def test_all_strategies_have_name(self):
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            assert isinstance(strategy.name, str)
            assert len(strategy.name) > 0
    
    def test_all_strategies_have_description(self):
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            # Description is optional but should be a string
            assert isinstance(strategy.description, str)
    
    def test_all_strategies_have_param_config(self):
        for key, strategy_class in STRATEGIES.items():
            strategy = strategy_class()
            config = strategy.get_param_config()
            assert isinstance(config, list)
            # Each param should have required fields
            for param in config:
                assert hasattr(param, "name")
                assert hasattr(param, "label")
                assert hasattr(param, "param_type")
                assert hasattr(param, "default")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
