"""Tests for backtest engine and cost model."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.costs import CostModel, OrderType
from backtest.position import Trade, Position
from backtest.metrics import calculate_metrics
from backtest.engine import BacktestEngine


class TestCostModel:
    """Tests for the cost model."""
    
    def test_fee_calculation(self):
        """Test fee calculations."""
        model = CostModel(
            maker_fee=0.0001,
            taker_fee=0.00035,
        )
        
        notional = 10000
        
        maker_fee = model.calculate_fee(notional, OrderType.LIMIT)
        assert maker_fee == pytest.approx(1.0)  # 0.01%
        
        taker_fee = model.calculate_fee(notional, OrderType.MARKET)
        assert taker_fee == pytest.approx(3.5)  # 0.035%
    
    def test_slippage_calculation(self):
        """Test slippage price impact."""
        model = CostModel(slippage_bps=1.0)
        
        price = 50000
        slippage = model.calculate_slippage(price)
        
        # 1 bp = 0.0001, so slippage = 50000 * 0.0001 = 5
        assert slippage == pytest.approx(5.0)
    
    def test_entry_slippage_direction(self):
        """Test slippage applies in the right direction for entry."""
        model = CostModel(slippage_bps=10.0)  # 10 bps for visibility
        
        price = 50000
        
        # Long entry: price should increase (worse fill)
        long_price = model.apply_entry_slippage(price, "long")
        assert long_price > price
        
        # Short entry: price should decrease (worse fill)
        short_price = model.apply_entry_slippage(price, "short")
        assert short_price < price
    
    def test_exit_slippage_direction(self):
        """Test slippage applies in the right direction for exit."""
        model = CostModel(slippage_bps=10.0)
        
        price = 50000
        
        # Long exit: price should decrease (worse fill)
        long_price = model.apply_exit_slippage(price, "long")
        assert long_price < price
        
        # Short exit: price should increase (worse fill)
        short_price = model.apply_exit_slippage(price, "short")
        assert short_price > price
    
    def test_funding_calculation(self):
        """Test funding cost/credit calculation."""
        model = CostModel(funding_rate_8h=0.0001)  # 0.01% per 8h
        
        notional = 10000
        hours = 8
        
        # Long pays funding
        long_funding = model.calculate_funding(notional, hours, "long")
        assert long_funding > 0
        assert long_funding == pytest.approx(1.0)  # 10000 * 0.0001 = 1
        
        # Short receives funding
        short_funding = model.calculate_funding(notional, hours, "short")
        assert short_funding < 0
        assert short_funding == pytest.approx(-1.0)


class TestPosition:
    """Tests for position management."""
    
    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions."""
        pos = Position(
            side="long",
            entry_time=datetime.now(),
            entry_price=50000,
            size=0.1,
            stop_price=49000,
        )
        
        # Price goes up
        pnl = pos.calculate_pnl(51000)
        assert pnl == pytest.approx(100)  # (51000-50000) * 0.1
        
        # Price goes down
        pnl_loss = pos.calculate_pnl(49500)
        assert pnl_loss == pytest.approx(-50)  # (49500-50000) * 0.1
    
    def test_r_multiple_calculation(self):
        """Test R-multiple calculation."""
        pos = Position(
            side="long",
            entry_time=datetime.now(),
            entry_price=50000,
            size=0.1,
            stop_price=49000,  # Risk = 1000 per unit
        )
        
        # 2R win (price goes to 52000, +2000 gain vs 1000 risk)
        r_mult = pos.calculate_r_multiple(52000)
        assert r_mult == pytest.approx(2.0)
        
        # 0.5R loss
        r_mult_loss = pos.calculate_r_multiple(49500)
        assert r_mult_loss == pytest.approx(-0.5)
    
    def test_stop_hit_detection(self):
        """Test stop-loss hit detection."""
        pos = Position(
            side="long",
            entry_time=datetime.now(),
            entry_price=50000,
            size=0.1,
            stop_price=49000,
        )
        
        # Bar that hits stop
        assert pos.check_stop_hit(high=50500, low=48500) == True
        
        # Bar that doesn't hit stop
        assert pos.check_stop_hit(high=51000, low=49500) == False
    
    def test_trailing_stop_update(self):
        """Test trailing stop updates correctly."""
        pos = Position(
            side="long",
            entry_time=datetime.now(),
            entry_price=50000,
            size=0.1,
            stop_price=49000,
            trailing_stop_distance=500,
        )
        
        # Price moves up
        pos.update_trailing_stop(high=51000, low=50500, close=50800)
        
        # Stop should have moved up
        assert pos.stop_price == pytest.approx(50500)  # 51000 - 500
        
        # Price moves higher
        pos.update_trailing_stop(high=52000, low=51500, close=51800)
        assert pos.stop_price == pytest.approx(51500)  # 52000 - 500


class TestMetrics:
    """Tests for performance metrics calculation."""
    
    def test_metrics_with_no_trades(self):
        """Test metrics calculation with empty trade list."""
        equity = pd.Series([10000, 10000, 10000], index=pd.date_range("2024-01-01", periods=3))
        metrics = calculate_metrics([], 10000, equity)
        
        assert metrics.total_trades == 0
        assert metrics.net_return == 0.0
        assert metrics.win_rate == 0.0
    
    def test_metrics_win_rate(self):
        """Test win rate calculation."""
        trades = [
            Trade(
                entry_time=datetime.now(), exit_time=datetime.now(),
                side="long", entry_price=50000, stop_price=49000, exit_price=51000,
                size=0.1, pnl_gross=100, pnl_net=90, r_multiple=1.0,
                fees=5, funding=3, slippage=2, exit_reason="target"
            ),
            Trade(
                entry_time=datetime.now(), exit_time=datetime.now(),
                side="long", entry_price=50000, stop_price=49000, exit_price=49500,
                size=0.1, pnl_gross=-50, pnl_net=-55, r_multiple=-0.5,
                fees=5, funding=0, slippage=0, exit_reason="stop"
            ),
        ]
        
        equity = pd.Series([10000, 10090, 10035], index=pd.date_range("2024-01-01", periods=3))
        metrics = calculate_metrics(trades, 10000, equity)
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(50.0)
    
    def test_metrics_drawdown(self):
        """Test max drawdown calculation."""
        # Create a trade that causes a drawdown
        trades = [
            Trade(
                entry_time=datetime(2024, 1, 1), exit_time=datetime(2024, 1, 3),
                side="long", entry_price=50000, stop_price=49000, exit_price=48000,
                size=0.1, pnl_gross=-200, pnl_net=-210, r_multiple=-2.0,
                fees=5, funding=3, slippage=2, exit_reason="stop"
            ),
        ]
        # Equity: goes up then down (simulating a loss)
        equity = pd.Series(
            [10000, 11000, 12000, 10500, 10800],
            index=pd.date_range("2024-01-01", periods=5)
        )
        
        metrics = calculate_metrics(trades, 10000, equity)
        
        # Max DD from 12000 to 10500 = -12.5%
        # Note: drawdown is calculated from equity curve, should be negative
        assert metrics.max_drawdown == pytest.approx(-12.5, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
