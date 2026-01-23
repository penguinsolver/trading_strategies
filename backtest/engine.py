"""
Core backtest engine with deterministic execution model.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import pandas as pd

from .costs import CostModel, OrderType
from .position import Trade, Position, EquityPoint
from .metrics import calculate_metrics, PerformanceMetrics

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: list[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    metrics: PerformanceMetrics
    data: pd.DataFrame  # Price data with signals
    strategy_name: str
    initial_capital: float
    final_equity: float
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def export_trades_csv(self, path: str) -> None:
        """Export trades to CSV."""
        self.get_trades_df().to_csv(path, index=False)
    
    def export_trades_json(self, path: str) -> None:
        """Export trades to JSON."""
        self.get_trades_df().to_json(path, orient="records", date_format="iso")


class BacktestEngine:
    """
    Bar-based backtest engine with explicit execution timing.
    
    Execution Model:
    - Signals are generated on bar[i] close
    - Entries execute at bar[i+1] open price
    - Stop-losses are checked against bar high/low, exit at stop price
    - Targets are checked similarly
    - Market exits execute at bar close
    
    This engine supports:
    - Long and short positions
    - Stop-loss orders
    - Take-profit targets
    - Trailing stops
    - Partial exits
    - Realistic cost modeling
    """
    
    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        entry_on_next_open: bool = True,
    ):
        """
        Initialize the backtest engine.
        
        Args:
            cost_model: Trading cost model (default creates new one)
            entry_on_next_open: If True, entries happen at next bar open
                               If False, entries happen at signal bar close
        """
        self.cost_model = cost_model or CostModel()
        self.entry_on_next_open = entry_on_next_open
    
    def run(
        self,
        strategy: "Strategy",
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
    ) -> BacktestResult:
        """
        Run backtest on the provided data.
        
        Args:
            strategy: Strategy instance with generate_signals method
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital in USD
            risk_per_trade: Fraction of capital to risk per trade (e.g., 0.01 = 1%)
            
        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        # Generate signals
        data = strategy.generate_signals(data.copy())
        
        # Initialize tracking
        equity = initial_capital
        trades: list[Trade] = []
        equity_points: list[EquityPoint] = []
        position: Optional[Position] = None
        peak_equity = initial_capital
        
        # Pending entry signal
        pending_entry = None
        
        # Iterate through bars
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = data.index[i]
            
            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_close = row["close"]
            
            # Check for pending entry at bar open
            if pending_entry is not None and self.entry_on_next_open:
                position = self._execute_entry(
                    pending_entry,
                    bar_open,
                    timestamp,
                    equity,
                    risk_per_trade,
                )
                pending_entry = None
            
            # If we have a position, check for exits
            if position is not None:
                exit_price = None
                exit_reason = None
                
                # Update trailing stop
                position.update_trailing_stop(bar_high, bar_low, bar_close)
                
                # Check stop-loss (priority 1)
                if position.check_stop_hit(bar_high, bar_low):
                    exit_price = position.stop_price
                    exit_reason = "stop"
                
                # Check target (priority 2)
                elif position.check_target_hit(bar_high, bar_low):
                    exit_price = position.target_price
                    exit_reason = "target"
                
                # Check exit signal (priority 3)
                elif "exit_signal" in row and row.get("exit_signal", False):
                    exit_price = bar_close
                    exit_reason = "signal"
                
                # Execute exit if triggered
                if exit_price is not None:
                    trade = self._execute_exit(
                        position,
                        exit_price,
                        timestamp,
                        exit_reason,
                        strategy.name,
                    )
                    trades.append(trade)
                    equity += trade.pnl_net
                    position = None
            
            # Check for new entry signal (only if no position)
            if position is None and "entry_signal" in data.columns:
                signal = row.get("entry_signal", 0)
                if signal != 0:
                    stop_price = row.get("stop_price", None)
                    target_price = row.get("target_price", None)
                    trailing_atr = row.get("trailing_stop_atr", None)
                    
                    entry_info = {
                        "side": "long" if signal > 0 else "short",
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "trailing_atr": trailing_atr,
                    }
                    
                    if self.entry_on_next_open:
                        pending_entry = entry_info
                    else:
                        # Enter at close
                        position = self._execute_entry(
                            entry_info,
                            bar_close,
                            timestamp,
                            equity,
                            risk_per_trade,
                        )
            
            # Record equity point
            unrealized_pnl = 0.0
            if position is not None:
                unrealized_pnl = position.calculate_pnl(bar_close)
            
            current_equity = equity + unrealized_pnl
            peak_equity = max(peak_equity, current_equity)
            drawdown = current_equity - peak_equity
            drawdown_pct = (drawdown / peak_equity * 100) if peak_equity > 0 else 0.0
            
            equity_points.append(EquityPoint(
                timestamp=timestamp,
                equity=current_equity,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct,
            ))
        
        # Close any remaining position at end
        if position is not None:
            last_row = data.iloc[-1]
            trade = self._execute_exit(
                position,
                last_row["close"],
                data.index[-1],
                "end_of_data",
                strategy.name,
            )
            trades.append(trade)
            equity += trade.pnl_net
        
        # Build equity curve
        equity_curve = pd.Series(
            [p.equity for p in equity_points],
            index=[p.timestamp for p in equity_points],
            name="equity",
        )
        
        drawdown_curve = pd.Series(
            [p.drawdown_pct for p in equity_points],
            index=[p.timestamp for p in equity_points],
            name="drawdown",
        )
        
        # Calculate metrics
        metrics = calculate_metrics(trades, initial_capital, equity_curve)
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            metrics=metrics,
            data=data,
            strategy_name=strategy.name,
            initial_capital=initial_capital,
            final_equity=equity,
        )
    
    def _execute_entry(
        self,
        entry_info: dict,
        price: float,
        timestamp: datetime,
        equity: float,
        risk_per_trade: float,
    ) -> Position:
        """
        Execute an entry and return the new position.
        
        NEW MODEL (Capital Allocation + Max Leverage):
        - Allocate `risk_per_trade` fraction of capital to each trade
        - Apply up to `max_leverage` on that allocation
        - Position size = (allocated_capital * leverage) / entry_price
        """
        side = entry_info["side"]
        stop_price = entry_info["stop_price"]
        
        # Apply slippage to entry
        entry_price = self.cost_model.apply_entry_slippage(price, side)
        
        # NEW: Capital allocation model with max leverage
        max_leverage = 10.0  # Hard constraint: max 10x leverage
        
        # Capital allocated to this trade (e.g., 10% of equity)
        allocated_capital = equity * risk_per_trade
        
        # Calculate position size based on leverage
        # Notional = allocated_capital * leverage
        # Position size = Notional / price
        notional = allocated_capital * max_leverage
        size = notional / entry_price
        
        # Calculate trailing stop distance
        trailing_distance = None
        if entry_info.get("trailing_atr") is not None:
            trailing_distance = entry_info["trailing_atr"]
        
        return Position(
            side=side,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            stop_price=stop_price if stop_price else (entry_price * 0.95 if side == "long" else entry_price * 1.05),
            target_price=entry_info.get("target_price"),
            trailing_stop_distance=trailing_distance,
        )
    
    def _execute_exit(
        self,
        position: Position,
        price: float,
        timestamp: datetime,
        reason: str,
        strategy_name: str,
    ) -> Trade:
        """Execute an exit and return the completed trade."""
        # Apply slippage
        exit_price = self.cost_model.apply_exit_slippage(price, position.side)
        
        # Get actual exit price for stop/target
        if reason == "stop":
            exit_price = self.cost_model.apply_exit_slippage(position.stop_price, position.side)
        elif reason == "target" and position.target_price:
            exit_price = self.cost_model.apply_exit_slippage(position.target_price, position.side)
        
        # Calculate P&L
        pnl_gross = position.calculate_pnl(exit_price)
        
        # Calculate costs
        entry_notional = position.entry_price * position.remaining_size
        exit_notional = exit_price * position.remaining_size
        hours_held = (timestamp - position.entry_time).total_seconds() / 3600
        
        costs = self.cost_model.total_trade_cost(
            entry_notional=entry_notional,
            exit_notional=exit_notional,
            hours_held=hours_held,
            side=position.side,
        )
        
        pnl_net = pnl_gross - costs["total_cost"]
        r_multiple = position.calculate_r_multiple(exit_price)
        
        return Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            side=position.side,
            entry_price=position.entry_price,
            stop_price=position.stop_price,
            exit_price=exit_price,
            size=position.remaining_size,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            r_multiple=r_multiple,
            fees=costs["total_fees"],
            funding=costs["funding"],
            slippage=costs["slippage"],
            exit_reason=reason,
            strategy=strategy_name,
        )
