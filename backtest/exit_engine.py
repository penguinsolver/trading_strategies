"""
Exit Engine - Standardized exit logic with hard stops and trailing stops.

This module provides a wrapper backtest engine that uses the same entry signals
from strategies but applies a standardized exit engine with configurable
hard stops, trailing stops, partial take-profits, and breakeven rules.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING, List
import pandas as pd
import numpy as np

from .costs import CostModel, OrderType
from .position import Trade, Position, EquityPoint
from .metrics import calculate_metrics, PerformanceMetrics

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class ExitEngineConfig:
    """Configuration for the standardized exit engine."""
    
    # ATR settings
    atr_period: int = 14
    
    # Hard stop settings
    hard_stop_mult: float = 1.5  # k_stop: hard stop = entry ± k_stop * ATR
    
    # Trailing stop settings
    trailing_type: str = "chandelier_atr"  # "chandelier_atr", "atr", "percent", "fixed"
    trailing_mult: float = 3.0  # k_trail: trail = extreme ± k_trail * ATR
    trail_activation_r: float = 1.0  # Activate trailing after +XR profit
    
    # Partial take-profit
    partial_tp_enabled: bool = True  # Take 50% at +1R
    partial_tp_r: float = 1.0  # R-multiple for partial TP
    partial_tp_percent: float = 0.5  # Portion to close (50%)
    
    # Breakeven rule
    breakeven_enabled: bool = True  # Move stop to breakeven at +1R
    breakeven_r: float = 1.0  # R-multiple for breakeven move
    
    # Time stop
    time_stop_bars: int = 0  # Exit after X bars if unrealized < threshold (0 = disabled)
    time_stop_threshold: float = 0.0  # Threshold as % of entry
    
    # Cooldown
    cooldown_bars: int = 4  # Bars to wait after a stop-out before new entry


@dataclass
class EnhancedPosition:
    """Enhanced position with exit engine tracking."""
    
    side: str
    entry_time: datetime
    entry_price: float
    size: float
    initial_size: float
    
    # Hard stop (fixed at entry)
    hard_stop: float
    
    # Trailing stop
    trail_stop: float
    trailing_active: bool = False
    
    # Risk tracking
    atr_at_entry: float = 0.0
    initial_risk: float = 0.0  # Entry - hard stop
    
    # Tracking
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    bars_held: int = 0
    
    # Partial exits
    partial_taken: bool = False
    breakeven_moved: bool = False
    
    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.initial_risk = abs(self.entry_price - self.hard_stop)
    
    @property
    def current_stop(self) -> float:
        """Get current active stop price."""
        if self.trailing_active:
            if self.side == "long":
                return max(self.hard_stop, self.trail_stop)
            else:
                return min(self.hard_stop, self.trail_stop)
        return self.hard_stop
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def calculate_r_multiple(self, current_price: float) -> float:
        """Calculate current R-multiple."""
        if self.initial_risk <= 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) / self.initial_risk
        else:
            return (self.entry_price - current_price) / self.initial_risk
    
    def check_stop_hit(self, high: float, low: float) -> bool:
        """Check if any stop was hit."""
        stop = self.current_stop
        if self.side == "long":
            return low <= stop
        else:
            return high >= stop


@dataclass
class ExitEngineResult:
    """Complete backtest results from exit engine."""
    
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    metrics: PerformanceMetrics
    data: pd.DataFrame
    strategy_name: str
    initial_capital: float
    final_equity: float
    config: ExitEngineConfig
    
    # Additional metrics
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    avg_r_multiple: float = 0.0
    avg_hold_bars: float = 0.0
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.trades:
            record = t.to_dict()
            # Add exit engine specific fields
            record["bars_held"] = getattr(t, "bars_held", 0)
            records.append(record)
        return pd.DataFrame(records)
    
    def export_trades_csv(self, path: str) -> None:
        """Export trades to CSV."""
        self.get_trades_df().to_csv(path, index=False)
    
    def get_summary_dict(self) -> dict:
        """Get summary metrics as dictionary."""
        return {
            "strategy": self.strategy_name,
            "net_return_pct": self.metrics.total_return,
            "max_drawdown_pct": self.metrics.max_drawdown,
            "profit_factor": self.metrics.profit_factor,
            "expectancy_r": self.avg_r_multiple,
            "win_rate_pct": self.metrics.win_rate,
            "total_trades": self.metrics.total_trades,
            "avg_hold_bars": self.avg_hold_bars,
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "cost_per_trade": (self.total_fees + self.total_slippage) / max(1, self.metrics.total_trades),
        }


class ExitEngineBacktester:
    """
    Backtest engine with standardized exit logic.
    
    Uses entry signals from strategies but applies consistent exit rules:
    - Hard stop based on ATR
    - Chandelier trailing stop
    - Optional partial take-profit
    - Optional breakeven move
    - Time-based exit
    - Post-stopout cooldown
    
    Execution Model:
    - Signals generated on bar[i] close
    - Entries execute at bar[i+1] open
    - Hard stop active immediately after entry
    - Trailing stop updates at bar close, applies from next bar
    """
    
    def __init__(
        self,
        config: Optional[ExitEngineConfig] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.config = config or ExitEngineConfig()
        self.cost_model = cost_model or CostModel()
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR without lookahead bias."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=self.config.atr_period).mean()
    
    def run(
        self,
        strategy: "Strategy",
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
    ) -> ExitEngineResult:
        """
        Run backtest with exit engine.
        
        Args:
            strategy: Strategy instance with generate_signals method
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital
            risk_per_trade: Fraction of capital to risk
            
        Returns:
            ExitEngineResult with trades, metrics, and summary
        """
        # Generate signals from strategy
        data = strategy.generate_signals(data.copy())
        
        # Calculate ATR
        data["_atr"] = self._calculate_atr(data)
        
        # Initialize tracking
        equity = initial_capital
        trades: List[Trade] = []
        equity_points: List[EquityPoint] = []
        position: Optional[EnhancedPosition] = None
        peak_equity = initial_capital
        
        # Pending entry
        pending_entry = None
        
        # Cooldown tracking
        cooldown_remaining = 0
        
        # Metrics accumulators
        total_gross = 0.0
        total_fees = 0.0
        total_slippage = 0.0
        total_r = 0.0
        total_bars = 0
        
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = data.index[i]
            
            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_close = row["close"]
            current_atr = row.get("_atr", 0)
            
            # Skip if ATR not available yet
            if pd.isna(current_atr) or current_atr <= 0:
                equity_points.append(EquityPoint(
                    timestamp=timestamp,
                    equity=equity,
                    drawdown=0.0,
                    drawdown_pct=0.0,
                ))
                continue
            
            # Decrease cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            
            # Execute pending entry at bar open
            if pending_entry is not None:
                position = self._execute_entry(
                    pending_entry,
                    bar_open,
                    timestamp,
                    equity,
                    risk_per_trade,
                    current_atr,
                )
                pending_entry = None
            
            # Process existing position
            if position is not None:
                position.bars_held += 1
                exit_price = None
                exit_reason = None
                
                # Update price extremes
                position.highest_price = max(position.highest_price, bar_high)
                position.lowest_price = min(position.lowest_price, bar_low)
                
                # Calculate current R-multiple
                current_r = position.calculate_r_multiple(bar_close)
                
                # Check for breakeven move (on bar close, applies next bar)
                if (self.config.breakeven_enabled and 
                    not position.breakeven_moved and
                    current_r >= self.config.breakeven_r):
                    position.hard_stop = position.entry_price
                    position.breakeven_moved = True
                
                # Check for trailing activation
                if (not position.trailing_active and 
                    current_r >= self.config.trail_activation_r):
                    position.trailing_active = True
                
                # Update trailing stop (on bar close, applies next bar)
                if position.trailing_active:
                    self._update_trailing_stop(position, bar_high, bar_low, current_atr)
                
                # Check time stop
                if (self.config.time_stop_bars > 0 and 
                    position.bars_held >= self.config.time_stop_bars):
                    unrealized_pct = (bar_close - position.entry_price) / position.entry_price * 100
                    if position.side == "short":
                        unrealized_pct = -unrealized_pct
                    if unrealized_pct < self.config.time_stop_threshold:
                        exit_price = bar_close
                        exit_reason = "time_stop"
                
                # Check hard/trailing stop
                if exit_reason is None and position.check_stop_hit(bar_high, bar_low):
                    exit_price = position.current_stop
                    if position.trailing_active and position.trail_stop == position.current_stop:
                        exit_reason = "trailing_stop"
                    else:
                        exit_reason = "hard_stop"
                
                # Check partial take-profit
                if (exit_reason is None and 
                    self.config.partial_tp_enabled and
                    not position.partial_taken and
                    current_r >= self.config.partial_tp_r):
                    # Take partial profit
                    partial_size = position.initial_size * self.config.partial_tp_percent
                    trade = self._execute_partial_exit(
                        position,
                        bar_close,
                        timestamp,
                        partial_size,
                        strategy.name,
                    )
                    trades.append(trade)
                    equity += trade.pnl_net
                    total_gross += trade.pnl_gross
                    total_fees += trade.fees
                    total_slippage += trade.slippage
                    total_r += trade.r_multiple
                    position.partial_taken = True
                
                # Execute full exit if triggered
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
                    total_gross += trade.pnl_gross
                    total_fees += trade.fees
                    total_slippage += trade.slippage
                    total_r += trade.r_multiple
                    total_bars += position.bars_held
                    
                    # Start cooldown if stopped out
                    if exit_reason in ["hard_stop", "trailing_stop"]:
                        cooldown_remaining = self.config.cooldown_bars
                    
                    position = None
            
            # Check for new entry signal (if no position and not in cooldown)
            if position is None and cooldown_remaining == 0:
                signal = row.get("entry_signal", 0)
                if signal != 0:
                    pending_entry = {
                        "side": "long" if signal > 0 else "short",
                        "atr": current_atr,
                    }
            
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
        
        # Close remaining position at end
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
            total_gross += trade.pnl_gross
            total_fees += trade.fees
            total_slippage += trade.slippage
            total_r += trade.r_multiple
            total_bars += position.bars_held
        
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
        
        # Calculate averages
        num_trades = len(trades)
        avg_r = total_r / num_trades if num_trades > 0 else 0.0
        avg_bars = total_bars / num_trades if num_trades > 0 else 0.0
        
        return ExitEngineResult(
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            metrics=metrics,
            data=data,
            strategy_name=strategy.name,
            initial_capital=initial_capital,
            final_equity=equity,
            config=self.config,
            gross_pnl=total_gross,
            total_fees=total_fees,
            total_slippage=total_slippage,
            avg_r_multiple=avg_r,
            avg_hold_bars=avg_bars,
        )
    
    def _execute_entry(
        self,
        entry_info: dict,
        price: float,
        timestamp: datetime,
        equity: float,
        risk_per_trade: float,
        atr: float,
    ) -> EnhancedPosition:
        """Execute entry with exit engine rules."""
        side = entry_info["side"]
        
        # Apply slippage
        entry_price = self.cost_model.apply_entry_slippage(price, side)
        
        # Calculate hard stop using ATR
        if side == "long":
            hard_stop = entry_price - (atr * self.config.hard_stop_mult)
        else:
            hard_stop = entry_price + (atr * self.config.hard_stop_mult)
        
        # Calculate position size based on risk
        risk_per_unit = abs(entry_price - hard_stop)
        if risk_per_unit > 0:
            risk_amount = equity * risk_per_trade
            size = risk_amount / risk_per_unit
        else:
            size = (equity * risk_per_trade) / entry_price
        
        # Initialize trailing stop at same level as hard stop
        trail_stop = hard_stop
        
        return EnhancedPosition(
            side=side,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            initial_size=size,
            hard_stop=hard_stop,
            trail_stop=trail_stop,
            atr_at_entry=atr,
        )
    
    def _update_trailing_stop(
        self,
        position: EnhancedPosition,
        high: float,
        low: float,
        atr: float,
    ) -> None:
        """Update trailing stop based on config."""
        if self.config.trailing_type == "chandelier_atr":
            if position.side == "long":
                new_stop = position.highest_price - (atr * self.config.trailing_mult)
                position.trail_stop = max(position.trail_stop, new_stop)
            else:
                new_stop = position.lowest_price + (atr * self.config.trailing_mult)
                position.trail_stop = min(position.trail_stop, new_stop)
        
        elif self.config.trailing_type == "atr":
            # Simple ATR trail from current price
            if position.side == "long":
                new_stop = high - (atr * self.config.trailing_mult)
                position.trail_stop = max(position.trail_stop, new_stop)
            else:
                new_stop = low + (atr * self.config.trailing_mult)
                position.trail_stop = min(position.trail_stop, new_stop)
        
        elif self.config.trailing_type == "percent":
            # Percentage trail
            pct = self.config.trailing_mult / 100  # Use mult as percentage
            if position.side == "long":
                new_stop = position.highest_price * (1 - pct)
                position.trail_stop = max(position.trail_stop, new_stop)
            else:
                new_stop = position.lowest_price * (1 + pct)
                position.trail_stop = min(position.trail_stop, new_stop)
    
    def _execute_partial_exit(
        self,
        position: EnhancedPosition,
        price: float,
        timestamp: datetime,
        size: float,
        strategy_name: str,
    ) -> Trade:
        """Execute partial take-profit."""
        exit_price = self.cost_model.apply_exit_slippage(price, position.side)
        
        # Calculate P&L for partial size
        if position.side == "long":
            pnl_gross = (exit_price - position.entry_price) * size
        else:
            pnl_gross = (position.entry_price - exit_price) * size
        
        # Calculate costs
        entry_notional = position.entry_price * size
        exit_notional = exit_price * size
        hours_held = (timestamp - position.entry_time).total_seconds() / 3600
        
        costs = self.cost_model.total_trade_cost(
            entry_notional=entry_notional,
            exit_notional=exit_notional,
            hours_held=hours_held,
            side=position.side,
        )
        
        pnl_net = pnl_gross - costs["total_cost"]
        
        # Calculate R-multiple for this partial
        if position.initial_risk > 0:
            r_mult = (exit_price - position.entry_price) / position.initial_risk
            if position.side == "short":
                r_mult = -r_mult
        else:
            r_mult = 0.0
        
        # Reduce position size
        position.size -= size
        
        return Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            side=position.side,
            entry_price=position.entry_price,
            stop_price=position.hard_stop,
            exit_price=exit_price,
            size=size,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            r_multiple=r_mult,
            fees=costs["total_fees"],
            funding=costs["funding"],
            slippage=costs["slippage"],
            exit_reason="partial_tp",
            strategy=strategy_name,
        )
    
    def _execute_exit(
        self,
        position: EnhancedPosition,
        price: float,
        timestamp: datetime,
        reason: str,
        strategy_name: str,
    ) -> Trade:
        """Execute full exit."""
        # Apply slippage
        if reason in ["hard_stop", "trailing_stop"]:
            exit_price = self.cost_model.apply_exit_slippage(position.current_stop, position.side)
        else:
            exit_price = self.cost_model.apply_exit_slippage(price, position.side)
        
        # Calculate P&L
        if position.side == "long":
            pnl_gross = (exit_price - position.entry_price) * position.size
        else:
            pnl_gross = (position.entry_price - exit_price) * position.size
        
        # Calculate costs
        entry_notional = position.entry_price * position.size
        exit_notional = exit_price * position.size
        hours_held = (timestamp - position.entry_time).total_seconds() / 3600
        
        costs = self.cost_model.total_trade_cost(
            entry_notional=entry_notional,
            exit_notional=exit_notional,
            hours_held=hours_held,
            side=position.side,
        )
        
        pnl_net = pnl_gross - costs["total_cost"]
        
        # Calculate R-multiple
        r_mult = position.calculate_r_multiple(exit_price)
        
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            side=position.side,
            entry_price=position.entry_price,
            stop_price=position.hard_stop,
            exit_price=exit_price,
            size=position.size,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            r_multiple=r_mult,
            fees=costs["total_fees"],
            funding=costs["funding"],
            slippage=costs["slippage"],
            exit_reason=reason,
            strategy=strategy_name,
        )
        
        # Store bars held in notes
        trade.notes = f"bars_held={position.bars_held}"
        
        return trade
