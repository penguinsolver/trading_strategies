"""
Position and trade management for backtesting.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """
    Represents a completed trade with all relevant information.
    """
    # Timing
    entry_time: datetime
    exit_time: datetime
    
    # Position details
    side: str  # "long" or "short"
    entry_price: float
    stop_price: float
    exit_price: float
    size: float  # Position size in contracts/units
    
    # P&L
    pnl_gross: float  # Before costs
    pnl_net: float    # After costs
    r_multiple: float  # Risk-adjusted return
    
    # Costs breakdown
    fees: float
    funding: float
    slippage: float
    
    # Metadata
    exit_reason: str  # "stop", "target", "trailing", "signal", "end_of_data"
    strategy: str = ""
    notes: str = ""
    
    @property
    def entry_notional(self) -> float:
        """Entry notional value."""
        return self.entry_price * self.size
    
    @property
    def exit_notional(self) -> float:
        """Exit notional value."""
        return self.exit_price * self.size
    
    @property
    def duration_hours(self) -> float:
        """Trade duration in hours."""
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 3600
    
    @property
    def is_winner(self) -> bool:
        """Whether the trade was profitable (after costs)."""
        return self.pnl_net > 0
    
    @property
    def risk_amount(self) -> float:
        """Initial risk amount based on stop distance."""
        if self.side == "long":
            return (self.entry_price - self.stop_price) * self.size
        else:
            return (self.stop_price - self.entry_price) * self.size
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame/export."""
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl_gross": self.pnl_gross,
            "pnl_net": self.pnl_net,
            "r_multiple": self.r_multiple,
            "fees": self.fees,
            "funding": self.funding,
            "slippage": self.slippage,
            "exit_reason": self.exit_reason,
            "duration_hours": self.duration_hours,
            "strategy": self.strategy,
        }


@dataclass
class Position:
    """
    Represents an open position being tracked during backtest.
    """
    side: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_price: float
    
    # Optional targets
    target_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    
    # Tracking
    highest_price: float = field(default=0.0)  # For long trailing stop
    lowest_price: float = field(default=float("inf"))  # For short trailing stop
    partial_exits: list = field(default_factory=list)
    remaining_size: float = field(default=0.0)
    
    def __post_init__(self):
        self.remaining_size = self.size
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
    
    @property
    def initial_risk(self) -> float:
        """Initial risk per unit."""
        if self.side == "long":
            return self.entry_price - self.stop_price
        else:
            return self.stop_price - self.entry_price
    
    @property
    def initial_risk_amount(self) -> float:
        """Total initial risk in dollars."""
        return self.initial_risk * self.size
    
    def update_trailing_stop(self, high: float, low: float, close: float) -> None:
        """Update trailing stop based on price movement."""
        if self.trailing_stop_distance is None:
            return
        
        if self.side == "long":
            self.highest_price = max(self.highest_price, high)
            new_stop = self.highest_price - self.trailing_stop_distance
            self.stop_price = max(self.stop_price, new_stop)
        else:
            self.lowest_price = min(self.lowest_price, low)
            new_stop = self.lowest_price + self.trailing_stop_distance
            self.stop_price = min(self.stop_price, new_stop)
    
    def check_stop_hit(self, high: float, low: float) -> bool:
        """Check if stop was hit during the bar."""
        if self.side == "long":
            return low <= self.stop_price
        else:
            return high >= self.stop_price
    
    def check_target_hit(self, high: float, low: float) -> bool:
        """Check if target was hit during the bar."""
        if self.target_price is None:
            return False
        
        if self.side == "long":
            return high >= self.target_price
        else:
            return low <= self.target_price
    
    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate gross P&L for remaining position."""
        if self.side == "long":
            return (exit_price - self.entry_price) * self.remaining_size
        else:
            return (self.entry_price - exit_price) * self.remaining_size
    
    def calculate_r_multiple(self, exit_price: float) -> float:
        """Calculate R-multiple for the trade."""
        pnl_per_unit = exit_price - self.entry_price if self.side == "long" else self.entry_price - exit_price
        if self.initial_risk > 0:
            return pnl_per_unit / self.initial_risk
        return 0.0


@dataclass
class EquityPoint:
    """A point in the equity curve."""
    timestamp: datetime
    equity: float
    drawdown: float
    drawdown_pct: float
