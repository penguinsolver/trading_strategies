"""
Performance metrics calculation for backtest results.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from .position import Trade


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Returns
    net_return: float  # Percentage return
    net_return_dollars: float
    gross_return_dollars: float
    
    # Risk
    max_drawdown: float  # Maximum drawdown percentage
    max_drawdown_dollars: float
    
    # Win/Loss
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    profit_factor: float  # Gross profit / Gross loss
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk-adjusted
    avg_r_multiple: float
    expectancy: float  # Average profit per trade as % of account
    sharpe_ratio: Optional[float]
    
    # Activity
    trades_per_day: float
    avg_hold_time_hours: float
    
    # Costs
    total_fees: float
    total_funding: float
    total_slippage: float
    total_costs: float
    pnl_before_costs: float
    pnl_after_costs: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "Net Return (%)": f"{self.net_return:.2f}%",
            "Net Return ($)": f"${self.net_return_dollars:,.2f}",
            "Max Drawdown (%)": f"{self.max_drawdown:.2f}%",
            "Total Trades": self.total_trades,
            "Win Rate (%)": f"{self.win_rate:.1f}%",
            "Profit Factor": f"{self.profit_factor:.2f}" if self.profit_factor != float('inf') else "∞",
            "Avg R-Multiple": f"{self.avg_r_multiple:.2f}",
            "Expectancy (%)": f"{self.expectancy:.3f}%",
            "Trades/Day": f"{self.trades_per_day:.1f}",
            "Avg Hold Time": f"{self.avg_hold_time_hours:.1f}h",
            "Total Fees": f"${self.total_fees:,.2f}",
            "Total Funding": f"${self.total_funding:,.2f}",
            "Total Slippage": f"${self.total_slippage:,.2f}",
            "P&L Before Costs": f"${self.pnl_before_costs:,.2f}",
            "P&L After Costs": f"${self.pnl_after_costs:,.2f}",
        }


def calculate_metrics(
    trades: list[Trade],
    initial_capital: float,
    equity_curve: pd.Series,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from trade list.
    
    Args:
        trades: List of completed trades
        initial_capital: Starting capital
        equity_curve: Series of equity values indexed by timestamp
        
    Returns:
        PerformanceMetrics object
    """
    if not trades:
        return PerformanceMetrics(
            net_return=0.0,
            net_return_dollars=0.0,
            gross_return_dollars=0.0,
            max_drawdown=0.0,
            max_drawdown_dollars=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_r_multiple=0.0,
            expectancy=0.0,
            sharpe_ratio=None,
            trades_per_day=0.0,
            avg_hold_time_hours=0.0,
            total_fees=0.0,
            total_funding=0.0,
            total_slippage=0.0,
            total_costs=0.0,
            pnl_before_costs=0.0,
            pnl_after_costs=0.0,
        )
    
    # Convert trades to DataFrame for easier analysis
    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    
    # Basic counts
    total_trades = len(trades)
    winning_trades = len(trades_df[trades_df["pnl_net"] > 0])
    losing_trades = len(trades_df[trades_df["pnl_net"] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
    
    # P&L
    gross_pnl = trades_df["pnl_gross"].sum()
    net_pnl = trades_df["pnl_net"].sum()
    net_return = (net_pnl / initial_capital) * 100
    
    # Win/Loss analysis
    winners = trades_df[trades_df["pnl_net"] > 0]["pnl_net"]
    losers = trades_df[trades_df["pnl_net"] <= 0]["pnl_net"]
    
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = losers.mean() if len(losers) > 0 else 0.0
    largest_win = winners.max() if len(winners) > 0 else 0.0
    largest_loss = losers.min() if len(losers) > 0 else 0.0
    
    # Profit factor
    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # R-multiples
    r_multiples = trades_df["r_multiple"]
    avg_r = r_multiples.mean() if len(r_multiples) > 0 else 0.0
    
    # Expectancy (average P&L per trade as % of capital)
    expectancy = (net_pnl / total_trades / initial_capital) * 100 if total_trades > 0 else 0.0
    
    # Drawdown calculation
    if len(equity_curve) > 0:
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve - rolling_max
        drawdown_pct = (drawdown / rolling_max) * 100
        max_drawdown = drawdown_pct.min()
        max_drawdown_dollars = drawdown.min()
    else:
        max_drawdown = 0.0
        max_drawdown_dollars = 0.0
    
    # Sharpe ratio (if we have returns)
    sharpe = None
    if len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Annualize assuming ~8760 hours per year for hourly data
            # Adjust based on actual data frequency
            periods_per_year = 365 * 24  # Assuming hourly data roughly
            sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    
    # Activity metrics
    if len(trades) > 0:
        first_trade = min(t.entry_time for t in trades)
        last_trade = max(t.exit_time for t in trades)
        duration_days = (last_trade - first_trade).total_seconds() / 86400
        trades_per_day = total_trades / duration_days if duration_days > 0 else 0.0
        avg_hold_time = trades_df["duration_hours"].mean()
    else:
        trades_per_day = 0.0
        avg_hold_time = 0.0
    
    # Costs
    total_fees = trades_df["fees"].sum()
    total_funding = trades_df["funding"].sum()
    total_slippage = trades_df["slippage"].sum()
    total_costs = total_fees + total_funding + total_slippage
    
    return PerformanceMetrics(
        net_return=net_return,
        net_return_dollars=net_pnl,
        gross_return_dollars=gross_pnl,
        max_drawdown=max_drawdown,
        max_drawdown_dollars=max_drawdown_dollars,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_r_multiple=avg_r,
        expectancy=expectancy,
        sharpe_ratio=sharpe,
        trades_per_day=trades_per_day,
        avg_hold_time_hours=avg_hold_time,
        total_fees=total_fees,
        total_funding=total_funding,
        total_slippage=total_slippage,
        total_costs=total_costs,
        pnl_before_costs=gross_pnl,
        pnl_after_costs=net_pnl,
    )
