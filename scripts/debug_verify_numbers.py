"""
Debug script to verify model numbers are correct.
"""
import sys
sys.path.insert(0, '.')

from data import CandleFetcher
from backtest import BacktestEngine, CostModel
from strategies import STRATEGIES

# Fetch data
print("=" * 60)
print("VERIFYING MA CROSSOVER NUMBERS")
print("=" * 60)

fetcher = CandleFetcher(coin="BTC", use_cache=True)
data = fetcher.fetch_data(interval="1h", window="90d")

print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"First close: ${data.iloc[0]['close']:.2f}")
print(f"Last close: ${data.iloc[-1]['close']:.2f}")
print()

# Run MA Crossover with different risk values
engine = BacktestEngine(cost_model=CostModel())
ma_strategy = STRATEGIES["ma_crossover"]()

print("Testing different risk levels:")
print("-" * 60)

for risk in [0.01, 0.02, 0.05]:
    result = engine.run(ma_strategy, data.copy(), 10000, risk)
    print(f"Risk {risk*100:.0f}%: ROI={result.metrics.net_return:+.2f}%, "
          f"Trades={result.metrics.total_trades}, "
          f"Win Rate={result.metrics.win_rate:.1f}%, "
          f"Profit Factor={result.metrics.profit_factor:.2f}")

print()
print("=" * 60)
print("TRADE DETAILS (Risk=2%)")
print("=" * 60)

result = engine.run(ma_strategy, data.copy(), 10000, 0.02)
for i, trade in enumerate(result.trades, 1):
    print(f"Trade {i}: {trade.side.upper()} "
          f"Entry=${trade.entry_price:.2f} Exit=${trade.exit_price:.2f} "
          f"PnL=${trade.pnl_net:+.2f} ({trade.exit_reason})")

print()
print(f"Total P&L: ${sum(t.pnl_net for t in result.trades):+.2f}")
print(f"Final Equity: ${result.final_equity:,.2f}")
print(f"Net Return: {result.metrics.net_return:+.2f}%")
