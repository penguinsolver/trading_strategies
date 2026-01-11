"""Test all strategies and find profitable ones."""
from strategies import STRATEGIES
from backtest import BacktestEngine, CostModel
from data import CandleFetcher
import pandas as pd

print(f'Total strategies: {len(STRATEGIES)}')
print('='*70)

# Fetch real 15m data
fetcher = CandleFetcher(coin='BTC', use_cache=True)
data = fetcher.fetch_data(interval='15m', window='7d')
print(f'Fetched {len(data)} 15m candles')

cost_model = CostModel(maker_fee=0.0001, taker_fee=0.00035, slippage_bps=1.0)
engine = BacktestEngine(cost_model=cost_model)

results = []
for key in STRATEGIES.keys():
    try:
        strategy = STRATEGIES[key]()
        result = engine.run(strategy, data.copy(), initial_capital=10000, risk_per_trade=0.01)
        m = result.metrics
        results.append({
            'strategy': key,
            'return': m.net_return,
            'trades': m.total_trades,
            'winrate': m.win_rate,
            'max_dd': m.max_drawdown
        })
    except Exception as e:
        results.append({
            'strategy': key,
            'return': -999,
            'trades': 0,
            'winrate': 0,
            'max_dd': 0
        })
        print(f'ERROR {key}: {str(e)[:60]}')

# Sort by return
results.sort(key=lambda x: x['return'], reverse=True)

print('\n' + '='*70)
print('TOP STRATEGIES BY RETURN (15min, 7d):')
print('='*70)
for r in results[:15]:
    status = '[5%+]' if r['return'] >= 5 else ('[OK]' if r['return'] > 0 else '[NEG]')
    print(f"{status} {r['strategy']:25s}: {r['return']:+7.2f}% | {r['trades']:3d} trades | WR: {r['winrate']:5.1f}%")

# Check if any >= 5%
has_5pct = any(r['return'] >= 5 for r in results)
print('\n' + '='*70)
if has_5pct:
    print('SUCCESS: Found strategy with 5%+ returns!')
else:
    print('NO 5%+ strategy found. Need more strategies.')
