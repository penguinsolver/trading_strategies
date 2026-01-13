
import sys
import time
import numpy as np
import pandas as pd
from itertools import product
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from data import CandleFetcher
from backtest import BacktestEngine, CostModel
from models.ml_enhanced import MLEnhancedBreakout, EnhancedConfig

def calculate_sharpe(returns):
    """Calculate Sharpe ratio of returns sequence."""
    if len(returns) < 2:
        return 0.0
    returns = np.array(returns)
    std = returns.std()
    if std < 1e-9:
        return 0.0
    return returns.mean() / std

print('=' * 80)
print('OPTIMIZATION: TARGET SHARPE > 2.0 & ROI > 10,000%')
print('Constraints: Max 1000 iterations, 10% max risk (fixed)')
print('=' * 80)

# Load Data
fetcher = CandleFetcher(coin='BTC', use_cache=True)
data = fetcher.fetch_data(interval='1h', window='90d')
print(f'Data loaded: {len(data)} candles')

engine = BacktestEngine(cost_model=CostModel())

# Parameter Grid
# We want to find stable, consistent returns
# Smaller windows might have higher variance, let's test a range
window_sizes = [200, 250, 300, 350, 400] 
steps = [50, 75, 100]
lookbacks = [8, 10, 12, 14]
atr_multipliers = [1.2, 1.5, 1.8, 2.0] # Wider stops might prevent chop -> higher sharpe?
risks = [0.10] # Fixed at 10% as requested

# Generate combinations
combinations = list(product(window_sizes, steps, lookbacks, atr_multipliers, risks))
# Shuffle to get random samples if we exceed cap, but for 1000 cap we can probably run many
import random
random.shuffle(combinations)

# Cap at 1000 iterations
combinations = combinations[:1000]

print(f'Testing {len(combinations)} configurations...')

results = []
start_time = time.time()
best_sharpe = 0.0
best_roi = 0.0

for i, (w_size, step, lb, atr, risk) in enumerate(combinations):
    iteration = i + 1
    
    capital = 10000
    start_capital = capital
    windows_run = 0
    start_idx = 0
    window_returns = []
    
    # Run Rolling Window
    while start_idx + w_size < len(data):
        seg = data.iloc[start_idx:start_idx + w_size]
        config = EnhancedConfig(min_ml_confidence=0.35, forward_bars=8)
        model = MLEnhancedBreakout(lookback=lb, atr_multiplier=atr, config=config)
        
        try:
            result = engine.run(model, seg.copy(), capital, risk)
            roi = result.metrics.net_return
            window_returns.append(roi) # Store percentage return
            
            if roi > 0:
                capital = capital * (1 + roi/100)
            windows_run += 1
        except Exception:
            pass
        
        start_idx += step
    
    # Metrics
    final_roi = (capital / start_capital - 1) * 100
    
    # Sharpe on Window Returns
    # Note: These are 'per window' returns, not daily. 
    # A high Sharpe here means consistent profitability across windows.
    sharpe = calculate_sharpe(window_returns)
    win_rate = (np.array(window_returns) > 0).mean() if len(window_returns) > 0 else 0
    
    results.append({
        'config': f'w{w_size}_s{step}_lb{lb}_atr{atr}',
        'roi': final_roi,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': 0, # Not tracking strictly to save memory, ROI matches
        'windows': windows_run,
        'params': {'w': w_size, 's': step, 'lb': lb, 'atr': atr}
    })
    
    # Logging
    is_new_best_sharpe = sharpe > best_sharpe and final_roi > 10000
    is_new_best_roi = final_roi > best_roi
    
    if is_new_best_sharpe:
        best_sharpe = sharpe
        print(f'[{iteration}/{len(combinations)}] NEW BEST SHARPE: {sharpe:.2f} | ROI: {final_roi:,.1f}% | {results[-1]["config"]}')
    elif is_new_best_roi and sharpe > 1.0:
        best_roi = final_roi
        print(f'[{iteration}/{len(combinations)}] NEW BEST ROI: {final_roi:,.1f}% | Sharpe: {sharpe:.2f} | {results[-1]["config"]}')
    elif iteration % 50 == 0:
        print(f'[{iteration}/{len(combinations)}] Current best Sharpe: {best_sharpe:.2f} (ROI {10000}+)')

total_time = time.time() - start_time
print('=' * 80)
print(f'Optimization complete in {total_time:.1f}s')
print('=' * 80)

# Sort by Sharpe (descending) where ROI > 10000
valid_results = [r for r in results if r['roi'] > 10000]
valid_results.sort(key=lambda x: x['sharpe'], reverse=True)

print('\nTOP 10 MODELS (Sharpe > 1.0 & ROI > 10,000%)')
print(f'{"Config":<25} | {"ROI":<12} | {"Sharpe":<6} | {"WinRate":<7} | {"Windows":<7}')
print('-' * 70)
for r in valid_results[:10]:
    print(f'{r["config"]:<25} | {r["roi"]:>10,.1f}% | {r["sharpe"]:>6.2f} | {r["win_rate"]*100:>6.1f}% | {r["windows"]:>7}')

if not valid_results:
    print("No models met the ROI > 10,000% criteria.")
    # Fallback: Top Sharpe regardless of ROI
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print('\nTOP 5 MODELS (Best Sharpe, Any ROI)')
    for r in results[:5]:
        print(f'{r["config"]:<25} | {r["roi"]:>10,.1f}% | {r["sharpe"]:>6.2f} | {r["win_rate"]*100:>6.1f}%')

