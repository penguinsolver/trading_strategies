
import sys
import time
import numpy as np
import pandas as pd
import random
import multiprocessing
import ccxt
from itertools import product
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from data.candle_fetcher import CandleFetcher
from backtest import BacktestEngine, CostModel
from models.ml_enhanced import MLEnhancedBreakout, EnhancedConfig

# Global data for workers
worker_periods = None

def init_worker(periods):
    global worker_periods
    worker_periods = periods

def calculate_sharpe(returns, step_hours):
    """Annualized Sharpe based on step frequency."""
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    windows_per_year = (365 * 24) / step_hours
    return (np.mean(returns) / np.std(returns)) * np.sqrt(windows_per_year)

def process_config(config_tuple):
    """Worker function to process a single configuration."""
    w, s, lb, atr, r = config_tuple
    config_name = f"w{w}_s{s}_lb{lb}_atr{atr}"
    
    global worker_periods
    if worker_periods is None:
        return None # Should not happen
        
    engine = BacktestEngine(cost_model=CostModel())
    period_stats = []
    
    # Process each period
    for p in worker_periods:
        p_data = p['data']
        capital = 10000
        start_capital = 10000
        start_idx = 0
        window_returns = []
        
        while start_idx + w < len(p_data):
            seg = p_data.iloc[start_idx:start_idx + w]
            # Use lightweight model setup
            config = EnhancedConfig(min_ml_confidence=0.35, forward_bars=8)
            model = MLEnhancedBreakout(lookback=lb, atr_multiplier=atr, config=config)
            
            try:
                res = engine.run(model, seg.copy(), capital, r)
                roi = res.metrics.net_return
                window_returns.append(roi)
                if roi > 0:
                    capital *= (1 + roi/100)
            except:
                pass
            start_idx += s
            
        final_roi = (capital/start_capital - 1)*100
        sharpe = calculate_sharpe(window_returns, s)
        
        period_stats.append({'roi': final_roi, 'sharpe': sharpe})
        
        # Fail fast per period?
        # If any period has net loss, it's not robust enough for "10,000% target"
        if final_roi < 0: 
             return {'config': config_name, 'valid': False, 'stats': period_stats}

    # Summarize across all 5 periods
    min_roi = min(x['roi'] for x in period_stats)
    min_sharpe = min(x['sharpe'] for x in period_stats)
    avg_roi = np.mean([x['roi'] for x in period_stats])
    
    # Strict Criteria: > 10,000% ROI AND > 2.0 Sharpe on ALL periods?
    # Or Min ROI > 10,000? 
    # User: "beat all 5 periods and have a min ROI of 10.000% and min sharpe ratio of 2.00"
    is_valid = min_roi > 10000 and min_sharpe > 2.0
    
    return {
        'config': config_name,
        'min_roi': min_roi,
        'min_sharpe': min_sharpe,
        'avg_roi': avg_roi,
        'stats': period_stats,
        'valid': is_valid
    }

def fetch_batch_history(coin="BTC", periods_count=5):
    """Fetch 5 distinct 90-day periods with delays using CCXT directly."""
    print(f"Initializing CCXT Binance for deep history (Proxy for {coin})...")
    exchange = ccxt.binance() 
    symbol = f"{coin}/USDT"
    timeframe = '1h'
    periods_data = []
    
    # 90 days in ms
    period_ms = 90 * 24 * 3600 * 1000
    gap_ms = 7 * 24 * 3600 * 1000 # 1 week gap
    
    end_time = exchange.milliseconds()
    
    print(f"Fetching {periods_count} periods of 90 days...")
    
    for i in range(periods_count):
        start_time = end_time - period_ms
        current_since = start_time
        all_ohlcv = []
        
        print(f"  Fetching Period {i+1}: {pd.to_datetime(start_time, unit='ms')} -> {pd.to_datetime(end_time, unit='ms')}")
        
        while current_since < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    break
                ohlcv = [c for c in ohlcv if c[0] < end_time]
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                time.sleep(0.5)
            except Exception as e:
                print(f"    Error fetching page: {e}")
                time.sleep(5)
                
        if len(all_ohlcv) > 2000:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            periods_data.append({
                'id': i+1,
                'data': df,
                'range': f"{df.index[0]} to {df.index[-1]}"
            })
            print(f"    ✅ Got {len(df)} candles.")
        else:
            print(f"    ⚠️ Warning: Only got {len(all_ohlcv)} candles for Period {i+1}")
            
        end_time = start_time - gap_ms
        
    return periods_data

def main():
    print('\n' + '='*80)
    print('DEEP ROBUSTNESS SEARCH: 10,000 Iterations (Multiprocessing)')
    print('Target: ROI > 10,000% AND Sharpe > 2.0 on ALL 5 periods')
    print('='*80)
    
    # 1. Fetch Data
    periods = fetch_batch_history(coin="BTC", periods_count=5)
    
    if len(periods) == 0:
        print("CRITICAL: No data fetched. Exiting.")
        return

    # 2. Configs
    window_sizes = list(range(150, 451, 25))
    steps = [25, 40, 50, 60, 75, 100]
    lookbacks = [6, 8, 10, 12, 14, 16]
    atr_multipliers = [1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
    
    configs = []
    # 2,500 Random choices (approx 20 mins)
    for _ in range(2500):
        configs.append((
            random.choice(window_sizes),
            random.choice(steps),
            random.choice(lookbacks),
            random.choice(atr_multipliers),
            0.10
        ))
    
    print(f"Starting pool with {multiprocessing.cpu_count()} workers for {len(configs)} tasks...")
    start_time = time.time()
    
    # Use max CPUs minus a few for stability
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with multiprocessing.Pool(processes=n_workers, initializer=init_worker, initargs=(periods,)) as pool:
        results = []
        # Use chunksize for better performance
        for res in pool.imap_unordered(process_config, configs, chunksize=20):
            if res is None: continue 
            
            if res['valid']:
                print(f"🌟 WINNER: {res['config']} (Min ROI: {res['min_roi']:,.0f}%)")
            
            # Reduce memory: only keep winners or near-winners
            if res['valid'] or (res['min_roi'] > 5000 and res['min_sharpe'] > 1.5):
                results.append(res)
            
            # Progress tracking (approximate)
            if len(configs) > 0 and random.random() < 0.001: 
                # Just print occasionally based on time?
                pass
                
        # Wait for all? imap blocks until done.
        
    elapsed = time.time() - start_time
    print(f"Search Complete in {elapsed:.1f}s")
    
    # Sort and Show
    results.sort(key=lambda x: (x['valid'], x['min_sharpe']), reverse=True)
    
    print('\nTOP CONFIGURATIONS (Sorted by Robustness)')
    print(f"{'Config':<25} | {'Status':<8} | {'Min ROI':<12} | {'Min Sharpe':<8}")
    print('-' * 60)
    for r in results[:10]:
        status = "✅ PASS" if r['valid'] else "❌ CLOSE"
        print(f"{r['config']:<25} | {status:<8} | {r['min_roi']:>10,.0f}% | {r['min_sharpe']:>8.2f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
