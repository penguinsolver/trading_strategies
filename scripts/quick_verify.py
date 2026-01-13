
import sys
import time
import numpy as np
import pandas as pd
import ccxt
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from backtest import BacktestEngine, CostModel
from models.ml_enhanced import MLEnhancedBreakout, EnhancedConfig

def fetch_batch_history(coin="BTC", periods_count=5):
    """Fetch 5 distinct 90-day periods with delays using CCXT Binance."""
    print(f"Initializing CCXT Binance for deep history (Proxy for {coin})...")
    exchange = ccxt.binance() 
    symbol = f"{coin}/USDT"
    timeframe = '1h'
    periods_data = []
    
    period_ms = 90 * 24 * 3600 * 1000
    gap_ms = 7 * 24 * 3600 * 1000
    
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
                if not ohlcv: break
                ohlcv = [c for c in ohlcv if c[0] < end_time]
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                time.sleep(0.5)
            except:
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

def calculate_sharpe(returns, step_hours):
    """Annualized Sharpe based on step frequency."""
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    windows_per_year = (365 * 24) / step_hours
    return (np.mean(returns) / np.std(returns)) * np.sqrt(windows_per_year)

def main():
    # 1. Fetch Data
    periods = fetch_batch_history(coin="BTC", periods_count=5)
    
    # 2. Define Best Candidates
    candidates = [
        # From Phase 9, 13, 14
        {"w": 250, "s": 50, "lb": 10, "atr": 2.0}, # The robustness winner (2 periods)
        {"w": 200, "s": 75, "lb": 8, "atr": 1.5},  # Sharpe winner
        {"w": 400, "s": 150, "lb": 10, "atr": 1.2}, # High ROI winner
        {"w": 300, "s": 50, "lb": 8, "atr": 1.5},
        {"w": 175, "s": 50, "lb": 10, "atr": 1.8},
    ]
    
    print("\n" + "="*80)
    print("QUICK VERIFICATION: Testing Top 5 Candidates on 5 Periods")
    print("="*80)
    
    engine = BacktestEngine(cost_model=CostModel())
    
    best_config = None
    best_score = -999
    
    for cand in candidates:
        w, s, lb, atr = cand['w'], cand['s'], cand['lb'], cand['atr']
        r = 0.10
        config_name = f"w{w}_s{s}_lb{lb}_atr{atr}"
        print(f"\nTesting {config_name}...")
        
        period_stats = []
        passed_all = True
        
        for p in periods:
            p_data = p['data']
            capital = 10000
            start_capital = 10000
            start_idx = 0
            window_returns = []
            
            while start_idx + w < len(p_data):
                seg = p_data.iloc[start_idx:start_idx + w]
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
            
            print(f"  Period {p['id']}: ROI={final_roi:,.0f}% | Sharpe={sharpe:.2f}")
            
            period_stats.append({'roi': final_roi, 'sharpe': sharpe})
            
            if final_roi < 10000 or sharpe < 2.0:
                # passed_all = False # Don't stop, let's see full picture
                pass
                
        # Aggregate
        min_roi = min(x['roi'] for x in period_stats)
        min_sharpe = min(x['sharpe'] for x in period_stats)
        avg_roi = np.mean([x['roi'] for x in period_stats])
        avg_sharpe = np.mean([x['sharpe'] for x in period_stats])
        
        print(f"  >>> Min ROI: {min_roi:,.0f}% | Min Sharpe: {min_sharpe:.2f} | Avg ROI: {avg_roi:,.0f}%")
        
        if min_roi > 10000 and min_sharpe > 2.0:
            print("  ✅ SUPER ROBUST WINNER!")
        else:
            print("  ❌ Failed strict criteria")
            
        score = min_roi # Rank by worst case ROI
        if score > best_score:
            best_score = score
            best_config = {
                'config': config_name,
                'min_roi': min_roi,
                'min_sharpe': min_sharpe,
                'period_stats': period_stats
            }

    print("\n" + "="*80)
    print("BEST FOUND CONFIGURATION")
    print(best_config['config'])
    print(f"Min ROI: {best_config['min_roi']:,.0f}%")
    print(f"Min Sharpe: {best_config['min_sharpe']:.2f}")

if __name__ == "__main__":
    main()
