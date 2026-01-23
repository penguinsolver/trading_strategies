"""
5-PERIOD ROBUSTNESS OPTIMIZER
Target: ROI > 10,000% AND Sharpe > 2.0 on ALL 5 periods
Max Iterations: 10,000
Approach: Batch learning - analyze patterns every 1000 iterations and refocus

LEARNINGS FROM PREVIOUS RUNS:
- Small step (10-25) generates more trades → stronger compounding
- Short lookback (3-5) outperforms longer (10+)
- Higher risk (20-35%) needed for higher returns
- Lower ML confidence (0.3-0.4) allows more signals
- ATR 1.2-2.5 optimal range
"""
import sys
import time
import random
import json
import numpy as np
import pandas as pd
import ccxt
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from backtest import BacktestEngine, CostModel
from models.ml_enhanced import MLEnhancedBreakout, EnhancedConfig

def fetch_all_periods(exchange, symbol, num_periods=5):
    """Fetch 5 distinct non-overlapping 90-day periods."""
    periods = []
    period_ms = 90 * 24 * 3600 * 1000
    gap_ms = 7 * 24 * 3600 * 1000
    end_time = exchange.milliseconds()
    
    for i in range(num_periods):
        start_time = end_time - period_ms
        print(f"  Fetching Period {i+1}...", end=" ")
        
        all_ohlcv = []
        current_since = start_time
        
        while current_since < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                if not ohlcv: break
                ohlcv = [c for c in ohlcv if c[0] < end_time]
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                time.sleep(0.15)
            except:
                time.sleep(1)
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        periods.append({
            'id': i + 1,
            'data': df,
            'range': f"{df.index[0]} to {df.index[-1]}"
        })
        print(f"✅ {len(df)} candles ({periods[-1]['range']})")
        
        end_time = start_time - gap_ms
    
    return periods

def calculate_sharpe(returns, step_hours=50):
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    windows_per_year = (365 * 24) / step_hours
    return (np.mean(returns) / np.std(returns)) * np.sqrt(windows_per_year)

def run_on_period(data, w, s, lb, atr, risk, ml_conf=0.35, start_capital=10000):
    """Run strategy on one period, return ROI, Sharpe, trades, trade_details."""
    engine = BacktestEngine(cost_model=CostModel())
    config = EnhancedConfig(min_ml_confidence=ml_conf, forward_bars=8)
    
    capital = start_capital
    start_idx = 0
    window_returns = []
    trade_details = []
    
    while start_idx + w < len(data):
        seg = data.iloc[start_idx:start_idx + w].copy()
        capital_before = capital
        
        try:
            model = MLEnhancedBreakout(lookback=lb, atr_multiplier=atr, config=config)
            result = engine.run(model, seg, capital, risk)
            
            for trade in result.trades:
                pnl = trade.pnl_net
                pnl_pct = (pnl / capital) * 100
                capital += pnl
                
                trade_details.append({
                    'entry_time': str(trade.entry_time),
                    'exit_time': str(trade.exit_time),
                    'direction': trade.side,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'stop_loss': trade.stop_price,
                    'position_size': trade.size,
                    'pnl_dollars': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': capital
                })
            
            window_return = (capital - capital_before) / capital_before
            window_returns.append(window_return)
                
        except:
            pass
            
        start_idx += s
    
    final_roi = (capital / start_capital - 1) * 100
    sharpe = calculate_sharpe(window_returns, s)
    
    return final_roi, sharpe, len(trade_details), trade_details

def test_config_on_all_periods(periods, w, s, lb, atr, risk, ml_conf):
    """Test a config on all 5 periods. Return results for each."""
    results = []
    all_trades = []
    
    for p in periods:
        roi, sharpe, num_trades, trades = run_on_period(
            p['data'], w, s, lb, atr, risk, ml_conf
        )
        results.append({
            'period': p['id'],
            'roi': roi,
            'sharpe': sharpe,
            'trades': num_trades
        })
        all_trades.extend(trades)
    
    return results, all_trades

def check_targets(results, roi_target=10000, sharpe_target=2.0):
    """Check if ALL periods meet targets."""
    all_pass = all(r['roi'] >= roi_target and r['sharpe'] >= sharpe_target for r in results)
    min_roi = min(r['roi'] for r in results)
    min_sharpe = min(r['sharpe'] for r in results)
    avg_roi = np.mean([r['roi'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    
    return all_pass, min_roi, min_sharpe, avg_roi, avg_sharpe

def generate_config(phase, top_configs, best_config):
    """Generate config based on phase and learnings."""
    if phase == 1:
        # Aggressive exploration
        return {
            'w': random.choice([100, 150, 200, 250]),
            's': random.choice([10, 15, 20, 25]),
            'lb': random.choice([3, 4, 5, 6]),
            'atr': random.choice([1.0, 1.2, 1.5, 2.0, 2.5]),
            'risk': random.choice([0.20, 0.25, 0.30, 0.35]),
            'ml_conf': random.choice([0.30, 0.35, 0.40]),
        }
    elif phase == 2 and top_configs:
        # Focus near best performers
        base = random.choice(top_configs[:10])
        return {
            'w': max(100, base['w'] + random.choice([-25, 0, 25])),
            's': max(10, base['s'] + random.choice([-5, 0, 5])),
            'lb': max(3, base['lb'] + random.choice([-1, 0, 1])),
            'atr': round(max(0.5, base['atr'] + random.choice([-0.2, 0, 0.2])), 1),
            'risk': round(min(0.40, max(0.15, base['risk'] + random.choice([-0.05, 0, 0.05]))), 2),
            'ml_conf': round(base['ml_conf'] + random.choice([-0.05, 0, 0.05]), 2),
        }
    else:
        # Fine-tune around best
        if best_config:
            return {
                'w': max(100, best_config['w'] + random.choice([-10, 0, 10])),
                's': max(5, best_config['s'] + random.choice([-2, 0, 2])),
                'lb': max(2, best_config['lb'] + random.choice([-1, 0, 1])),
                'atr': round(max(0.5, best_config['atr'] + random.choice([-0.1, 0, 0.1])), 1),
                'risk': round(min(0.45, max(0.15, best_config['risk'] + random.choice([-0.02, 0, 0.02]))), 2),
                'ml_conf': round(best_config['ml_conf'] + random.choice([-0.02, 0, 0.02]), 2),
            }
        return generate_config(1, [], None)

def main():
    print("="*80)
    print("5-PERIOD ROBUSTNESS OPTIMIZER")
    print("Target: ROI > 10,000% AND Sharpe > 2.0 on ALL 5 periods")
    print("Max Iterations: 10,000")
    print("="*80)
    
    # Fetch all 5 periods
    exchange = ccxt.binance()
    symbol = "BTC/USDT"
    
    print("\nFetching 5 independent 90-day periods...")
    periods = fetch_all_periods(exchange, symbol, 5)
    print(f"✅ Loaded {len(periods)} periods\n")
    
    best_config = None
    best_score = -999  # Score based on min_roi
    top_configs = []
    all_results = []
    target_found = False
    
    # Phase thresholds
    phase1_end = 3000
    phase2_end = 7000
    
    print("PHASE 1: Aggressive Exploration (0-3000)")
    print("-"*80)
    
    for i in range(10000):
        # Determine phase
        if i < phase1_end:
            phase = 1
        elif i < phase2_end:
            if i == phase1_end:
                print(f"\n{'='*80}")
                print("PHASE 2: Focused Search (3000-7000)")
                print(f"Best so far: ROI={best_score:+,.0f}%")
                print("="*80)
            phase = 2
        else:
            if i == phase2_end:
                print(f"\n{'='*80}")
                print("PHASE 3: Fine-Tuning (7000-10000)")
                print(f"Best so far: ROI={best_score:+,.0f}%")
                print("="*80)
            phase = 3
        
        cfg = generate_config(phase, top_configs, best_config)
        w, s, lb, atr, risk, ml_conf = cfg['w'], cfg['s'], cfg['lb'], cfg['atr'], cfg['risk'], cfg['ml_conf']
        
        try:
            results, trades = test_config_on_all_periods(periods, w, s, lb, atr, risk, ml_conf)
            all_pass, min_roi, min_sharpe, avg_roi, avg_sharpe = check_targets(results)
            
            cfg['results'] = results
            cfg['min_roi'] = min_roi
            cfg['min_sharpe'] = min_sharpe
            cfg['avg_roi'] = avg_roi
            cfg['avg_sharpe'] = avg_sharpe
            cfg['all_pass'] = all_pass
            
            all_results.append(cfg)
            
            # Track top performers by min_roi
            top_configs.append(cfg)
            top_configs = sorted(top_configs, key=lambda x: x['min_roi'], reverse=True)[:50]
            
            if min_roi > best_score:
                best_score = min_roi
                best_config = cfg.copy()
                best_config['trades'] = trades
                
                config_str = f"w{w}_s{s}_lb{lb}_atr{atr}_r{int(risk*100)}%_ml{ml_conf}"
                print(f"[{i+1:5d}] 🏆 NEW BEST: {config_str}")
                print(f"        Min ROI: {min_roi:+,.0f}% | Min Sharpe: {min_sharpe:.2f} | Avg ROI: {avg_roi:+,.0f}%")
                for r in results:
                    print(f"        Period {r['period']}: ROI={r['roi']:+,.0f}% Sharpe={r['sharpe']:.2f}")
                
                if all_pass:
                    print(f"\n🎉 TARGET ACHIEVED! Config passes ALL criteria on ALL periods!")
                    target_found = True
            
            if (i+1) % 500 == 0:
                print(f"[{i+1:5d}] Progress... Best Min ROI: {best_score:+,.0f}%")
                
        except Exception as e:
            pass
    
    # Final report
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - 10,000 Iterations")
    print("="*80)
    
    print(f"\n🎯 BEST CONFIGURATION:")
    print(f"  Window: {best_config['w']} hours")
    print(f"  Step: {best_config['s']} hours")
    print(f"  Lookback: {best_config['lb']} bars")
    print(f"  ATR Multiplier: {best_config['atr']}")
    print(f"  Risk per Trade: {best_config['risk']*100:.0f}%")
    print(f"  ML Confidence: {best_config['ml_conf']}")
    print(f"\n📊 RESULTS:")
    print(f"  Min ROI (worst period): {best_config['min_roi']:+,.2f}%")
    print(f"  Min Sharpe (worst period): {best_config['min_sharpe']:.2f}")
    print(f"  Avg ROI (all periods): {best_config['avg_roi']:+,.2f}%")
    print(f"  Avg Sharpe (all periods): {best_config['avg_sharpe']:.2f}")
    
    print(f"\n📅 PERIOD-BY-PERIOD:")
    for r in best_config['results']:
        status = "✅ PASS" if r['roi'] >= 10000 and r['sharpe'] >= 2.0 else "❌ FAIL"
        print(f"  Period {r['period']}: ROI={r['roi']:+,.0f}% | Sharpe={r['sharpe']:.2f} | {status}")
    
    if target_found:
        print(f"\n✅ SUCCESS: Found config meeting ALL targets!")
    else:
        print(f"\n⚠️ TARGET NOT MET. Best configuration shown above.")
        print(f"  Consider: Different strategy, more iterations, or relaxed targets.")
    
    # Save detailed results
    output = {
        'best_config': {k: v for k, v in best_config.items() if k != 'trades'},
        'sample_trades': best_config['trades'][:20] if best_config.get('trades') else [],
        'target_achieved': target_found
    }
    
    with open('robustness_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to robustness_optimization_results.json")

if __name__ == "__main__":
    main()
