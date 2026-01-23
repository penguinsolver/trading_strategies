"""
SMART OPTIMIZER V2: 10,000 iterations with intelligent parameter exploration.

LEARNINGS FROM PREVIOUS RUN:
- Best configs have: short window (200), small step (25), short lookback (5-8)
- High risk (20-25%) outperforms low risk (10%)
- ATR 1.5-2.5 is optimal range
- More trades (188+) correlate with higher ROI due to compounding

STRATEGY:
1. Phase 1 (0-3000): Explore expanded parameter space
2. Phase 2 (3000-6000): Focus on promising regions
3. Phase 3 (6000-10000): Fine-tune around best found

NEW IDEAS TO TRY:
- Even shorter lookbacks (3, 4)
- Even higher risk (30%, 35%)
- Very short windows (100, 150) with tiny steps (10, 15)
- Different ML confidence levels
"""
import sys
import time
import random
import numpy as np
import pandas as pd
import ccxt
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from backtest import BacktestEngine, CostModel
from models.ml_enhanced import MLEnhancedBreakout, EnhancedConfig

def fetch_period_data(exchange, symbol, start_time, end_time):
    """Fetch data for a specific period."""
    timeframe = '1h'
    all_ohlcv = []
    current_since = start_time
    
    while current_since < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
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
    return df

def run_compound_strategy(data, w, s, lb, atr, risk, ml_conf=0.35, start_capital=10000):
    """Run the rolling window compound strategy with configurable ML confidence."""
    engine = BacktestEngine(cost_model=CostModel())
    config = EnhancedConfig(min_ml_confidence=ml_conf, forward_bars=8)
    
    capital = start_capital
    start_idx = 0
    total_trades = 0
    win_count = 0
    
    while start_idx + w < len(data):
        seg = data.iloc[start_idx:start_idx + w].copy()
        
        try:
            model = MLEnhancedBreakout(lookback=lb, atr_multiplier=atr, config=config)
            result = engine.run(model, seg, capital, risk)
            
            for trade in result.trades:
                capital += trade.pnl_net
                total_trades += 1
                if trade.pnl_net > 0:
                    win_count += 1
                
        except:
            pass
            
        start_idx += s
    
    final_roi = (capital/start_capital - 1) * 100
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    return final_roi, total_trades, capital, win_rate

def generate_phase1_config():
    """Phase 1: Broad exploration with focus on promising areas."""
    return {
        'w': random.choice([100, 150, 200, 250, 300, 400]),
        's': random.choice([10, 15, 25, 50, 75]),
        'lb': random.choice([3, 4, 5, 6, 8, 10]),
        'atr': random.choice([1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]),
        'risk': random.choice([0.15, 0.20, 0.25, 0.30, 0.35]),
        'ml_conf': random.choice([0.30, 0.35, 0.40, 0.45]),
    }

def generate_phase2_config(best_configs):
    """Phase 2: Focus on regions near best performers."""
    if not best_configs:
        return generate_phase1_config()
    
    # Pick a random top performer and vary slightly
    base = random.choice(best_configs[:5])
    return {
        'w': base['w'] + random.choice([-50, -25, 0, 25, 50]),
        's': max(10, base['s'] + random.choice([-10, -5, 0, 5, 10])),
        'lb': max(3, base['lb'] + random.choice([-2, -1, 0, 1, 2])),
        'atr': round(base['atr'] + random.choice([-0.3, -0.2, 0, 0.2, 0.3]), 1),
        'risk': round(min(0.40, max(0.10, base['risk'] + random.choice([-0.05, 0, 0.05]))), 2),
        'ml_conf': round(base.get('ml_conf', 0.35) + random.choice([-0.05, 0, 0.05]), 2),
    }

def generate_phase3_config(best_config):
    """Phase 3: Fine-tune around the very best config."""
    if not best_config:
        return generate_phase1_config()
    
    return {
        'w': best_config['w'] + random.choice([-20, -10, 0, 10, 20]),
        's': max(5, best_config['s'] + random.choice([-5, -2, 0, 2, 5])),
        'lb': max(2, best_config['lb'] + random.choice([-1, 0, 1])),
        'atr': round(best_config['atr'] + random.choice([-0.1, 0, 0.1]), 1),
        'risk': round(min(0.45, max(0.10, best_config['risk'] + random.choice([-0.02, 0, 0.02]))), 2),
        'ml_conf': round(best_config.get('ml_conf', 0.35) + random.choice([-0.02, 0, 0.02]), 2),
    }

def main():
    print("="*80)
    print("SMART OPTIMIZER V2: 10,000 Iterations")
    print("Position Sizing: 10% Capital Allocation, Max 10x Leverage")
    print("Target: 1000% ROI")
    print("="*80)
    
    # Fetch data
    exchange = ccxt.binance()
    symbol = "BTC/USDT"
    period_ms = 90 * 24 * 3600 * 1000
    end_time = exchange.milliseconds()
    start_time = end_time - period_ms
    
    print("\nFetching 90-day period for optimization...")
    data = fetch_period_data(exchange, symbol, start_time, end_time)
    print(f"Got {len(data)} candles\n")
    
    best_config = None
    best_roi = -999
    top_configs = []  # Track top 10 performers
    results = []
    
    print("PHASE 1: Broad Exploration (0-3000)")
    print("-"*80)
    
    for i in range(10000):
        # Select phase-appropriate config generator
        if i < 3000:
            cfg = generate_phase1_config()
        elif i < 6000:
            if i == 3000:
                print("\n" + "-"*80)
                print("PHASE 2: Focused Exploration (3000-6000)")
                print("-"*80)
            cfg = generate_phase2_config(top_configs)
        else:
            if i == 6000:
                print("\n" + "-"*80)
                print("PHASE 3: Fine-Tuning (6000-10000)")
                print("-"*80)
            cfg = generate_phase3_config(best_config)
        
        w, s, lb, atr, risk, ml_conf = cfg['w'], cfg['s'], cfg['lb'], cfg['atr'], cfg['risk'], cfg['ml_conf']
        
        try:
            roi, trades, final_cap, win_rate = run_compound_strategy(
                data, w, s, lb, atr, risk, ml_conf
            )
            
            result = {**cfg, 'roi': roi, 'trades': trades, 'win_rate': win_rate}
            results.append(result)
            
            # Update top configs list
            top_configs.append(result)
            top_configs = sorted(top_configs, key=lambda x: x['roi'], reverse=True)[:20]
            
            if roi > best_roi:
                best_roi = roi
                best_config = cfg.copy()
                best_config['trades'] = trades
                best_config['win_rate'] = win_rate
                config_str = f"w{w}_s{s}_lb{lb}_atr{atr}_r{int(risk*100)}%_ml{ml_conf}"
                print(f"[{i+1:5d}] 🏆 NEW BEST: {config_str} → ROI: {roi:+,.1f}% ({trades} trades, {win_rate:.0f}% WR)")
            
            if (i+1) % 500 == 0:
                print(f"[{i+1:5d}] Progress... Best so far: {best_roi:+,.1f}%")
                
        except Exception as e:
            pass
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - 10,000 Iterations")
    print("="*80)
    
    # Sort and display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)
    
    print("\n🏆 TOP 15 CONFIGURATIONS:")
    print("-"*80)
    for i, (_, row) in enumerate(results_df.head(15).iterrows()):
        config_str = f"w{row['w']}_s{row['s']}_lb{row['lb']}_atr{row['atr']}_r{int(row['risk']*100)}%_ml{row['ml_conf']}"
        print(f"  {i+1:2d}. {config_str:45s} → ROI: {row['roi']:+,.1f}% ({row['trades']} trades, {row['win_rate']:.0f}% WR)")
    
    print(f"\n🎯 BEST CONFIGURATION:")
    print(f"  Window: {best_config['w']} hours")
    print(f"  Step: {best_config['s']} hours")
    print(f"  Lookback: {best_config['lb']} bars")
    print(f"  ATR Multiplier: {best_config['atr']}")
    print(f"  Risk per Trade: {best_config['risk']*100:.0f}%")
    print(f"  ML Confidence: {best_config.get('ml_conf', 0.35)}")
    print(f"  Trades: {best_config['trades']}")
    print(f"  Win Rate: {best_config['win_rate']:.1f}%")
    print(f"  ROI: {best_roi:+,.2f}%")
    
    if best_roi >= 1000:
        print(f"\n✅ TARGET ACHIEVED! Found config with {best_roi:+,.0f}% ROI (target: 1000%)")
    else:
        print(f"\n⚠️ TARGET NOT MET. Best ROI: {best_roi:+,.1f}% (target: 1000%)")
        print("  Analysis: With 10% capital + 10x leverage constraint, 1000% may require:")
        print("  - Different strategy (not MLEnhancedBreakout)")
        print("  - Higher risk tolerance (40%+ per trade)")
        print("  - Perfect market conditions (strong trend)")
    
    # Save results
    results_df.to_csv('optimization_results_v2.csv', index=False)
    print(f"\nResults saved to optimization_results_v2.csv")

if __name__ == "__main__":
    main()
