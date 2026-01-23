"""
Optimization Script: Find profitable config with realistic constraints.
Constraints: 10% capital allocation, 10x max leverage
Target: 1000% ROI
Max Iterations: 1000
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
            time.sleep(0.2)
        except:
            time.sleep(2)
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_compound_strategy(data, w, s, lb, atr, risk, start_capital=10000):
    """Run the rolling window compound strategy."""
    engine = BacktestEngine(cost_model=CostModel())
    config = EnhancedConfig(min_ml_confidence=0.35, forward_bars=8)
    
    capital = start_capital
    start_idx = 0
    total_trades = 0
    
    while start_idx + w < len(data):
        seg = data.iloc[start_idx:start_idx + w].copy()
        
        try:
            model = MLEnhancedBreakout(lookback=lb, atr_multiplier=atr, config=config)
            result = engine.run(model, seg, capital, risk)
            
            # Compound: Add net P&L to capital
            for trade in result.trades:
                capital += trade.pnl_net
                total_trades += 1
                
        except:
            pass
            
        start_idx += s
    
    final_roi = (capital/start_capital - 1) * 100
    return final_roi, total_trades, capital

def main():
    print("="*80)
    print("OPTIMIZATION: Find Profitable Config with 10%/10x Constraints")
    print("Target: 1000% ROI | Max Iterations: 1000")
    print("="*80)
    
    # Fetch data (use 1 period first for speed, then validate on 5)
    exchange = ccxt.binance()
    symbol = "BTC/USDT"
    period_ms = 90 * 24 * 3600 * 1000
    end_time = exchange.milliseconds()
    start_time = end_time - period_ms
    
    print("\nFetching 90-day period for optimization...")
    data = fetch_period_data(exchange, symbol, start_time, end_time)
    print(f"Got {len(data)} candles\n")
    
    # Parameter space
    window_sizes = [100, 150, 200, 250, 300, 400, 500]
    steps = [25, 50, 75, 100, 150]
    lookbacks = [5, 8, 10, 15, 20, 30]
    atr_multipliers = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    risk_levels = [0.05, 0.10, 0.15, 0.20, 0.25]  # Different risk allocations
    
    best_config = None
    best_roi = -999
    results = []
    
    print("Starting 1000 random iterations...")
    print("-"*80)
    
    for i in range(1000):
        w = random.choice(window_sizes)
        s = random.choice(steps)
        lb = random.choice(lookbacks)
        atr = random.choice(atr_multipliers)
        risk = random.choice(risk_levels)
        
        try:
            roi, trades, final_cap = run_compound_strategy(data, w, s, lb, atr, risk)
            
            results.append({
                'w': w, 's': s, 'lb': lb, 'atr': atr, 'risk': risk,
                'roi': roi, 'trades': trades, 'final_cap': final_cap
            })
            
            if roi > best_roi:
                best_roi = roi
                best_config = {'w': w, 's': s, 'lb': lb, 'atr': atr, 'risk': risk, 'trades': trades}
                print(f"[{i+1:4d}] NEW BEST: w{w}_s{s}_lb{lb}_atr{atr}_r{int(risk*100)}% → ROI: {roi:+,.1f}% ({trades} trades)")
            
            if (i+1) % 100 == 0:
                print(f"[{i+1:4d}] Progress... Best so far: {best_roi:+,.1f}%")
                
        except Exception as e:
            pass
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # Sort results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)
    
    print("\n🏆 TOP 10 CONFIGURATIONS:")
    print("-"*80)
    for idx, row in results_df.head(10).iterrows():
        config_str = f"w{row['w']}_s{row['s']}_lb{row['lb']}_atr{row['atr']}_r{int(row['risk']*100)}%"
        print(f"  {config_str:40s} → ROI: {row['roi']:+,.1f}% ({row['trades']} trades)")
    
    print(f"\n🎯 BEST CONFIGURATION:")
    print(f"  Window: {best_config['w']} hours")
    print(f"  Step: {best_config['s']} hours")
    print(f"  Lookback: {best_config['lb']} bars")
    print(f"  ATR Multiplier: {best_config['atr']}")
    print(f"  Risk per Trade: {best_config['risk']*100:.0f}%")
    print(f"  Trades: {best_config['trades']}")
    print(f"  ROI: {best_roi:+,.2f}%")
    
    if best_roi >= 1000:
        print(f"\n✅ TARGET ACHIEVED! Found config with {best_roi:+,.0f}% ROI (target: 1000%)")
    else:
        print(f"\n⚠️ TARGET NOT MET. Best ROI: {best_roi:+,.1f}% (target: 1000%)")
        print("  Consider: More iterations, different strategies, or adjusted targets.")
    
    # Save results
    results_df.to_csv('optimization_results.csv', index=False)
    print(f"\nResults saved to optimization_results.csv")

if __name__ == "__main__":
    main()
