"""
Generate trade logs for the BEST configuration from 5-period optimization.
Config: w200_s10_lb5_atr2.5_r30%_ml0.3
"""
import sys
import json
import time
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
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_sharpe(returns, step_hours=10):
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    windows_per_year = (365 * 24) / step_hours
    return (np.mean(returns) / np.std(returns)) * np.sqrt(windows_per_year)

def main():
    print("="*80)
    print("GENERATING TRADE LOGS FOR BEST CONFIG")
    print("Config: w200_s10_lb5_atr2.5_r30%_ml0.3")
    print("="*80)
    
    # Best config parameters from optimization
    w, s, lb, atr, risk, ml_conf = 200, 10, 5, 2.5, 0.30, 0.30
    
    exchange = ccxt.binance()
    symbol = "BTC/USDT"
    
    period_ms = 90 * 24 * 3600 * 1000
    gap_ms = 7 * 24 * 3600 * 1000
    end_time = exchange.milliseconds()
    
    all_period_logs = []
    
    for period_idx in range(5):
        start_time = end_time - period_ms
        print(f"\nPeriod {period_idx+1}: Fetching data...")
        
        data = fetch_period_data(exchange, symbol, start_time, end_time)
        print(f"  Got {len(data)} candles ({data.index[0]} to {data.index[-1]})")
        
        # Run strategy
        engine = BacktestEngine(cost_model=CostModel())
        config = EnhancedConfig(min_ml_confidence=ml_conf, forward_bars=8)
        
        capital = 10000
        start_capital = 10000
        start_idx = 0
        all_trades = []
        window_returns = []
        
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
                    
                    # Calculate position details
                    position_size = trade.size
                    notional = position_size * trade.entry_price
                    allocated_capital = capital_before * risk
                    leverage_used = notional / allocated_capital if allocated_capital > 0 else 0
                    
                    if trade.side == "long":
                        price_change_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    else:
                        price_change_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100
                    
                    trade_log = {
                        'window_start': str(seg.index[0]),
                        'window_end': str(seg.index[-1]),
                        'entry_time': str(trade.entry_time),
                        'exit_time': str(trade.exit_time),
                        'direction': trade.side.upper(),
                        'entry_price': float(trade.entry_price),
                        'exit_price': float(trade.exit_price),
                        'stop_loss': float(trade.stop_price),
                        'exit_reason': trade.exit_reason,
                        'position_size': float(position_size),
                        'notional': float(notional),
                        'allocated_capital': float(allocated_capital),
                        'leverage_used': float(leverage_used),
                        'price_change_pct': float(price_change_pct),
                        'pnl_dollars': float(pnl),
                        'pnl_pct': float(pnl_pct),
                        'capital_before': float(capital_before),
                        'capital_after': float(capital)
                    }
                    all_trades.append(trade_log)
                
                window_returns.append((capital - capital_before) / capital_before)
                    
            except Exception as e:
                pass
                
            start_idx += s
        
        final_roi = (capital/start_capital - 1)*100
        sharpe = calculate_sharpe(window_returns, s)
        wins = sum(1 for t in all_trades if t['pnl_dollars'] > 0)
        win_rate = (wins / len(all_trades) * 100) if all_trades else 0
        
        print(f"  Period {period_idx+1} Complete:")
        print(f"    ROI: {final_roi:+,.0f}% | Sharpe: {sharpe:.2f} | Trades: {len(all_trades)} | Win Rate: {win_rate:.0f}%")
        
        all_period_logs.append({
            'period': period_idx + 1,
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'start_capital': start_capital,
            'end_capital': capital,
            'roi_pct': final_roi,
            'sharpe': sharpe,
            'num_trades': len(all_trades),
            'win_rate': win_rate,
            'trades': all_trades
        })
        
        end_time = start_time - gap_ms
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_roi = np.mean([p['roi_pct'] for p in all_period_logs])
    min_roi = min(p['roi_pct'] for p in all_period_logs)
    avg_sharpe = np.mean([p['sharpe'] for p in all_period_logs])
    min_sharpe = min(p['sharpe'] for p in all_period_logs)
    total_trades = sum(p['num_trades'] for p in all_period_logs)
    
    print(f"Avg ROI: {avg_roi:+,.0f}% | Min ROI: {min_roi:+,.0f}%")
    print(f"Avg Sharpe: {avg_sharpe:.2f} | Min Sharpe: {min_sharpe:.2f}")
    print(f"Total Trades: {total_trades}")
    
    for p in all_period_logs:
        status = "✅" if p['sharpe'] >= 2.0 else "❌"
        print(f"  Period {p['period']}: ROI={p['roi_pct']:+,.0f}% Sharpe={p['sharpe']:.2f} {status}")
    
    # Save to JSON
    output_path = 'dashboard/data/trade_logs.json'
    import os
    os.makedirs('dashboard/data', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_period_logs, f, indent=2, default=str)
    
    print(f"\n✅ Trade log saved to {output_path}")

if __name__ == "__main__":
    main()
