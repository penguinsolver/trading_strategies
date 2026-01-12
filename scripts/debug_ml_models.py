"""
Debug all ML models to understand why they underperform.
"""
import sys
sys.path.insert(0, '.')

from data import CandleFetcher
from backtest import BacktestEngine, CostModel
from strategies import STRATEGIES
from models.ml_signal_generator import MLSignalGenerator, MLSignalConfig

# Fetch data
print("=" * 70)
print("VERIFYING ML MODEL NUMBERS")
print("=" * 70)

fetcher = CandleFetcher(coin="BTC", use_cache=True)
data = fetcher.fetch_data(interval="1h", window="90d")

print(f"Data: {len(data)} candles from {data.index[0].date()} to {data.index[-1].date()}")
print()

engine = BacktestEngine(cost_model=CostModel())
risk = 0.02  # 2% risk

# First, verify MA Crossover
print("=" * 70)
print("1. MA CROSSOVER (BASELINE)")
print("=" * 70)
ma_strategy = STRATEGIES["ma_crossover"]()
ma_result = engine.run(ma_strategy, data.copy(), 10000, risk)
print(f"ROI: {ma_result.metrics.net_return:+.2f}%")
print(f"Trades: {ma_result.metrics.total_trades}")
print()

# Test ML models
models_to_test = ["xgboost", "random_forest", "logistic", "gradient_boosting"]

for model_type in models_to_test:
    print("=" * 70)
    print(f"2. ML_{model_type.upper()}")
    print("=" * 70)
    
    try:
        config = MLSignalConfig(
            model_type=model_type,
            n_estimators=100,
            forward_bars=12,
            profit_threshold=0.005,
            min_probability=0.55,
        )
        
        ml = MLSignalGenerator(config)
        
        # Train on first 70%
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        
        metrics = ml.train(train_data)
        print(f"Training samples: {metrics.get('n_samples', 0)}")
        print(f"Validation accuracy: {metrics.get('val_accuracy_mean', 0):.1%}")
        print(f"Class distribution: {metrics.get('class_distribution', {})}")
        
        # Generate signals
        signals = ml.predict_signals(data.copy())
        
        # Count signals
        long_signals = (signals["entry_signal"] == 1).sum()
        short_signals = (signals["entry_signal"] == -1).sum()
        print(f"Long signals: {long_signals}, Short signals: {short_signals}")
        
        # Run backtest
        result = engine.run(ml, data.copy(), 10000, risk)
        print(f"ROI: {result.metrics.net_return:+.2f}%")
        print(f"Trades: {result.metrics.total_trades}")
        
        if result.trades:
            for t in result.trades[:3]:  # Show first 3 trades
                print(f"  {t.side.upper()} ${t.entry_price:.0f} -> ${t.exit_price:.0f}: ${t.pnl_net:+.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()

print("=" * 70)
print("DIAGNOSIS: Why ML models underperform")
print("=" * 70)
print("""
1. XGBoost/AdaBoost: 0 trades - min_probability=0.55 filters all signals
2. Logistic Regression: Only 3 trades with small positions
3. The ML models predict the 'neutral' class too often

Recommendations:
- Lower min_probability to 0.45
- Use lower profit_threshold (0.003 instead of 0.005)
- Or use ML to filter MA Crossover signals instead of replacing them
""")
