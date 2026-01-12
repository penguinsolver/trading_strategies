"""
Comprehensive ML Model Comparison Test

This script tests all ML models against MA Crossover's 11.71% ROI benchmark.
Run from project root: python scripts/test_all_models.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

from data import CandleFetcher
from backtest import BacktestEngine, CostModel
from strategies import STRATEGIES
from models.ml_signal_generator import MLSignalGenerator, MLSignalConfig, MultiModelEnsemble

# Benchmark
MA_CROSSOVER_BENCHMARK = 11.71


def run_model_test(data: pd.DataFrame, model_type: str, capital: float, risk: float):
    """Run a single model test and return results."""
    engine = BacktestEngine(cost_model=CostModel())
    
    # Split data for training
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    
    try:
        config = MLSignalConfig(
            model_type=model_type,
            n_estimators=100,  # Faster for testing
            forward_bars=12,
            profit_threshold=0.005,
            min_probability=0.55,
        )
        
        model = MLSignalGenerator(config)
        metrics = model.train(train_data)
        
        result = engine.run(model, data.copy(), capital, risk)
        
        return {
            "success": True,
            "net_return": result.metrics.net_return,
            "total_trades": result.metrics.total_trades,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "max_drawdown": result.metrics.max_drawdown,
            "val_accuracy": metrics.get("val_accuracy_mean", 0) * 100,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "net_return": 0,
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "val_accuracy": 0,
        }


def run_strategy_test(data: pd.DataFrame, strategy_key: str, capital: float, risk: float):
    """Run a traditional strategy test."""
    engine = BacktestEngine(cost_model=CostModel())
    
    try:
        strategy = STRATEGIES[strategy_key]()
        result = engine.run(strategy, data.copy(), capital, risk)
        
        return {
            "success": True,
            "net_return": result.metrics.net_return,
            "total_trades": result.metrics.total_trades,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "max_drawdown": result.metrics.max_drawdown,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "net_return": 0,
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
        }


def main():
    print("=" * 70)
    print("🧪 COMPREHENSIVE ML MODEL COMPARISON TEST")
    print("=" * 70)
    print(f"📊 Benchmark: MA Crossover = {MA_CROSSOVER_BENCHMARK}% ROI\n")
    
    # Fetch data
    print("📡 Fetching 90d/1h BTC data...")
    fetcher = CandleFetcher(coin="BTC", use_cache=True)
    data = fetcher.fetch_data(interval="1h", window="90d")
    
    if data.empty:
        print("❌ Failed to fetch data!")
        return
    
    print(f"✅ Fetched {len(data)} candles\n")
    
    capital = 10000
    risk = 0.02  # 2% risk per trade (as fraction, not percentage)
    
    results = []
    
    # 1. Test MA Crossover (benchmark)
    print("🔄 Testing MA Crossover (benchmark)...")
    ma_result = run_strategy_test(data, "ma_crossover", capital, risk)
    results.append({
        "Model": "MA Crossover",
        "Type": "Technical",
        "ROI %": ma_result["net_return"],
        "Trades": ma_result["total_trades"],
        "Win Rate %": ma_result["win_rate"],
        "Profit Factor": ma_result["profit_factor"],
        "Beat Benchmark": "BASELINE",
    })
    print(f"   ROI: {ma_result['net_return']:.2f}%\n")
    
    # 2. Test ML Models
    ml_models = [
        ("xgboost", "XGBoost"),
        ("random_forest", "Random Forest"),
        ("logistic", "Logistic Regression"),
        ("gradient_boosting", "Gradient Boosting"),
        ("adaboost", "AdaBoost"),
        ("knn", "K-Nearest Neighbors"),
    ]
    
    for model_key, model_name in ml_models:
        print(f"🔄 Testing {model_name}...")
        result = run_model_test(data, model_key, capital, risk)
        
        beat = "✅ YES" if result["net_return"] > MA_CROSSOVER_BENCHMARK else "❌ NO"
        
        results.append({
            "Model": f"ML_{model_key.upper()}",
            "Type": "ML",
            "ROI %": result["net_return"],
            "Trades": result["total_trades"],
            "Win Rate %": result["win_rate"],
            "Profit Factor": result["profit_factor"],
            "Beat Benchmark": beat,
        })
        print(f"   ROI: {result['net_return']:.2f}% ({beat})\n")
    
    # 3. Test Multi-Model Ensemble
    print("🔄 Testing Multi-Model Ensemble (XGBoost + RF)...")
    try:
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        
        ensemble = MultiModelEnsemble(
            model_types=["xgboost", "random_forest"],
            min_agreement=2,
        )
        ensemble.train(train_data)
        
        engine = BacktestEngine(cost_model=CostModel())
        ensemble_result = engine.run(ensemble, data.copy(), capital, risk)
        
        beat = "✅ YES" if ensemble_result.metrics.net_return > MA_CROSSOVER_BENCHMARK else "❌ NO"
        
        results.append({
            "Model": "ML_ENSEMBLE",
            "Type": "ML Ensemble",
            "ROI %": ensemble_result.metrics.net_return,
            "Trades": ensemble_result.metrics.total_trades,
            "Win Rate %": ensemble_result.metrics.win_rate,
            "Profit Factor": ensemble_result.metrics.profit_factor,
            "Beat Benchmark": beat,
        })
        print(f"   ROI: {ensemble_result.metrics.net_return:.2f}% ({beat})\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # 4. Test top traditional strategies for comparison
    top_strategies = ["trend_pullback", "obv_divergence", "chandelier_trend"]
    for strat_key in top_strategies:
        print(f"🔄 Testing {strat_key}...")
        result = run_strategy_test(data, strat_key, capital, risk)
        
        beat = "✅ YES" if result["net_return"] > MA_CROSSOVER_BENCHMARK else "❌ NO"
        
        results.append({
            "Model": STRATEGIES[strat_key]().name,
            "Type": "Technical",
            "ROI %": result["net_return"],
            "Trades": result["total_trades"],
            "Win Rate %": result["win_rate"],
            "Profit Factor": result["profit_factor"],
            "Beat Benchmark": beat,
        })
        print(f"   ROI: {result['net_return']:.2f}% ({beat})\n")
    
    # Print summary table
    df = pd.DataFrame(results)
    df = df.sort_values("ROI %", ascending=False)
    
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"\n🎯 Target: Beat MA Crossover benchmark of {MA_CROSSOVER_BENCHMARK}%\n")
    
    # Format for display
    display_df = df.copy()
    display_df["ROI %"] = display_df["ROI %"].apply(lambda x: f"{x:+.2f}%")
    display_df["Win Rate %"] = display_df["Win Rate %"].apply(lambda x: f"{x:.1f}%")
    display_df["Profit Factor"] = display_df["Profit Factor"].apply(lambda x: f"{x:.2f}")
    
    print(tabulate(display_df, headers="keys", tablefmt="pretty", showindex=False))
    
    # Winners summary
    winners = df[df["ROI %"] > MA_CROSSOVER_BENCHMARK]
    if len(winners) > 0:
        print(f"\n🎉 {len(winners)} models beat the MA Crossover benchmark!")
        for _, row in winners.iterrows():
            diff = row["ROI %"] - MA_CROSSOVER_BENCHMARK
            print(f"   ✅ {row['Model']}: +{row['ROI %']:.2f}% (beats by {diff:.2f}%)")
    else:
        print(f"\n⚠️ No models beat the MA Crossover benchmark of {MA_CROSSOVER_BENCHMARK}%")
    
    # Save results
    output_file = Path(__file__).parent.parent / "model_comparison_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
