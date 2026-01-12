"""
Test Advanced ML Models to Beat MA Crossover.

Tests: Stacking, Neural Network, CatBoost, Mean Reversion, Hybrid, Voting Ensemble
Target: Beat MA Crossover's 11.58% ROI (at 1% risk)
"""
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings("ignore")

from data import CandleFetcher
from backtest import BacktestEngine, CostModel
from strategies import STRATEGIES

# Advanced models
from models.advanced_models import (
    StackingEnsemble, NeuralNetworkModel, MeanReversionStrategy,
    HybridMACrossover, VotingEnsembleModel, AdvancedConfig,
)

try:
    from models.advanced_models import CatBoostModel
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False


def test_model(data, model, risk, name=None):
    """Test a model and return results."""
    engine = BacktestEngine(cost_model=CostModel())
    model_name = name or getattr(model, 'name', str(type(model).__name__))
    
    try:
        result = engine.run(model, data.copy(), 10000, risk)
        return {
            "name": model_name,
            "roi": result.metrics.net_return,
            "trades": result.metrics.total_trades,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "success": True,
        }
    except Exception as e:
        return {
            "name": model_name,
            "roi": 0,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "success": False,
            "error": str(e)[:50],
        }


def main():
    print("=" * 70)
    print("🚀 ADVANCED ML MODELS TEST - GOAL: BEAT MA CROSSOVER")
    print("=" * 70)
    
    # Fetch data
    print("\n📡 Fetching 90d/1h BTC data...")
    fetcher = CandleFetcher(coin="BTC", use_cache=True)
    data = fetcher.fetch_data(interval="1h", window="90d")
    print(f"✅ {len(data)} candles from {data.index[0].date()} to {data.index[-1].date()}")
    
    risk = 0.01  # 1% risk to match dashboard
    results = []
    
    # 1. Baseline: MA Crossover
    print("\n" + "=" * 70)
    print("1. BASELINE: MA CROSSOVER")
    print("=" * 70)
    ma = STRATEGIES["ma_crossover"]()
    r = test_model(data, ma, risk, "MA Crossover")
    results.append(r)
    baseline_roi = r["roi"]
    print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | Win Rate: {r['win_rate']:.1f}%")
    
    # 2. Stacking Ensemble
    print("\n" + "=" * 70)
    print("2. STACKING ENSEMBLE (RF + GB + LR meta-learner)")
    print("=" * 70)
    try:
        stacking = StackingEnsemble(AdvancedConfig(min_probability=0.45))
        r = test_model(data, stacking, risk)
        results.append(r)
        beat = "✅ BEATS BASELINE!" if r["roi"] > baseline_roi else "❌"
        print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | {beat}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 3. Neural Network
    print("\n" + "=" * 70)
    print("3. NEURAL NETWORK (MLP 128-64-32)")
    print("=" * 70)
    try:
        nn = NeuralNetworkModel(AdvancedConfig(min_probability=0.45))
        r = test_model(data, nn, risk)
        results.append(r)
        beat = "✅ BEATS BASELINE!" if r["roi"] > baseline_roi else "❌"
        print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | {beat}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 4. CatBoost
    if HAS_CATBOOST:
        print("\n" + "=" * 70)
        print("4. CATBOOST")
        print("=" * 70)
        try:
            cat = CatBoostModel(AdvancedConfig(min_probability=0.45))
            r = test_model(data, cat, risk)
            results.append(r)
            beat = "✅ BEATS BASELINE!" if r["roi"] > baseline_roi else "❌"
            print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | {beat}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print("\n⚠️  CatBoost not installed. Run: pip install catboost")
    
    # 5. Mean Reversion
    print("\n" + "=" * 70)
    print("5. MEAN REVERSION (Z-Score Strategy)")
    print("=" * 70)
    try:
        mr = MeanReversionStrategy(lookback=20, zscore_entry=2.0)
        r = test_model(data, mr, risk)
        results.append(r)
        beat = "✅ BEATS BASELINE!" if r["roi"] > baseline_roi else "❌"
        print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | {beat}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 6. Hybrid MA + ML
    print("\n" + "=" * 70)
    print("6. HYBRID: MA CROSSOVER + ML FILTER")
    print("=" * 70)
    try:
        hybrid = HybridMACrossover(min_ml_confidence=0.40)
        r = test_model(data, hybrid, risk)
        results.append(r)
        beat = "✅ BEATS BASELINE!" if r["roi"] > baseline_roi else "❌"
        print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | {beat}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 7. Voting Ensemble
    print("\n" + "=" * 70)
    print("7. VOTING ENSEMBLE (RF + GB + MLP)")
    print("=" * 70)
    try:
        voting = VotingEnsembleModel(AdvancedConfig(min_probability=0.45))
        r = test_model(data, voting, risk)
        results.append(r)
        beat = "✅ BEATS BASELINE!" if r["roi"] > baseline_roi else "❌"
        print(f"   ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | {beat}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - SORTED BY ROI")
    print("=" * 70)
    
    results.sort(key=lambda x: x["roi"], reverse=True)
    
    print(f"\n{'Model':<30} {'ROI':>10} {'Trades':>8} {'Win Rate':>10} {'Beat MA?':>10}")
    print("-" * 70)
    
    for r in results:
        beat = "✅ YES" if r["roi"] > baseline_roi and r["name"] != "MA Crossover" else ("BASELINE" if r["name"] == "MA Crossover" else "❌ NO")
        print(f"{r['name']:<30} {r['roi']:>+9.2f}% {r['trades']:>8} {r['win_rate']:>9.1f}% {beat:>10}")
    
    # Winners
    winners = [r for r in results if r["roi"] > baseline_roi and r["name"] != "MA Crossover"]
    if winners:
        print(f"\n🎉 {len(winners)} MODEL(S) BEAT MA CROSSOVER!")
        for w in winners:
            diff = w["roi"] - baseline_roi
            print(f"   ✅ {w['name']}: +{w['roi']:.2f}% (beats by {diff:.2f}%)")
    else:
        print(f"\n⚠️  No models beat MA Crossover's {baseline_roi:.2f}% ROI")
        print("   Consider: different parameters, longer data, or 4h candles")


if __name__ == "__main__":
    main()
