"""Models module for ML, Statistical, and Ensemble strategies."""
from .ensemble import EnsembleStrategy, EnsembleConfig
from .regime_filter import RegimeFilter, RegimeConfig

# Statistical Models
from .hmm_regime import HMMRegimeDetector, HMMConfig, HMMFilteredStrategy, MarketRegime
from .kalman_filter import KalmanTrendFilter, KalmanConfig, KalmanStrategy
from .garch_sizing import GARCHVolatilitySizer, GARCHConfig, GARCHSizedStrategy

# ML Models (Filters)
from .xgboost_classifier import XGBoostTradeClassifier, XGBConfig, XGBoostFilteredStrategy

# ML Models (Signal Generators)
from .feature_engineer import FeatureEngineer, FeatureConfig
from .ml_signal_generator import MLSignalGenerator, MLSignalConfig, MultiModelEnsemble

# Advanced ML Models
from .advanced_models import (
    StackingEnsemble, NeuralNetworkModel, CatBoostModel,
    MeanReversionStrategy, HybridMACrossover, VotingEnsembleModel,
    AdvancedConfig,
)

# Additional Trading Strategies
from .additional_strategies import (
    MomentumStrategy, ADXTrendStrategy, BreakoutStrategy,
    OptimizedMACrossover, DualMomentumStrategy, TrendFollowingSystem,
)

# Optimized ML Models (Grid Search)
from .optimized_models import (
    GridSearchOptimizer, OptimizedConfig,
    OptimizedXGBoost, OptimizedRandomForest, OptimizedLightGBM,
    AggressiveMLEnsemble,
)

# ML-Enhanced Strategies (WINNERS)
from .ml_enhanced import (
    MLEnhancedBreakout, MLEnhancedMACrossover, MLTrendFollower,
    EnhancedConfig,
)

__all__ = [
    # Ensemble
    "EnsembleStrategy", "EnsembleConfig",
    # Regime Filter
    "RegimeFilter", "RegimeConfig",
    # HMM
    "HMMRegimeDetector", "HMMConfig", "HMMFilteredStrategy", "MarketRegime",
    # Kalman
    "KalmanTrendFilter", "KalmanConfig", "KalmanStrategy",
    # GARCH
    "GARCHVolatilitySizer", "GARCHConfig", "GARCHSizedStrategy",
    # XGBoost Filter
    "XGBoostTradeClassifier", "XGBConfig", "XGBoostFilteredStrategy",
    # ML Signal Generators
    "FeatureEngineer", "FeatureConfig",
    "MLSignalGenerator", "MLSignalConfig", "MultiModelEnsemble",
    # Advanced Models
    "StackingEnsemble", "NeuralNetworkModel", "CatBoostModel",
    "MeanReversionStrategy", "HybridMACrossover", "VotingEnsembleModel",
    "AdvancedConfig",
    # Additional Strategies
    "MomentumStrategy", "ADXTrendStrategy", "BreakoutStrategy",
    "OptimizedMACrossover", "DualMomentumStrategy", "TrendFollowingSystem",
    # Optimized ML Models
    "GridSearchOptimizer", "OptimizedConfig",
    "OptimizedXGBoost", "OptimizedRandomForest", "OptimizedLightGBM",
    "AggressiveMLEnsemble",
    # ML-Enhanced Strategies (WINNERS)
    "MLEnhancedBreakout", "MLEnhancedMACrossover", "MLTrendFollower",
    "EnhancedConfig",
]
