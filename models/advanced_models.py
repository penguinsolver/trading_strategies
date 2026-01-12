"""
Advanced ML Models - Sophisticated approaches to beat MA Crossover.

Includes:
1. Stacking Ensemble - Meta-learner on multiple base models
2. Neural Network (MLP) - Deep learning approach
3. CatBoost - Advanced gradient boosting
4. Hybrid ML-Technical - Enhance MA Crossover with ML
5. Mean Reversion - Pure statistical approach
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import warnings

try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier,
        StackingClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from .feature_engineer import FeatureEngineer


@dataclass
class AdvancedConfig:
    """Configuration for advanced ML models."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    forward_bars: int = 12
    profit_threshold: float = 0.005
    min_probability: float = 0.50  # Lower threshold for more signals
    random_state: int = 42


class StackingEnsemble:
    """
    Stacking Ensemble with meta-learner.
    
    Uses Random Forest, Gradient Boosting, and Logistic Regression as base models,
    with a Logistic Regression meta-learner on top.
    """
    
    def __init__(self, config: Optional[AdvancedConfig] = None):
        if not HAS_SKLEARN:
            raise ImportError("sklearn required")
        
        self.config = config or AdvancedConfig()
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "STACKING_ENSEMBLE"
    
    def _create_model(self):
        """Create stacking classifier."""
        base_estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=self.config.random_state,
                n_jobs=-1,
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state,
            )),
            ('lr', LogisticRegression(
                max_iter=500,
                random_state=self.config.random_state,
            )),
        ]
        
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=500),
            cv=3,
            n_jobs=-1,
        )
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the stacking ensemble."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.config.forward_bars,
            profit_threshold=self.config.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        # Encode labels for sklearn
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Train
        self._model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_scaled, y_encoded)
        
        self._is_fitted = True
        
        # Get accuracy
        y_pred = self._model.predict(X_scaled)
        acc = accuracy_score(y_encoded, y_pred)
        
        return {"train_accuracy": acc, "n_samples": len(X)}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self.train(data.iloc[:train_size])
        
        data = data.copy()
        features = self.feature_engineer.calculate_features(data)
        
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        
        predictions = self._model.predict(X_scaled)
        probabilities = self._model.predict_proba(X_scaled)
        
        # Decode predictions back to -1, 0, 1
        predictions_decoded = self._label_encoder.inverse_transform(predictions)
        
        data["ml_signal"] = predictions_decoded
        data["ml_confidence"] = probabilities.max(axis=1)
        
        data["entry_signal"] = 0
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.config.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.config.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        return data


class NeuralNetworkModel:
    """
    Multi-Layer Perceptron Neural Network.
    
    Uses 3 hidden layers with ReLU activation for signal generation.
    """
    
    def __init__(self, config: Optional[AdvancedConfig] = None):
        if not HAS_SKLEARN:
            raise ImportError("sklearn required")
        
        self.config = config or AdvancedConfig()
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "NEURAL_NETWORK"
    
    def _create_model(self):
        """Create MLP classifier."""
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.config.random_state,
        )
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the neural network."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.config.forward_bars,
            profit_threshold=self.config.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_scaled, y_encoded)
        
        self._is_fitted = True
        
        y_pred = self._model.predict(X_scaled)
        acc = accuracy_score(y_encoded, y_pred)
        
        return {"train_accuracy": acc, "n_samples": len(X)}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self.train(data.iloc[:train_size])
        
        data = data.copy()
        features = self.feature_engineer.calculate_features(data)
        
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        
        predictions = self._model.predict(X_scaled)
        probabilities = self._model.predict_proba(X_scaled)
        
        predictions_decoded = self._label_encoder.inverse_transform(predictions)
        
        data["ml_signal"] = predictions_decoded
        data["ml_confidence"] = probabilities.max(axis=1)
        
        data["entry_signal"] = 0
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.config.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.config.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        return data


class CatBoostModel:
    """
    CatBoost Gradient Boosting.
    
    Often performs better than XGBoost on tabular data with categorical features.
    """
    
    def __init__(self, config: Optional[AdvancedConfig] = None):
        if not HAS_CATBOOST:
            raise ImportError("catboost required. pip install catboost")
        
        self.config = config or AdvancedConfig()
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "CATBOOST"
    
    def _create_model(self):
        """Create CatBoost classifier."""
        return CatBoostClassifier(
            iterations=self.config.n_estimators,
            depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.random_state,
            verbose=False,
            thread_count=-1,
        )
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train CatBoost model."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.config.forward_bars,
            profit_threshold=self.config.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = self._create_model()
        self._model.fit(X_scaled, y_encoded)
        
        self._is_fitted = True
        
        y_pred = self._model.predict(X_scaled)
        acc = accuracy_score(y_encoded, y_pred)
        
        return {"train_accuracy": acc, "n_samples": len(X)}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self.train(data.iloc[:train_size])
        
        data = data.copy()
        features = self.feature_engineer.calculate_features(data)
        
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        
        predictions = self._model.predict(X_scaled)
        probabilities = self._model.predict_proba(X_scaled)
        
        # CatBoost returns predictions as 2D array sometimes
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            predictions = predictions.flatten()
        
        predictions_decoded = self._label_encoder.inverse_transform(predictions.astype(int))
        
        data["ml_signal"] = predictions_decoded
        data["ml_confidence"] = probabilities.max(axis=1)
        
        data["entry_signal"] = 0
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.config.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.config.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        return data


class MeanReversionStrategy:
    """
    Mean Reversion Z-Score Strategy.
    
    Pure statistical approach: Go long when z-score is very negative (oversold),
    go short when z-score is very positive (overbought).
    """
    
    def __init__(
        self,
        lookback: int = 20,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        use_bollinger: bool = True,
    ):
        self.lookback = lookback
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.use_bollinger = use_bollinger
    
    @property
    def name(self) -> str:
        return "MEAN_REVERSION"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals based on z-score."""
        data = data.copy()
        close = data["close"]
        
        # Calculate z-score
        sma = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        zscore = (close - sma) / (std + 1e-10)
        
        data["zscore"] = zscore
        data["entry_signal"] = 0
        
        # Oversold: go long (expecting mean reversion up)
        data.loc[zscore < -self.zscore_entry, "entry_signal"] = 1
        
        # Overbought: go short (expecting mean reversion down)
        data.loc[zscore > self.zscore_entry, "entry_signal"] = -1
        
        # Exit when z-score returns to mean
        data["exit_signal"] = np.abs(zscore) < self.zscore_exit
        
        # Add stop based on ATR
        atr = (data["high"] - data["low"]).rolling(14).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            data["close"] - 2 * atr,
            np.where(
                data["entry_signal"] == -1,
                data["close"] + 2 * atr,
                np.nan
            )
        )
        
        return data


class HybridMACrossover:
    """
    Hybrid Strategy: MA Crossover enhanced with ML confidence filtering.
    
    Uses MA Crossover for signal generation, but only takes trades 
    when ML model agrees with the direction.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        min_ml_confidence: float = 0.45,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_ml_confidence = min_ml_confidence
        
        self.feature_engineer = FeatureEngineer()
        self._ml_model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
    
    @property
    def name(self) -> str:
        return "HYBRID_MA_ML"
    
    def _train_ml(self, data: pd.DataFrame):
        """Train the ML enhancement model."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data, forward_bars=12, profit_threshold=0.003
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Use simple RF for speed
        self._ml_model = RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42, n_jobs=-1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ml_model.fit(X_scaled, y_encoded)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate hybrid MA + ML signals."""
        data = data.copy()
        close = data["close"]
        
        # 1. Calculate MA Crossover signals
        fast_ma = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ma = close.ewm(span=self.slow_period, adjust=False).mean()
        
        ma_cross = fast_ma > slow_ma
        ma_cross_shift = ma_cross.shift(1).fillna(False).astype(bool)
        
        data["ma_long_signal"] = ma_cross & ~ma_cross_shift  # Golden cross
        data["ma_short_signal"] = ~ma_cross & ma_cross_shift.astype(bool)  # Death cross
        
        # 2. Train ML model if needed
        if self._ml_model is None:
            train_size = int(len(data) * 0.7)
            self._train_ml(data.iloc[:train_size])
        
        # 3. Get ML predictions
        features = self.feature_engineer.calculate_features(data)
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        ml_preds = self._ml_model.predict(X_scaled)
        ml_probs = self._ml_model.predict_proba(X_scaled)
        
        ml_decoded = self._label_encoder.inverse_transform(ml_preds)
        data["ml_pred"] = ml_decoded
        data["ml_confidence"] = ml_probs.max(axis=1)
        
        # 4. Combine: MA signal + ML agreement
        data["entry_signal"] = 0
        
        # Long: MA long signal AND (ML agrees OR is neutral with high confidence)
        long_mask = (
            data["ma_long_signal"] & 
            ((data["ml_pred"] == 1) | 
             ((data["ml_pred"] == 0) & (data["ml_confidence"] >= self.min_ml_confidence)))
        )
        data.loc[long_mask, "entry_signal"] = 1
        
        # Short: MA short signal AND (ML agrees OR is neutral with high confidence)
        short_mask = (
            data["ma_short_signal"] & 
            ((data["ml_pred"] == -1) | 
             ((data["ml_pred"] == 0) & (data["ml_confidence"] >= self.min_ml_confidence)))
        )
        data.loc[short_mask, "entry_signal"] = -1
        
        # 5. Add ATR-based stops
        atr = (data["high"] - data["low"]).rolling(self.atr_period).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - self.atr_multiplier * atr,
            np.where(
                data["entry_signal"] == -1,
                close + self.atr_multiplier * atr,
                np.nan
            )
        )
        
        return data


class VotingEnsembleModel:
    """
    Soft Voting Ensemble.
    
    Combines predictions from multiple models using weighted probability averaging.
    """
    
    def __init__(self, config: Optional[AdvancedConfig] = None):
        if not HAS_SKLEARN:
            raise ImportError("sklearn required")
        
        self.config = config or AdvancedConfig()
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "VOTING_ENSEMBLE"
    
    def _create_model(self):
        """Create voting classifier."""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
        ]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1,
        )
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the voting ensemble."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.config.forward_bars,
            profit_threshold=self.config.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_scaled, y_encoded)
        
        self._is_fitted = True
        
        y_pred = self._model.predict(X_scaled)
        acc = accuracy_score(y_encoded, y_pred)
        
        return {"train_accuracy": acc, "n_samples": len(X)}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self.train(data.iloc[:train_size])
        
        data = data.copy()
        features = self.feature_engineer.calculate_features(data)
        
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        
        predictions = self._model.predict(X_scaled)
        probabilities = self._model.predict_proba(X_scaled)
        
        predictions_decoded = self._label_encoder.inverse_transform(predictions)
        
        data["ml_signal"] = predictions_decoded
        data["ml_confidence"] = probabilities.max(axis=1)
        
        data["entry_signal"] = 0
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.config.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.config.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        return data
