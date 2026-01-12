"""
ML Signal Generator - ML model that generates trading signals directly.

Unlike the XGBoost filter which just filters existing signals, this model
generates entry/exit signals based on predicted profitability from features.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    HAS_ML = True
except ImportError:
    HAS_ML = False
    warnings.warn("xgboost or sklearn not installed. ML features unavailable.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .feature_engineer import FeatureEngineer, FeatureConfig


@dataclass
class MLSignalConfig:
    """Configuration for ML Signal Generator."""
    
    # Model selection: xgboost, lightgbm, random_forest, logistic, svm, knn, adaboost, gradient_boosting
    model_type: str = "xgboost"
    
    # XGBoost parameters
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training parameters
    train_size_days: int = 60
    test_size_days: int = 30
    walk_forward_steps: int = 3  # Number of walk-forward windows
    
    # Label parameters
    forward_bars: int = 12  # Look ahead for profitability
    profit_threshold: float = 0.005  # 0.5% for long
    loss_threshold: float = -0.005  # -0.5% for short
    
    # Signal generation
    min_probability: float = 0.55  # Minimum prob to generate signal
    signal_cooldown: int = 3  # Minimum bars between signals
    
    # Position sizing based on confidence
    use_confidence_sizing: bool = True
    confidence_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.7, 1.0),   # High confidence: full size
        (0.6, 0.75),  # Medium-high: 75%
        (0.55, 0.5),  # Medium: 50%
    ])
    
    random_state: int = 42


class MLSignalGenerator:
    """
    ML model that generates trading signals directly from features.
    
    This is a complete trading strategy, not just a filter.
    Uses walk-forward training to prevent overfitting.
    
    Supported models:
    - xgboost: XGBoost Classifier
    - lightgbm: LightGBM Classifier
    - random_forest: Random Forest
    - logistic: Logistic Regression
    - svm: Support Vector Classifier
    - knn: K-Nearest Neighbors
    - adaboost: AdaBoost
    - gradient_boosting: Scikit-learn Gradient Boosting
    """
    
    def __init__(self, config: Optional[MLSignalConfig] = None):
        if not HAS_ML:
            raise ImportError("xgboost and sklearn required. Install with: pip install xgboost scikit-learn")
        
        self.config = config or MLSignalConfig()
        self.feature_engineer = FeatureEngineer()
        
        self._model = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        
        self._training_metrics: Dict = {}
        self._feature_importance: Dict[str, float] = {}
    
    @property
    def name(self) -> str:
        return f"ML_{self.config.model_type.upper()}"
    
    @property
    def description(self) -> str:
        return f"ML-based signal generator using {self.config.model_type}"
    
    def _create_model(self):
        """Create the ML model based on config."""
        model_type = self.config.model_type.lower()
        
        if model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
            )
        elif model_type == "lightgbm" and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                verbose=-1,
            )
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_child_weight,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif model_type == "svm":
            return SVC(
                probability=True,
                random_state=self.config.random_state,
                kernel='rbf',
            )
        elif model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1,
            )
        elif model_type == "adaboost":
            return AdaBoostClassifier(
                n_estimators=min(self.config.n_estimators, 100),
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=min(self.config.n_estimators, 100),
                max_depth=min(self.config.max_depth, 4),
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )
        else:
            # Default to XGBoost
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
            )
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels from data."""
        # Calculate features
        features = self.feature_engineer.calculate_features(data)
        
        # Calculate labels
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.config.forward_bars,
            profit_threshold=self.config.profit_threshold,
            loss_threshold=self.config.loss_threshold,
        )
        
        # Align and drop NaN
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        return X, y
    
    def train(self, data: pd.DataFrame) -> Dict:
        """
        Train the ML model using walk-forward validation.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        X, y = self._prepare_data(data)
        
        if len(X) < 200:
            raise ValueError("Need at least 200 samples for training")
        
        self._feature_names = list(X.columns)
        
        # Time series cross-validation
        n_splits = min(self.config.walk_forward_steps, len(X) // 100)
        tscv = TimeSeriesSplit(n_splits=max(2, n_splits))
        
        all_train_metrics = []
        all_val_metrics = []
        
        # Walk-forward training
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = self._create_model()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            all_train_metrics.append(train_acc)
            all_val_metrics.append(val_acc)
        
        # Final training on all data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_scaled, y)
        
        self._is_fitted = True
        
        # Calculate feature importance
        if hasattr(self._model, 'feature_importances_'):
            importance = self._model.feature_importances_
            self._feature_importance = dict(zip(self._feature_names, importance))
        
        # Store metrics
        self._training_metrics = {
            "train_accuracy_mean": np.mean(all_train_metrics),
            "train_accuracy_std": np.std(all_train_metrics),
            "val_accuracy_mean": np.mean(all_val_metrics),
            "val_accuracy_std": np.std(all_val_metrics),
            "n_features": len(self._feature_names),
            "n_samples": len(X),
            "class_distribution": dict(y.value_counts()),
        }
        
        return self._training_metrics
    
    def predict_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added signal columns:
            - ml_signal: Raw signal (-1, 0, 1)
            - ml_prob_long: Probability of long
            - ml_prob_short: Probability of short
            - ml_confidence: Max probability for direction
            - entry_signal: Final signal after filtering
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first. Call train() first.")
        
        data = data.copy()
        
        # Calculate features
        features = self.feature_engineer.calculate_features(data)
        
        # Ensure columns match training
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names]
        
        # Handle NaN
        features_filled = features.fillna(0)
        
        # Scale
        features_scaled = self._scaler.transform(features_filled)
        
        # Predict
        predictions = self._model.predict(features_scaled)
        probabilities = self._model.predict_proba(features_scaled)
        
        # Map predictions to signals
        data["ml_signal"] = predictions
        
        # Get class probabilities
        classes = self._model.classes_
        
        # Initialize probability columns
        data["ml_prob_short"] = 0.0
        data["ml_prob_neutral"] = 0.0
        data["ml_prob_long"] = 0.0
        
        for i, cls in enumerate(classes):
            if cls == -1:
                data["ml_prob_short"] = probabilities[:, i]
            elif cls == 0:
                data["ml_prob_neutral"] = probabilities[:, i]
            elif cls == 1:
                data["ml_prob_long"] = probabilities[:, i]
        
        # Calculate confidence
        data["ml_confidence"] = probabilities.max(axis=1)
        
        # Apply signal filtering
        data["entry_signal"] = 0
        
        # Long signals: high confidence long prediction
        long_mask = (
            (data["ml_signal"] == 1) & 
            (data["ml_prob_long"] >= self.config.min_probability)
        )
        data.loc[long_mask, "entry_signal"] = 1
        
        # Short signals: high confidence short prediction
        short_mask = (
            (data["ml_signal"] == -1) & 
            (data["ml_prob_short"] >= self.config.min_probability)
        )
        data.loc[short_mask, "entry_signal"] = -1
        
        # Apply cooldown between signals
        if self.config.signal_cooldown > 0:
            last_signal_idx = -self.config.signal_cooldown
            for i in range(len(data)):
                if data.iloc[i]["entry_signal"] != 0:
                    if i - last_signal_idx < self.config.signal_cooldown:
                        data.iloc[i, data.columns.get_loc("entry_signal")] = 0
                    else:
                        last_signal_idx = i
        
        # Calculate position sizing if enabled
        if self.config.use_confidence_sizing:
            data["position_scale"] = 0.5  # Default to 50%
            
            for prob_threshold, scale in self.config.confidence_tiers:
                mask = data["ml_confidence"] >= prob_threshold
                data.loc[mask, "position_scale"] = scale
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standard strategy interface: generate signals from data.
        
        If not trained, will auto-train on the provided data.
        Note: For best results, train separately on historical data.
        """
        if not self._is_fitted:
            # Auto-train on first 70% of data
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            self.train(train_data)
        
        return self.predict_signals(data)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if not self._feature_importance:
            return {}
        
        sorted_importance = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_importance[:top_n])
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "training_metrics": self._training_metrics,
            "config": self.config,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> "MLSignalGenerator":
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self._model = state["model"]
        self._scaler = state["scaler"]
        self._feature_names = state["feature_names"]
        self._feature_importance = state["feature_importance"]
        self._training_metrics = state["training_metrics"]
        self.config = state["config"]
        self._is_fitted = True
        
        return self


class MultiModelEnsemble:
    """
    Ensemble of multiple ML models that vote on signals.
    
    Only generates a signal when multiple models agree.
    """
    
    def __init__(
        self,
        model_types: List[str] = None,
        min_agreement: int = 2,
        config: Optional[MLSignalConfig] = None,
    ):
        if model_types is None:
            model_types = ["xgboost", "random_forest"]
            if HAS_LIGHTGBM:
                model_types.append("lightgbm")
        
        self.model_types = model_types
        self.min_agreement = min_agreement
        self.base_config = config or MLSignalConfig()
        
        self._models: List[MLSignalGenerator] = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return f"ML_Ensemble_{len(self.model_types)}Models"
    
    @property
    def description(self) -> str:
        return f"Ensemble of {', '.join(self.model_types)} with {self.min_agreement}+ agreement"
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train all models in the ensemble."""
        self._models = []
        all_metrics = {}
        
        for model_type in self.model_types:
            config = MLSignalConfig(
                model_type=model_type,
                n_estimators=self.base_config.n_estimators,
                max_depth=self.base_config.max_depth,
                learning_rate=self.base_config.learning_rate,
                forward_bars=self.base_config.forward_bars,
                profit_threshold=self.base_config.profit_threshold,
                min_probability=self.base_config.min_probability,
            )
            
            model = MLSignalGenerator(config)
            metrics = model.train(data)
            
            self._models.append(model)
            all_metrics[model_type] = metrics
        
        self._is_fitted = True
        return all_metrics
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals with ensemble voting.
        
        Only signals when min_agreement models agree.
        """
        if not self._is_fitted:
            # Auto-train
            train_size = int(len(data) * 0.7)
            self.train(data.iloc[:train_size])
        
        data = data.copy()
        
        # Get signals from each model
        all_signals = pd.DataFrame(index=data.index)
        
        for i, model in enumerate(self._models):
            model_data = model.predict_signals(data.copy())
            all_signals[f"model_{i}"] = model_data["entry_signal"]
        
        # Count votes
        long_votes = (all_signals == 1).sum(axis=1)
        short_votes = (all_signals == -1).sum(axis=1)
        
        # Generate ensemble signal
        data["ensemble_long_votes"] = long_votes
        data["ensemble_short_votes"] = short_votes
        
        data["entry_signal"] = 0
        data.loc[long_votes >= self.min_agreement, "entry_signal"] = 1
        data.loc[short_votes >= self.min_agreement, "entry_signal"] = -1
        
        # If both have enough votes, go with the majority
        both_signal = (long_votes >= self.min_agreement) & (short_votes >= self.min_agreement)
        data.loc[both_signal & (long_votes > short_votes), "entry_signal"] = 1
        data.loc[both_signal & (short_votes > long_votes), "entry_signal"] = -1
        data.loc[both_signal & (short_votes == long_votes), "entry_signal"] = 0
        
        return data
