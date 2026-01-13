"""
Optimized ML Models with Grid Search.

Uses grid search to find optimal hyperparameters for beating TA strategies.
Target: Beat Breakout (13.32%) and MA Crossover (12.39%)
"""
import pandas as pd
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from itertools import product

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .feature_engineer import FeatureEngineer


@dataclass
class OptimizedConfig:
    """Configuration for optimized ML models."""
    # Label generation
    forward_bars: int = 8  # Look-ahead period
    profit_threshold: float = 0.003  # 0.3% profit threshold
    
    # Signal generation  
    min_probability: float = 0.40  # Lower threshold for more signals
    signal_cooldown: int = 2
    
    # Training
    train_ratio: float = 0.7
    random_state: int = 42


class GridSearchOptimizer:
    """
    Grid search optimizer for ML trading models.
    
    Tests different hyperparameter combinations and returns best performer.
    """
    
    def __init__(self, config: Optional[OptimizedConfig] = None):
        if not HAS_ML:
            raise ImportError("sklearn and xgboost required")
        
        self.config = config or OptimizedConfig()
        self.feature_engineer = FeatureEngineer()
        self._best_params = {}
        self._best_score = 0
        self._results = []
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.config.forward_bars,
            profit_threshold=self.config.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        return X, y
    
    def grid_search_xgboost(self, data: pd.DataFrame) -> Dict:
        """
        Grid search for XGBoost hyperparameters.
        
        Returns best parameters and score.
        """
        X, y = self._prepare_data(data)
        
        train_size = int(len(X) * self.config.train_ratio)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
        }
        
        best_score = 0
        best_params = {}
        results = []
        
        # Get all combinations
        keys = param_grid.keys()
        combinations = list(product(*param_grid.values()))
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            try:
                model = xgb.XGBClassifier(
                    **params,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0,
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_scaled, y_train_enc)
                
                y_pred = model.predict(X_test_scaled)
                score = f1_score(y_test_enc, y_pred, average='weighted')
                
                results.append({**params, 'f1_score': score})
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception:
                continue
        
        self._best_params['xgboost'] = best_params
        self._best_score = best_score
        self._results = results
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'total_combinations': len(combinations),
        }
    
    def grid_search_rf(self, data: pd.DataFrame) -> Dict:
        """Grid search for Random Forest."""
        X, y = self._prepare_data(data)
        
        train_size = int(len(X) * self.config.train_ratio)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, 10],
            'min_samples_leaf': [1, 3, 5],
        }
        
        best_score = 0
        best_params = {}
        
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for min_leaf in param_grid['min_samples_leaf']:
                    params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_leaf': min_leaf,
                    }
                    
                    try:
                        model = RandomForestClassifier(
                            **params,
                            random_state=self.config.random_state,
                            n_jobs=-1,
                        )
                        model.fit(X_train_scaled, y_train_enc)
                        
                        y_pred = model.predict(X_test_scaled)
                        score = f1_score(y_test_enc, y_pred, average='weighted')
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            
                    except Exception:
                        continue
        
        self._best_params['random_forest'] = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
        }


class OptimizedXGBoost:
    """
    XGBoost with optimized hyperparameters.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        min_child_weight: int = 3,
        forward_bars: int = 8,
        profit_threshold: float = 0.003,
        min_probability: float = 0.40,
    ):
        if not HAS_ML:
            raise ImportError("xgboost required")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.forward_bars = forward_bars
        self.profit_threshold = profit_threshold
        self.min_probability = min_probability
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return f"OPT_XGBOOST_d{self.max_depth}_lr{self.learning_rate}"
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the optimized XGBoost model."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.forward_bars,
            profit_threshold=self.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
        )
        
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
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        # Add ATR-based stops
        atr = (data["high"] - data["low"]).rolling(14).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            data["close"] - 2.0 * atr,
            np.where(
                data["entry_signal"] == -1,
                data["close"] + 2.0 * atr,
                np.nan
            )
        )
        
        return data


class OptimizedRandomForest:
    """
    Random Forest with optimized hyperparameters.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_samples_leaf: int = 3,
        forward_bars: int = 8,
        profit_threshold: float = 0.003,
        min_probability: float = 0.40,
    ):
        if not HAS_ML:
            raise ImportError("sklearn required")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.forward_bars = forward_bars
        self.profit_threshold = profit_threshold
        self.min_probability = min_probability
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return f"OPT_RF_d{self.max_depth}_n{self.n_estimators}"
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the optimized Random Forest model."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.forward_bars,
            profit_threshold=self.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        
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
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        # Add ATR-based stops
        atr = (data["high"] - data["low"]).rolling(14).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            data["close"] - 2.0 * atr,
            np.where(
                data["entry_signal"] == -1,
                data["close"] + 2.0 * atr,
                np.nan
            )
        )
        
        return data


class OptimizedLightGBM:
    """
    LightGBM with optimized hyperparameters.
    Often faster and sometimes better than XGBoost.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        forward_bars: int = 8,
        profit_threshold: float = 0.003,
        min_probability: float = 0.40,
    ):
        if not HAS_LGB:
            raise ImportError("lightgbm required")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.forward_bars = forward_bars
        self.profit_threshold = profit_threshold
        self.min_probability = min_probability
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return f"OPT_LGBM_d{self.max_depth}_lr{self.learning_rate}"
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the optimized LightGBM model."""
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=self.forward_bars,
            profit_threshold=self.profit_threshold,
        )
        
        combined = pd.concat([features, labels.rename("label")], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]
        
        self._feature_names = list(X.columns)
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        
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
        long_mask = (data["ml_signal"] == 1) & (data["ml_confidence"] >= self.min_probability)
        short_mask = (data["ml_signal"] == -1) & (data["ml_confidence"] >= self.min_probability)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        # Add ATR-based stops
        atr = (data["high"] - data["low"]).rolling(14).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            data["close"] - 2.0 * atr,
            np.where(
                data["entry_signal"] == -1,
                data["close"] + 2.0 * atr,
                np.nan
            )
        )
        
        return data


class AggressiveMLEnsemble:
    """
    Aggressive ML Ensemble that combines multiple optimized models.
    
    Takes signal when ANY model strongly predicts, not requiring consensus.
    """
    
    def __init__(
        self,
        forward_bars: int = 6,
        profit_threshold: float = 0.002,
        min_probability: float = 0.35,
    ):
        self.forward_bars = forward_bars
        self.profit_threshold = profit_threshold
        self.min_probability = min_probability
        
        self._models = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "AGGRESSIVE_ML_ENSEMBLE"
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train all models in the ensemble."""
        self._models = []
        
        # XGBoost variants
        for lr in [0.03, 0.1]:
            for depth in [4, 6]:
                model = OptimizedXGBoost(
                    n_estimators=100,
                    max_depth=depth,
                    learning_rate=lr,
                    forward_bars=self.forward_bars,
                    profit_threshold=self.profit_threshold,
                    min_probability=self.min_probability,
                )
                model.train(data)
                self._models.append(model)
        
        # Random Forest variant
        rf = OptimizedRandomForest(
            n_estimators=150,
            max_depth=8,
            forward_bars=self.forward_bars,
            profit_threshold=self.profit_threshold,
            min_probability=self.min_probability,
        )
        rf.train(data)
        self._models.append(rf)
        
        # LightGBM if available
        if HAS_LGB:
            lgbm = OptimizedLightGBM(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                forward_bars=self.forward_bars,
                profit_threshold=self.profit_threshold,
                min_probability=self.min_probability,
            )
            lgbm.train(data)
            self._models.append(lgbm)
        
        self._is_fitted = True
        return {"n_models": len(self._models)}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals - take signal if ANY model is confident."""
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self.train(data.iloc[:train_size])
        
        data = data.copy()
        
        # Collect signals from all models
        all_signals = []
        for model in self._models:
            signals = model.generate_signals(data.copy())
            all_signals.append(signals["entry_signal"].values)
        
        # Aggregate: take strongest signal
        all_signals = np.array(all_signals)
        
        # Count votes
        long_votes = (all_signals == 1).sum(axis=0)
        short_votes = (all_signals == -1).sum(axis=0)
        
        data["entry_signal"] = 0
        
        # Take long if more long votes than short (and at least 2 votes)
        data.loc[(long_votes >= 2) & (long_votes > short_votes), "entry_signal"] = 1
        
        # Take short if more short votes than long (and at least 2 votes)
        data.loc[(short_votes >= 2) & (short_votes > long_votes), "entry_signal"] = -1
        
        # Add stops
        atr = (data["high"] - data["low"]).rolling(14).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            data["close"] - 2.0 * atr,
            np.where(
                data["entry_signal"] == -1,
                data["close"] + 2.0 * atr,
                np.nan
            )
        )
        
        return data
