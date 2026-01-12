"""
XGBoost Meta-Classifier - ML model that predicts profitable trades.

Uses signals from all strategies plus market features to predict
whether the next trade will be profitable.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, TYPE_CHECKING, Any
import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost or sklearn not installed. ML features will be unavailable.")

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class XGBConfig:
    """Configuration for XGBoost classifier."""
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    
    # Training parameters
    early_stopping_rounds: int = 10
    eval_metric: str = "auc"
    random_state: int = 42
    
    # Feature engineering
    use_strategy_signals: bool = True
    use_technical_features: bool = True
    use_time_features: bool = True
    
    # Technical feature periods
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    volatility_period: int = 20
    
    # Target definition
    forward_return_bars: int = 12  # Look 12 bars ahead for profitability
    profit_threshold: float = 0.001  # 0.1% profit to be "profitable"
    
    # Prediction thresholds
    min_probability: float = 0.55  # Minimum P(profit) to take trade
    
    # Signal filtering
    only_filter_entries: bool = True  # Only filter entry signals, not exits


class XGBoostTradeClassifier:
    """
    XGBoost classifier for trade signal filtering.
    
    Predicts whether a potential trade will be profitable based on:
    - Signals from multiple strategies
    - Technical indicators
    - Time-based features
    """
    
    def __init__(self, config: Optional[XGBConfig] = None):
        if not HAS_XGBOOST:
            raise ImportError("xgboost and sklearn are required. Install with: pip install xgboost scikit-learn")
        
        self.config = config or XGBConfig()
        self._model: Optional[xgb.XGBClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._feature_importance: Optional[Dict[str, float]] = None
        self._training_metrics: Dict[str, float] = {}
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0
        
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.ewm(span=period, adjust=False).mean()
    
    def _calculate_atr_ratio(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ATR as ratio to price."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr / close
    
    def build_features(
        self,
        data: pd.DataFrame,
        strategies: Optional[List["Strategy"]] = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix from price data and strategy signals.
        
        Args:
            data: DataFrame with OHLCV columns
            strategies: Optional list of strategies to get signals from
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=data.index)
        
        # Strategy signals
        if self.config.use_strategy_signals and strategies:
            for strategy in strategies:
                try:
                    df_copy = data.copy()
                    df_with_signals = strategy.generate_signals(df_copy)
                    signal = df_with_signals.get("entry_signal", pd.Series(0, index=data.index))
                    features[f"signal_{strategy.name}"] = signal
                except Exception:
                    features[f"signal_{strategy.name}"] = 0
        
        # Technical features
        if self.config.use_technical_features:
            close = data["close"]
            high = data["high"]
            low = data["low"]
            
            # RSI
            features["rsi"] = self._calculate_rsi(close, self.config.rsi_period)
            
            # RSI zones
            features["rsi_oversold"] = (features["rsi"] < 30).astype(float)
            features["rsi_overbought"] = (features["rsi"] > 70).astype(float)
            
            # ADX (trend strength)
            features["adx"] = self._calculate_adx(high, low, close, self.config.adx_period)
            features["adx_trending"] = (features["adx"] > 25).astype(float)
            
            # ATR ratio (volatility)
            features["atr_ratio"] = self._calculate_atr_ratio(high, low, close, self.config.atr_period)
            
            # Returns
            features["return_1"] = close.pct_change(1)
            features["return_5"] = close.pct_change(5)
            features["return_10"] = close.pct_change(10)
            
            # Volatility
            features["volatility"] = close.pct_change().rolling(self.config.volatility_period).std()
            
            # Price position relative to range
            high_20 = high.rolling(20).max()
            low_20 = low.rolling(20).min()
            features["price_position"] = (close - low_20) / (high_20 - low_20 + 1e-10)
            
            # Volume features (if available)
            if "volume" in data.columns:
                features["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
        
        # Time features
        if self.config.use_time_features:
            if isinstance(data.index, pd.DatetimeIndex):
                features["hour"] = data.index.hour / 24.0
                features["day_of_week"] = data.index.dayofweek / 6.0
                features["is_weekend"] = (data.index.dayofweek >= 5).astype(float)
        
        return features
    
    def _create_target(self, data: pd.DataFrame) -> pd.Series:
        """Create binary target: 1 if trade would be profitable."""
        forward_return = data["close"].shift(-self.config.forward_return_bars) / data["close"] - 1
        return (forward_return > self.config.profit_threshold).astype(int)
    
    def train(
        self,
        data: pd.DataFrame,
        strategies: List["Strategy"],
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the XGBoost classifier.
        
        Args:
            data: DataFrame with OHLCV columns
            strategies: List of strategies for signal features
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Build features and target
        features = self.build_features(data, strategies)
        target = self._create_target(data)
        
        # Align and drop NaN
        combined = pd.concat([features, target.rename("target")], axis=1).dropna()
        
        if len(combined) < 100:
            raise ValueError("Not enough data for training (need at least 100 samples)")
        
        X = combined.drop("target", axis=1)
        y = combined["target"]
        
        # Store feature names
        self._feature_names = list(X.columns)
        
        # Split into train/validation (time-based, not random)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_val_scaled = self._scaler.transform(X_val)
        
        # Create and train model
        self._model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            eval_metric=self.config.eval_metric,
            use_label_encoder=False,
        )
        
        # Fit with early stopping
        self._model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )
        
        self._is_fitted = True
        
        # Calculate metrics
        y_train_pred = self._model.predict(X_train_scaled)
        y_val_pred = self._model.predict(X_val_scaled)
        
        self._training_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
            "n_train_samples": len(y_train),
            "n_val_samples": len(y_val),
            "n_features": len(self._feature_names),
        }
        
        # Feature importance
        importance = self._model.feature_importances_
        self._feature_importance = dict(zip(self._feature_names, importance))
        
        return self._training_metrics
    
    def predict_proba(
        self,
        data: pd.DataFrame,
        strategies: Optional[List["Strategy"]] = None,
    ) -> pd.Series:
        """
        Predict probability of profitable trade.
        
        Args:
            data: DataFrame with OHLCV columns
            strategies: Optional list of strategies (if not provided, uses only technical features)
            
        Returns:
            Series with P(profitable) for each bar
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first. Call train() first.")
        
        # Build features
        features = self.build_features(data, strategies)
        
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
        proba = self._model.predict_proba(features_scaled)[:, 1]
        
        return pd.Series(proba, index=data.index, name="profit_probability")
    
    def filter_signals(
        self,
        data: pd.DataFrame,
        strategies: Optional[List["Strategy"]] = None,
    ) -> pd.DataFrame:
        """
        Filter trading signals based on profit probability.
        
        Only allows trades where P(profit) >= threshold.
        """
        if "entry_signal" not in data.columns:
            return data
        
        # Get probabilities
        proba = self.predict_proba(data, strategies)
        data["ml_profit_proba"] = proba
        
        # Store original signals
        data["entry_signal_unfiltered"] = data["entry_signal"]
        
        # Filter: only keep signals where probability is high enough
        low_proba = proba < self.config.min_probability
        
        if self.config.only_filter_entries:
            # Only filter entry signals (1 and -1), not during positions
            data.loc[low_proba & (data["entry_signal"] != 0), "entry_signal"] = 0
        else:
            data.loc[low_proba, "entry_signal"] = 0
        
        return data
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if self._feature_importance is None:
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
    
    def load(self, path: str) -> "XGBoostTradeClassifier":
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


class XGBoostFilteredStrategy:
    """
    Wrapper that applies XGBoost filtering to any strategy.
    """
    
    def __init__(
        self,
        strategy: "Strategy",
        classifier: Optional[XGBoostTradeClassifier] = None,
        all_strategies: Optional[List["Strategy"]] = None,
        auto_train: bool = True,
    ):
        self._strategy = strategy
        self.classifier = classifier or XGBoostTradeClassifier()
        self._all_strategies = all_strategies or [strategy]
        self._auto_train = auto_train
        self._is_trained = False
    
    @property
    def name(self) -> str:
        return f"XGB_{self._strategy.name}"
    
    @property
    def description(self) -> str:
        return f"{self._strategy.description} (with XGBoost ML filter)"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with XGBoost filtering."""
        # Auto-train if needed
        if self._auto_train and not self._is_trained:
            try:
                self.classifier.train(data, self._all_strategies)
                self._is_trained = True
            except Exception:
                # If training fails, continue without filtering
                pass
        
        # Generate base strategy signals
        data = self._strategy.generate_signals(data)
        
        # Apply ML filtering
        if self._is_trained:
            data = self.classifier.filter_signals(data, self._all_strategies)
        
        return data
