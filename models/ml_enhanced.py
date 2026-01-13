"""
ML-Enhanced Trading Strategies.

Uses ML models to enhance/filter existing TA signals rather than generate signals independently.
This combines the best of both worlds: TA for signal generation, ML for filtering.
"""
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False

from .feature_engineer import FeatureEngineer


@dataclass
class EnhancedConfig:
    """Configuration for ML-enhanced strategies."""
    # ML config
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.03
    
    # Signal enhancement
    min_ml_confidence: float = 0.40  # Take signal if ML is at least this confident
    boost_threshold: float = 0.55  # Boost position if ML is very confident
    
    # Training
    forward_bars: int = 6
    profit_threshold: float = 0.002


class MLEnhancedBreakout:
    """
    Breakout strategy enhanced with ML confirmation.
    
    Uses Donchian channel breakouts for signals, but only takes trades
    where ML model predicts favorable outcome.
    Also uses ML confidence to size positions.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        config: Optional[EnhancedConfig] = None,
    ):
        if not HAS_ML:
            raise ImportError("sklearn and xgboost required")
        
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.config = config or EnhancedConfig()
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "ML_ENHANCED_BREAKOUT"
    
    def _train_ml(self, data: pd.DataFrame):
        """Train ML enhancement model."""
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
        
        self._model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=1,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_scaled, y_encoded)
        
        self._is_fitted = True
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-enhanced breakout signals."""
        data = data.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # 1. Calculate breakout signals (Donchian)
        upper = high.rolling(self.lookback).max().shift(1)
        lower = low.rolling(self.lookback).min().shift(1)
        
        breakout_long = close > upper
        breakout_short = close < lower
        
        # ATR for stops
        atr = (high - low).rolling(self.atr_period).mean()
        
        # 2. Train ML if needed
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self._train_ml(data.iloc[:train_size])
        
        # 3. Get ML predictions
        features = self.feature_engineer.calculate_features(data)
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        ml_preds = self._model.predict(X_scaled)
        ml_probs = self._model.predict_proba(X_scaled)
        
        ml_decoded = self._label_encoder.inverse_transform(ml_preds)
        ml_confidence = ml_probs.max(axis=1)
        
        data["breakout_long"] = breakout_long
        data["breakout_short"] = breakout_short
        data["ml_pred"] = ml_decoded
        data["ml_confidence"] = ml_confidence
        
        # 4. Combine: Breakout + ML agreement
        data["entry_signal"] = 0
        
        # Long: Breakout long AND (ML agrees with long OR is neutral with confidence)
        long_mask = breakout_long & (
            (ml_decoded == 1) |  # ML predicts up
            ((ml_decoded == 0) & (ml_confidence >= self.config.min_ml_confidence))  # Neutral but confident
        )
        
        # Short: Breakout short AND (ML agrees with short OR is neutral with confidence)
        short_mask = breakout_short & (
            (ml_decoded == -1) |  # ML predicts down
            ((ml_decoded == 0) & (ml_confidence >= self.config.min_ml_confidence))
        )
        
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        # 5. Add stops
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


class MLEnhancedMACrossover:
    """
    MA Crossover enhanced with ML confirmation.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        config: Optional[EnhancedConfig] = None,
    ):
        if not HAS_ML:
            raise ImportError("sklearn and xgboost required")
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.config = config or EnhancedConfig()
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "ML_ENHANCED_MA"
    
    def _train_ml(self, data: pd.DataFrame):
        """Train ML enhancement model."""
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
        
        self._model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
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
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-enhanced MA crossover signals."""
        data = data.copy()
        close = data["close"]
        
        # 1. Calculate MA crossover signals
        fast_ma = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ma = close.ewm(span=self.slow_period, adjust=False).mean()
        
        cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # ATR for stops
        atr = (data["high"] - data["low"]).rolling(self.atr_period).mean()
        
        # 2. Train ML if needed
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self._train_ml(data.iloc[:train_size])
        
        # 3. Get ML predictions
        features = self.feature_engineer.calculate_features(data)
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        ml_preds = self._model.predict(X_scaled)
        ml_probs = self._model.predict_proba(X_scaled)
        
        ml_decoded = self._label_encoder.inverse_transform(ml_preds)
        ml_confidence = ml_probs.max(axis=1)
        
        data["cross_up"] = cross_up
        data["cross_down"] = cross_down
        data["ml_pred"] = ml_decoded
        data["ml_confidence"] = ml_confidence
        
        # 4. Combine: MA signal + ML agreement
        data["entry_signal"] = 0
        
        # Long: MA cross up AND (ML agrees or neutral+confident)
        long_mask = cross_up & (
            (ml_decoded == 1) | 
            ((ml_decoded != -1) & (ml_confidence >= self.config.min_ml_confidence))
        )
        
        # Short: MA cross down AND (ML agrees or neutral+confident)
        short_mask = cross_down & (
            (ml_decoded == -1) | 
            ((ml_decoded != 1) & (ml_confidence >= self.config.min_ml_confidence))
        )
        
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[short_mask, "entry_signal"] = -1
        
        # 5. Add stops
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


class MLTrendFollower:
    """
    Pure ML trend following strategy.
    
    Uses ML to predict trend direction, then follows trends
    with momentum confirmation.
    """
    
    def __init__(
        self,
        trend_period: int = 20,
        momentum_period: int = 10,
        min_probability: float = 0.45,
        n_estimators: int = 100,
        max_depth: int = 4,
    ):
        if not HAS_ML:
            raise ImportError("sklearn and xgboost required")
        
        self.trend_period = trend_period
        self.momentum_period = momentum_period
        self.min_probability = min_probability
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "ML_TREND_FOLLOWER"
    
    def _train(self, data: pd.DataFrame):
        """Train the trend prediction model."""
        # Use shorter forward_bars for more reactive predictions
        features = self.feature_engineer.calculate_features(data)
        labels = self.feature_engineer.calculate_labels(
            data,
            forward_bars=4,  # Short-term
            profit_threshold=0.002,  # 0.2%
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
            learning_rate=0.05,
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
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML trend-following signals."""
        data = data.copy()
        close = data["close"]
        
        # 1. Train ML if needed
        if not self._is_fitted:
            train_size = int(len(data) * 0.7)
            self._train(data.iloc[:train_size])
        
        # 2. Calculate trend and momentum
        trend_ema = close.ewm(span=self.trend_period, adjust=False).mean()
        momentum = close.pct_change(self.momentum_period)
        
        uptrend = close > trend_ema
        downtrend = close < trend_ema
        
        # 3. Get ML predictions
        features = self.feature_engineer.calculate_features(data)
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_names].fillna(0)
        
        X_scaled = self._scaler.transform(features)
        ml_preds = self._model.predict(X_scaled)
        ml_probs = self._model.predict_proba(X_scaled)
        
        ml_decoded = self._label_encoder.inverse_transform(ml_preds)
        ml_confidence = ml_probs.max(axis=1)
        
        data["uptrend"] = uptrend
        data["downtrend"] = downtrend
        data["momentum"] = momentum
        data["ml_pred"] = ml_decoded
        data["ml_confidence"] = ml_confidence
        
        # 4. Combine signals
        data["entry_signal"] = 0
        
        # Long: In uptrend + positive momentum + ML predicts up with confidence
        long_mask = (
            uptrend &
            (momentum > 0) &
            (ml_decoded == 1) &
            (ml_confidence >= self.min_probability)
        )
        
        # Short: In downtrend + negative momentum + ML predicts down with confidence
        short_mask = (
            downtrend &
            (momentum < 0) &
            (ml_decoded == -1) &
            (ml_confidence >= self.min_probability)
        )
        
        # Only signal on state changes
        prev_uptrend = uptrend.shift(1).fillna(False)
        trend_change = uptrend != prev_uptrend
        
        data.loc[long_mask & trend_change, "entry_signal"] = 1
        data.loc[short_mask & trend_change, "entry_signal"] = -1
        
        # 5. Add stops
        atr = (data["high"] - data["low"]).rolling(14).mean()
        data["stop_price"] = np.where(
            data["entry_signal"] == 1,
            close - 2.0 * atr,
            np.where(
                data["entry_signal"] == -1,
                close + 2.0 * atr,
                np.nan
            )
        )
        
        return data
