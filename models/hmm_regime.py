"""
HMM Regime Detection - Detect market states using Hidden Markov Models.

Uses Gaussian HMM to classify market into different regimes (Bull, Bear, Sideways)
based on returns, volatility, and volume patterns.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
from enum import IntEnum
import warnings

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    warnings.warn("hmmlearn not installed. HMM features will be unavailable.")

if TYPE_CHECKING:
    from strategies.base import Strategy


class MarketRegime(IntEnum):
    """Market regime states."""
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


@dataclass
class HMMConfig:
    """Configuration for HMM regime detection."""
    
    # Number of hidden states
    n_states: int = 3
    
    # Features to use
    use_returns: bool = True
    use_volatility: bool = True
    use_volume: bool = True
    
    # Lookback periods
    returns_period: int = 1  # 1-bar returns
    volatility_period: int = 20  # Rolling volatility window
    volume_period: int = 20  # Rolling volume average
    
    # Training parameters
    n_iter: int = 100
    random_state: int = 42
    
    # Trading filters
    trade_in_bull: bool = True
    trade_in_sideways: bool = True
    trade_in_bear: bool = False  # Usually don't want to trade in bear
    
    # Minimum probability for regime classification
    min_regime_probability: float = 0.6


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    Classifies market into Bull, Bear, or Sideways states based on:
    - Returns (direction)
    - Volatility (risk level)
    - Volume (participation)
    """
    
    def __init__(self, config: Optional[HMMConfig] = None):
        if not HAS_HMMLEARN:
            raise ImportError("hmmlearn is required for HMM regime detection. Install with: pip install hmmlearn")
        
        self.config = config or HMMConfig()
        self.model: Optional[GaussianHMM] = None
        self._is_fitted = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._regime_mapping: dict = {}  # Map HMM states to MarketRegime
    
    def _build_features(self, data: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from price data."""
        features = []
        
        if self.config.use_returns:
            returns = data["close"].pct_change(self.config.returns_period)
            features.append(returns)
        
        if self.config.use_volatility:
            returns = data["close"].pct_change()
            volatility = returns.rolling(self.config.volatility_period).std()
            features.append(volatility)
        
        if self.config.use_volume and "volume" in data.columns:
            volume_ratio = data["volume"] / data["volume"].rolling(self.config.volume_period).mean()
            features.append(volume_ratio)
        
        # Combine features
        feature_matrix = pd.concat(features, axis=1).dropna()
        return feature_matrix.values.astype(np.float64)
    
    def _normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features to zero mean, unit variance."""
        if fit:
            self._feature_means = np.nanmean(features, axis=0)
            self._feature_stds = np.nanstd(features, axis=0)
            self._feature_stds[self._feature_stds == 0] = 1  # Avoid division by zero
        
        normalized = (features - self._feature_means) / self._feature_stds
        # Replace NaN/Inf with 0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized
    
    def _map_states_to_regimes(self, features: np.ndarray, states: np.ndarray):
        """Map HMM hidden states to interpretable regime labels based on average returns."""
        # Calculate average returns per state
        returns_idx = 0  # Returns is always first feature if used
        state_returns = {}
        
        for state in range(self.config.n_states):
            mask = states == state
            if mask.sum() > 0:
                avg_return = features[mask, returns_idx].mean()
                state_returns[state] = avg_return
            else:
                state_returns[state] = 0.0
        
        # Sort states by average return
        sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s])
        
        # Map: lowest returns = BEAR, highest = BULL, middle = SIDEWAYS
        if self.config.n_states == 3:
            self._regime_mapping = {
                sorted_states[0]: MarketRegime.BEAR,
                sorted_states[1]: MarketRegime.SIDEWAYS,
                sorted_states[2]: MarketRegime.BULL,
            }
        elif self.config.n_states == 2:
            self._regime_mapping = {
                sorted_states[0]: MarketRegime.BEAR,
                sorted_states[1]: MarketRegime.BULL,
            }
        else:
            # For more states, map proportionally
            for i, state in enumerate(sorted_states):
                if i < len(sorted_states) // 3:
                    self._regime_mapping[state] = MarketRegime.BEAR
                elif i >= 2 * len(sorted_states) // 3:
                    self._regime_mapping[state] = MarketRegime.BULL
                else:
                    self._regime_mapping[state] = MarketRegime.SIDEWAYS
    
    def fit(self, data: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the HMM to historical price data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            self (for chaining)
        """
        # Build and normalize features
        features = self._build_features(data)
        features_normalized = self._normalize_features(features, fit=True)
        
        if len(features_normalized) < 50:
            raise ValueError("Need at least 50 samples to fit HMM")
        
        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type="full",
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(features_normalized)
        
        # Get states and map to regimes
        states = self.model.predict(features_normalized)
        self._map_states_to_regimes(features, states)
        
        self._is_fitted = True
        return self
    
    def predict_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market regime for each bar.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added columns:
            - hmm_state: Raw HMM state (0, 1, 2, ...)
            - hmm_regime: MarketRegime enum value
            - hmm_regime_label: String label (Bull, Bear, Sideways)
            - hmm_proba_*: Probability for each state
            - hmm_allow_trade: Whether trading is allowed
        """
        if not self._is_fitted:
            raise ValueError("HMM must be fitted before prediction. Call fit() first.")
        
        data = data.copy()
        
        # Build features
        features = self._build_features(data)
        features_normalized = self._normalize_features(features, fit=False)
        
        # Predict states and get probabilities
        states = self.model.predict(features_normalized)
        proba = self.model.predict_proba(features_normalized)
        
        # Create result series aligned with data index
        # Features have fewer rows due to lookback, so we need to align
        valid_idx = data.index[-len(states):]
        
        # Initialize columns with NaN
        data["hmm_state"] = np.nan
        data["hmm_regime"] = np.nan
        data["hmm_regime_label"] = ""
        data["hmm_allow_trade"] = False
        
        for i in range(self.config.n_states):
            data[f"hmm_proba_{i}"] = np.nan
        
        # Fill in values for valid indices
        data.loc[valid_idx, "hmm_state"] = states
        
        regime_values = [self._regime_mapping.get(s, MarketRegime.SIDEWAYS) for s in states]
        regime_labels = [MarketRegime(r).name for r in regime_values]
        
        data.loc[valid_idx, "hmm_regime"] = regime_values
        data.loc[valid_idx, "hmm_regime_label"] = regime_labels
        
        for i in range(self.config.n_states):
            data.loc[valid_idx, f"hmm_proba_{i}"] = proba[:, i]
        
        # Determine if trading is allowed
        allow_trade = pd.Series(False, index=data.index)
        
        if self.config.trade_in_bull:
            allow_trade |= (data["hmm_regime"] == MarketRegime.BULL)
        if self.config.trade_in_sideways:
            allow_trade |= (data["hmm_regime"] == MarketRegime.SIDEWAYS)
        if self.config.trade_in_bear:
            allow_trade |= (data["hmm_regime"] == MarketRegime.BEAR)
        
        # Also check minimum probability
        max_proba = data[[f"hmm_proba_{i}" for i in range(self.config.n_states)]].max(axis=1)
        allow_trade &= (max_proba >= self.config.min_regime_probability)
        
        data["hmm_allow_trade"] = allow_trade.fillna(False)
        
        return data
    
    def filter_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter trading signals based on HMM regime.
        
        Expects data to have "entry_signal" column.
        Zeroes out signals where regime doesn't allow trading.
        """
        if "entry_signal" not in data.columns:
            return data
        
        # Predict regime if not already done
        if "hmm_allow_trade" not in data.columns:
            data = self.predict_regime(data)
        
        # Store original signals
        data["entry_signal_unfiltered"] = data["entry_signal"]
        
        # Filter signals
        data.loc[~data["hmm_allow_trade"], "entry_signal"] = 0
        
        return data
    
    def get_regime_stats(self, data: pd.DataFrame) -> dict:
        """Get statistics about regime distribution in data."""
        if "hmm_regime" not in data.columns:
            data = self.predict_regime(data)
        
        stats = {}
        for regime in MarketRegime:
            count = (data["hmm_regime"] == regime).sum()
            pct = count / len(data) * 100
            stats[regime.name] = {"count": count, "percentage": pct}
        
        return stats


class HMMFilteredStrategy:
    """
    Wrapper that applies HMM regime filtering to any strategy.
    """
    
    def __init__(
        self,
        strategy: "Strategy",
        hmm_detector: Optional[HMMRegimeDetector] = None,
        auto_fit: bool = True,
    ):
        self._strategy = strategy
        self.hmm_detector = hmm_detector or HMMRegimeDetector()
        self._auto_fit = auto_fit
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return f"HMM_{self._strategy.name}"
    
    @property
    def description(self) -> str:
        return f"{self._strategy.description} (with HMM regime filter)"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals, then filter by HMM regime."""
        # Auto-fit if needed
        if self._auto_fit and not self._is_fitted:
            self.hmm_detector.fit(data)
            self._is_fitted = True
        
        # First, predict regime
        data = self.hmm_detector.predict_regime(data)
        
        # Generate signals from underlying strategy
        data = self._strategy.generate_signals(data)
        
        # Filter signals based on regime
        data = self.hmm_detector.filter_signals(data)
        
        return data
