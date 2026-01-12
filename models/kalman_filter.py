"""
Kalman Trend Filter - Adaptive trend detection using Kalman filtering.

Uses Kalman filter to smooth price and estimate trend velocity,
providing less laggy trend signals than moving averages.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
import warnings

try:
    from pykalman import KalmanFilter
    HAS_PYKALMAN = True
except ImportError:
    HAS_PYKALMAN = False
    warnings.warn("pykalman not installed. Kalman features will be unavailable.")

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass  
class KalmanConfig:
    """Configuration for Kalman trend filter."""
    
    # State: [price, velocity]
    # Observation: price only
    
    # Process noise (how much we expect state to change)
    transition_covariance_price: float = 0.01
    transition_covariance_velocity: float = 0.001
    
    # Observation noise (how noisy our measurements are)
    observation_covariance: float = 1.0
    
    # Velocity thresholds for trend direction
    velocity_threshold_up: float = 0.0001  # Velocity > this = uptrend
    velocity_threshold_down: float = -0.0001  # Velocity < this = downtrend
    
    # Signal generation
    use_velocity_crossover: bool = True  # Signal on velocity zero-cross
    use_price_cross_filtered: bool = True  # Signal on price crossing filtered price
    
    # Minimum velocity magnitude for signals
    min_velocity_magnitude: float = 0.00005
    
    # Lookback for velocity smoothing
    velocity_smoothing: int = 3


class KalmanTrendFilter:
    """
    Kalman filter for adaptive trend detection.
    
    Uses a constant velocity model:
    - State: [price, velocity]
    - Observation: price
    
    Provides smoother trend estimates than moving averages with less lag.
    """
    
    def __init__(self, config: Optional[KalmanConfig] = None):
        if not HAS_PYKALMAN:
            raise ImportError("pykalman is required. Install with: pip install pykalman")
        
        self.config = config or KalmanConfig()
        self._kf: Optional[KalmanFilter] = None
        self._setup_kalman()
    
    def _setup_kalman(self):
        """Initialize Kalman filter matrices."""
        # State transition matrix: [price, velocity] -> [price + velocity, velocity]
        transition_matrix = np.array([
            [1, 1],  # price_new = price + velocity
            [0, 1],  # velocity_new = velocity
        ])
        
        # Observation matrix: we only observe price
        observation_matrix = np.array([[1, 0]])
        
        # Process noise covariance
        transition_covariance = np.array([
            [self.config.transition_covariance_price, 0],
            [0, self.config.transition_covariance_velocity],
        ])
        
        # Observation noise covariance
        observation_covariance = np.array([[self.config.observation_covariance]])
        
        # Initial state
        initial_state_mean = np.array([0, 0])
        initial_state_covariance = np.array([
            [1, 0],
            [0, 1],
        ])
        
        self._kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman filter to price data.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            DataFrame with added columns:
            - kalman_price: Filtered price estimate
            - kalman_velocity: Estimated price velocity (trend speed)
            - kalman_trend: Trend direction (+1, 0, -1)
            - kalman_velocity_smooth: Smoothed velocity
        """
        data = data.copy()
        
        # Normalize prices for numerical stability
        prices = data["close"].values
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        prices_normalized = (prices - price_mean) / (price_std + 1e-10)
        
        # Run Kalman filter
        state_means, state_covariances = self._kf.filter(prices_normalized)
        
        # Extract filtered price and velocity
        filtered_price_normalized = state_means[:, 0]
        velocity = state_means[:, 1]
        
        # Denormalize price
        filtered_price = filtered_price_normalized * price_std + price_mean
        
        # Add to data
        data["kalman_price"] = filtered_price
        data["kalman_velocity"] = velocity
        
        # Smooth velocity
        data["kalman_velocity_smooth"] = pd.Series(velocity).rolling(
            self.config.velocity_smoothing, min_periods=1
        ).mean().values
        
        # Determine trend direction
        trend = np.zeros(len(data))
        vel_smooth = data["kalman_velocity_smooth"].values
        
        trend[vel_smooth > self.config.velocity_threshold_up] = 1
        trend[vel_smooth < self.config.velocity_threshold_down] = -1
        
        data["kalman_trend"] = trend.astype(int)
        
        # Velocity magnitude for signal strength
        data["kalman_velocity_magnitude"] = np.abs(vel_smooth)
        
        return data
    
    def get_trend_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Get trend direction for each bar.
        
        Returns:
            Series with +1 (uptrend), -1 (downtrend), 0 (flat)
        """
        if "kalman_trend" not in data.columns:
            data = self.filter(data)
        return data["kalman_trend"]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Kalman filter.
        
        Signals generated on:
        - Velocity crossover (zero-cross of velocity = trend change)
        - Price crossing filtered price
        """
        data = self.filter(data)
        
        # Initialize signals
        data["entry_signal"] = 0
        
        vel_smooth = data["kalman_velocity_smooth"]
        vel_magnitude = data["kalman_velocity_magnitude"]
        
        if self.config.use_velocity_crossover:
            # Signal on velocity zero-cross
            vel_cross_up = (vel_smooth > 0) & (vel_smooth.shift(1) <= 0)
            vel_cross_down = (vel_smooth < 0) & (vel_smooth.shift(1) >= 0)
            
            # Only signal if velocity magnitude is sufficient
            vel_cross_up = vel_cross_up & (vel_magnitude >= self.config.min_velocity_magnitude)
            vel_cross_down = vel_cross_down & (vel_magnitude >= self.config.min_velocity_magnitude)
            
            data.loc[vel_cross_up, "entry_signal"] = 1
            data.loc[vel_cross_down, "entry_signal"] = -1
        
        if self.config.use_price_cross_filtered:
            # Signal on price crossing filtered price (in trend direction)
            close = data["close"]
            filtered = data["kalman_price"]
            trend = data["kalman_trend"]
            
            cross_above = (close > filtered) & (close.shift(1) <= filtered.shift(1))
            cross_below = (close < filtered) & (close.shift(1) >= filtered.shift(1))
            
            # Only if in corresponding trend and not already signaled
            data.loc[(cross_above) & (trend == 1) & (data["entry_signal"] == 0), "entry_signal"] = 1
            data.loc[(cross_below) & (trend == -1) & (data["entry_signal"] == 0), "entry_signal"] = -1
        
        return data
    
    def filter_by_trend(self, data: pd.DataFrame, only_with_trend: bool = True) -> pd.DataFrame:
        """
        Filter existing signals to only allow trades in trend direction.
        
        Long signals only in uptrend, short signals only in downtrend.
        """
        if "entry_signal" not in data.columns:
            return data
        
        if "kalman_trend" not in data.columns:
            data = self.filter(data)
        
        if only_with_trend:
            # Store original
            data["entry_signal_unfiltered"] = data["entry_signal"]
            
            trend = data["kalman_trend"]
            signal = data["entry_signal"]
            
            # Zero out signals against trend
            data.loc[(signal == 1) & (trend != 1), "entry_signal"] = 0
            data.loc[(signal == -1) & (trend != -1), "entry_signal"] = 0
        
        return data


class KalmanStrategy:
    """
    Strategy wrapper that uses Kalman filter for trend detection.
    Can be used standalone or to filter another strategy's signals.
    """
    
    def __init__(
        self,
        base_strategy: Optional["Strategy"] = None,
        config: Optional[KalmanConfig] = None,
        filter_mode: bool = True,  # If True, filter base strategy signals; if False, generate own signals
    ):
        self._base_strategy = base_strategy
        self.kalman_filter = KalmanTrendFilter(config)
        self._filter_mode = filter_mode
    
    @property
    def name(self) -> str:
        if self._base_strategy:
            return f"Kalman_{self._base_strategy.name}"
        return "Kalman_Trend"
    
    @property
    def description(self) -> str:
        if self._base_strategy:
            return f"{self._base_strategy.description} (with Kalman trend filter)"
        return "Kalman filter-based trend following strategy"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate or filter signals using Kalman filter."""
        # Apply Kalman filter
        data = self.kalman_filter.filter(data)
        
        if self._base_strategy and self._filter_mode:
            # Generate base strategy signals
            data = self._base_strategy.generate_signals(data)
            # Filter by Kalman trend
            data = self.kalman_filter.filter_by_trend(data)
        else:
            # Generate signals directly from Kalman filter
            data = self.kalman_filter.generate_signals(data)
        
        return data
