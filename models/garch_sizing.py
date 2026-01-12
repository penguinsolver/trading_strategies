"""
GARCH Volatility Sizing - Dynamic position sizing based on GARCH volatility forecasts.

Uses GARCH(1,1) model to forecast volatility and scale position sizes inversely
to expected volatility (smaller positions in high vol, larger in low vol).
"""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
import warnings

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    warnings.warn("arch not installed. GARCH features will be unavailable.")

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class GARCHConfig:
    """Configuration for GARCH volatility sizing."""
    
    # GARCH model specification
    p: int = 1  # GARCH lag order
    q: int = 1  # ARCH lag order
    mean: str = "Zero"  # Mean model: "Zero", "Constant", "AR"
    vol: str = "GARCH"  # Volatility model: "GARCH", "EGARCH", "TARCH"
    dist: str = "normal"  # Distribution: "normal", "t", "skewt"
    
    # Fitting parameters
    min_history: int = 100  # Minimum history for fitting
    refit_frequency: int = 24  # Refit every N bars (0 = fit once)
    
    # Position sizing parameters
    target_volatility: float = 0.02  # Target daily volatility (2%)
    max_position_scale: float = 2.0  # Maximum position multiplier
    min_position_scale: float = 0.25  # Minimum position multiplier
    
    # Volatility bands for trading
    high_vol_threshold: float = 1.5  # Vol > avg * this = high volatility
    low_vol_threshold: float = 0.7  # Vol < avg * this = low volatility
    
    # Trading restrictions
    allow_high_vol_trading: bool = True
    reduce_size_high_vol: bool = True


class GARCHVolatilitySizer:
    """
    GARCH-based position sizing and volatility filtering.
    
    Uses GARCH(1,1) to:
    1. Forecast next-period volatility
    2. Scale position sizes inversely to volatility
    3. Optionally filter signals in extreme volatility regimes
    """
    
    def __init__(self, config: Optional[GARCHConfig] = None):
        if not HAS_ARCH:
            raise ImportError("arch is required. Install with: pip install arch")
        
        self.config = config or GARCHConfig()
        self._model = None
        self._fitted_model = None
        self._last_fit_idx = 0
        self._avg_volatility: Optional[float] = None
    
    def _prepare_returns(self, data: pd.DataFrame) -> pd.Series:
        """Prepare returns series for GARCH model."""
        returns = data["close"].pct_change().dropna() * 100  # Percentage returns
        return returns
    
    def fit(self, data: pd.DataFrame) -> "GARCHVolatilitySizer":
        """
        Fit GARCH model to historical data.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            self (for chaining)
        """
        returns = self._prepare_returns(data)
        
        if len(returns) < self.config.min_history:
            raise ValueError(f"Need at least {self.config.min_history} samples to fit GARCH")
        
        # Create GARCH model
        self._model = arch_model(
            returns,
            mean=self.config.mean,
            vol=self.config.vol,
            p=self.config.p,
            q=self.config.q,
            dist=self.config.dist,
        )
        
        # Fit with suppressed output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = self._model.fit(disp="off")
        
        self._last_fit_idx = len(returns)
        
        # Calculate average volatility for scaling
        self._avg_volatility = self._fitted_model.conditional_volatility.mean()
        
        return self
    
    def _should_refit(self, current_idx: int) -> bool:
        """Check if model should be refitted."""
        if self.config.refit_frequency == 0:
            return False
        return (current_idx - self._last_fit_idx) >= self.config.refit_frequency
    
    def forecast_volatility(self, data: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Forecast volatility using GARCH model.
        
        Args:
            data: DataFrame with 'close' column
            horizon: Forecast horizon in bars
            
        Returns:
            DataFrame with added columns:
            - garch_volatility: Conditional volatility
            - garch_forecast: Forecasted volatility
            - garch_vol_ratio: Current vol / average vol
            - garch_vol_regime: 'high', 'normal', or 'low'
        """
        data = data.copy()
        returns = self._prepare_returns(data)
        
        # Fit if not already fitted or needs refit
        if self._fitted_model is None:
            self.fit(data)
        elif self._should_refit(len(returns)):
            self.fit(data)
        
        # Get conditional volatility (historical)
        cond_vol = self._fitted_model.conditional_volatility
        
        # Align with data index
        data["garch_volatility"] = np.nan
        data.loc[cond_vol.index, "garch_volatility"] = cond_vol.values
        
        # One-step ahead forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._fitted_model.forecast(horizon=horizon)
        
        # Get forecasted variance and convert to volatility
        forecast_vol = np.sqrt(forecast.variance.values[-1, 0])
        
        # Add forecast to last row
        data["garch_forecast"] = np.nan
        data.iloc[-1, data.columns.get_loc("garch_forecast")] = forecast_vol
        
        # Forward fill forecast for practical use
        data["garch_forecast"] = data["garch_forecast"].fillna(method="ffill")
        data["garch_forecast"] = data["garch_forecast"].fillna(data["garch_volatility"])
        
        # Calculate volatility ratio
        data["garch_vol_ratio"] = data["garch_volatility"] / (self._avg_volatility + 1e-10)
        
        # Determine volatility regime
        data["garch_vol_regime"] = "normal"
        data.loc[data["garch_vol_ratio"] > self.config.high_vol_threshold, "garch_vol_regime"] = "high"
        data.loc[data["garch_vol_ratio"] < self.config.low_vol_threshold, "garch_vol_regime"] = "low"
        
        return data
    
    def get_position_scale(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position size scaling based on volatility.
        
        Higher volatility = smaller position size (inverse scaling)
        
        Returns:
            Series with position multipliers (0.25 to 2.0 typically)
        """
        if "garch_forecast" not in data.columns:
            data = self.forecast_volatility(data)
        
        forecast_vol = data["garch_forecast"]
        
        # Target volatility scaled position
        # scale = target_vol / forecast_vol
        # Clamp between min and max
        scale = self.config.target_volatility / (forecast_vol / 100 + 1e-10)  # Convert from percentage
        scale = scale.clip(
            lower=self.config.min_position_scale,
            upper=self.config.max_position_scale
        )
        
        return scale
    
    def apply_sizing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply GARCH-based position sizing to data.
        
        Adds columns:
        - garch_position_scale: Position size multiplier
        - risk_percent_adjusted: Adjusted risk based on volatility
        """
        data = self.forecast_volatility(data)
        data["garch_position_scale"] = self.get_position_scale(data)
        
        # If there's an existing risk_percent, adjust it
        if "risk_percent" in data.columns:
            data["risk_percent_original"] = data["risk_percent"]
            data["risk_percent"] = data["risk_percent"] * data["garch_position_scale"]
        
        return data
    
    def filter_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals based on volatility regime.
        
        Optionally removes signals in high volatility regimes.
        """
        if "entry_signal" not in data.columns:
            return data
        
        if "garch_vol_regime" not in data.columns:
            data = self.forecast_volatility(data)
        
        # Store original
        data["entry_signal_unfiltered"] = data["entry_signal"]
        
        # Filter high volatility signals if configured
        if not self.config.allow_high_vol_trading:
            data.loc[data["garch_vol_regime"] == "high", "entry_signal"] = 0
        
        return data
    
    def get_volatility_stats(self, data: pd.DataFrame) -> dict:
        """Get volatility statistics."""
        if "garch_vol_regime" not in data.columns:
            data = self.forecast_volatility(data)
        
        stats = {
            "current_volatility": data["garch_volatility"].iloc[-1],
            "forecast_volatility": data["garch_forecast"].iloc[-1],
            "average_volatility": self._avg_volatility,
            "current_regime": data["garch_vol_regime"].iloc[-1],
            "position_scale": data["garch_position_scale"].iloc[-1] if "garch_position_scale" in data.columns else 1.0,
        }
        
        # Regime distribution
        for regime in ["high", "normal", "low"]:
            count = (data["garch_vol_regime"] == regime).sum()
            pct = count / len(data) * 100
            stats[f"regime_{regime}_pct"] = pct
        
        return stats


class GARCHSizedStrategy:
    """
    Wrapper that applies GARCH volatility sizing to any strategy.
    """
    
    def __init__(
        self,
        strategy: "Strategy",
        garch_sizer: Optional[GARCHVolatilitySizer] = None,
        auto_fit: bool = True,
    ):
        self._strategy = strategy
        self.garch_sizer = garch_sizer or GARCHVolatilitySizer()
        self._auto_fit = auto_fit
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return f"GARCH_{self._strategy.name}"
    
    @property
    def description(self) -> str:
        return f"{self._strategy.description} (with GARCH volatility sizing)"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with GARCH volatility adjustments."""
        # Auto-fit if needed
        if self._auto_fit and not self._is_fitted:
            try:
                self.garch_sizer.fit(data)
                self._is_fitted = True
            except Exception:
                # If fitting fails, continue without GARCH
                pass
        
        # Generate base strategy signals
        data = self._strategy.generate_signals(data)
        
        # Apply GARCH volatility forecasting
        if self._is_fitted:
            data = self.garch_sizer.forecast_volatility(data)
            data = self.garch_sizer.apply_sizing(data)
            data = self.garch_sizer.filter_signals(data)
        
        return data
