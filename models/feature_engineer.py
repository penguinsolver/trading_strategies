"""
Feature Engineering - Comprehensive technical indicator features for ML models.

Calculates 50+ features including momentum, trend, mean-reversion, volatility,
and volume indicators for ML signal generation.
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Momentum periods
    return_periods: List[int] = None  # Default: [1, 3, 5, 10, 20, 50]
    
    # RSI periods
    rsi_periods: List[int] = None  # Default: [7, 14, 21]
    
    # EMA periods for trend
    ema_periods: List[int] = None  # Default: [10, 20, 50]
    
    # ATR period
    atr_period: int = 14
    
    # Bollinger settings
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ADX period
    adx_period: int = 14
    
    # Volume MA period
    volume_ma_period: int = 20
    
    # Regression slope period
    slope_period: int = 20
    
    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = [1, 3, 5, 10, 20, 50]
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21]
        if self.ema_periods is None:
            self.ema_periods = [10, 20, 50]


class FeatureEngineer:
    """
    Calculates comprehensive technical features for ML models.
    
    Features include:
    - Momentum (returns, ROC)
    - Trend (EMAs, ADX, slope)
    - Mean reversion (RSI, Bollinger, z-score)
    - Volatility (ATR ratio, BB width)
    - Volume (ratio, OBV direction)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self._feature_names.copy()
    
    def _safe_div(self, a, b, fill=0.0):
        """Safe division avoiding divide by zero."""
        return np.where(np.abs(b) > 1e-10, a / b, fill)
    
    # ==================== MOMENTUM FEATURES ====================
    
    def _calc_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for various lookback periods."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        
        for period in self.config.return_periods:
            returns = close.pct_change(period)
            features[f"return_{period}"] = returns
            self._feature_names.append(f"return_{period}")
        
        # Momentum (price vs price N bars ago)
        features["momentum_10"] = close - close.shift(10)
        self._feature_names.append("momentum_10")
        
        # Rate of change
        features["roc_10"] = (close - close.shift(10)) / (close.shift(10) + 1e-10) * 100
        self._feature_names.append("roc_10")
        
        return features
    
    def _calc_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI for multiple periods."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        
        for period in self.config.rsi_periods:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            features[f"rsi_{period}"] = rsi
            self._feature_names.append(f"rsi_{period}")
            
            # RSI zones
            features[f"rsi_{period}_oversold"] = (rsi < 30).astype(float)
            features[f"rsi_{period}_overbought"] = (rsi > 70).astype(float)
            self._feature_names.extend([f"rsi_{period}_oversold", f"rsi_{period}_overbought"])
        
        return features
    
    # ==================== TREND FEATURES ====================
    
    def _calc_ema_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA-based trend features."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        
        emas = {}
        for period in self.config.ema_periods:
            ema = close.ewm(span=period, adjust=False).mean()
            emas[period] = ema
            
            # Distance from EMA (normalized)
            features[f"ema_{period}_dist"] = (close - ema) / (ema + 1e-10)
            self._feature_names.append(f"ema_{period}_dist")
            
            # Price above/below EMA
            features[f"above_ema_{period}"] = (close > ema).astype(float)
            self._feature_names.append(f"above_ema_{period}")
        
        # EMA crossover features
        if len(self.config.ema_periods) >= 2:
            fast = emas[self.config.ema_periods[0]]
            slow = emas[self.config.ema_periods[1]]
            features["ema_cross"] = (fast > slow).astype(float)
            features["ema_cross_signal"] = features["ema_cross"].diff()
            self._feature_names.extend(["ema_cross", "ema_cross_signal"])
        
        return features
    
    def _calc_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX and directional indicators."""
        features = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        period = self.config.adx_period
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0
        
        # Smooth
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        features["adx"] = adx
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        features["di_diff"] = plus_di - minus_di
        features["trending"] = (adx > 25).astype(float)
        
        self._feature_names.extend(["adx", "plus_di", "minus_di", "di_diff", "trending"])
        
        return features
    
    def _calc_slope(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regression slope of price."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        period = self.config.slope_period
        
        def rolling_slope(x):
            if len(x) < period:
                return np.nan
            y = x.values
            x_vals = np.arange(len(y))
            slope = np.polyfit(x_vals, y, 1)[0]
            return slope
        
        slope = close.rolling(period).apply(rolling_slope, raw=False)
        features["price_slope"] = slope / (close + 1e-10)  # Normalize
        features["slope_positive"] = (slope > 0).astype(float)
        
        self._feature_names.extend(["price_slope", "slope_positive"])
        
        return features
    
    # ==================== MEAN REVERSION FEATURES ====================
    
    def _calc_bollinger(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Band features."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        period = self.config.bb_period
        std_mult = self.config.bb_std
        
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        
        # %B - position within bands
        features["bb_pct_b"] = (close - lower) / (upper - lower + 1e-10)
        
        # Bandwidth (volatility)
        features["bb_width"] = (upper - lower) / (sma + 1e-10)
        
        # Distance from bands
        features["bb_upper_dist"] = (upper - close) / (close + 1e-10)
        features["bb_lower_dist"] = (close - lower) / (close + 1e-10)
        
        # Band touch
        features["bb_above_upper"] = (close > upper).astype(float)
        features["bb_below_lower"] = (close < lower).astype(float)
        
        self._feature_names.extend([
            "bb_pct_b", "bb_width", "bb_upper_dist", "bb_lower_dist",
            "bb_above_upper", "bb_below_lower"
        ])
        
        return features
    
    def _calc_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate z-score of price."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        
        for period in [20, 50]:
            mean = close.rolling(period).mean()
            std = close.rolling(period).std()
            zscore = (close - mean) / (std + 1e-10)
            features[f"zscore_{period}"] = zscore
            self._feature_names.append(f"zscore_{period}")
        
        return features
    
    # ==================== VOLATILITY FEATURES ====================
    
    def _calc_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-based volatility features."""
        features = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        period = self.config.atr_period
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        avg_atr = atr.rolling(50).mean()
        
        features["atr_ratio"] = atr / (close + 1e-10)  # ATR as % of price
        features["atr_expansion"] = atr / (avg_atr + 1e-10)  # Current vs average
        features["high_vol"] = (features["atr_expansion"] > 1.5).astype(float)
        features["low_vol"] = (features["atr_expansion"] < 0.7).astype(float)
        
        self._feature_names.extend(["atr_ratio", "atr_expansion", "high_vol", "low_vol"])
        
        return features
    
    def _calc_range_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate high/low range features."""
        features = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        
        # Recent high/low distance
        for period in [10, 20, 50]:
            recent_high = high.rolling(period).max()
            recent_low = low.rolling(period).min()
            
            features[f"high_{period}_dist"] = (recent_high - close) / (close + 1e-10)
            features[f"low_{period}_dist"] = (close - recent_low) / (close + 1e-10)
            features[f"range_{period}_pos"] = (close - recent_low) / (recent_high - recent_low + 1e-10)
            
            self._feature_names.extend([
                f"high_{period}_dist", f"low_{period}_dist", f"range_{period}_pos"
            ])
        
        return features
    
    # ==================== VOLUME FEATURES ====================
    
    def _calc_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        if "volume" not in data.columns:
            return features
        
        volume = data["volume"]
        close = data["close"]
        period = self.config.volume_ma_period
        
        # Volume ratio
        vol_ma = volume.rolling(period).mean()
        features["volume_ratio"] = volume / (vol_ma + 1e-10)
        features["high_volume"] = (features["volume_ratio"] > 1.5).astype(float)
        
        # OBV direction
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        obv_ma = obv.rolling(period).mean()
        features["obv_trend"] = np.sign(obv - obv_ma)
        
        # Volume momentum
        features["volume_momentum"] = volume.pct_change(5)
        
        self._feature_names.extend([
            "volume_ratio", "high_volume", "obv_trend", "volume_momentum"
        ])
        
        return features
    
    # ==================== MACD FEATURES ====================
    
    def _calc_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD features."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        features["macd"] = macd / (close + 1e-10)  # Normalize
        features["macd_signal"] = signal / (close + 1e-10)
        features["macd_histogram"] = histogram / (close + 1e-10)
        features["macd_cross"] = (macd > signal).astype(float)
        features["macd_cross_signal"] = features["macd_cross"].diff()
        features["macd_positive"] = (macd > 0).astype(float)
        
        self._feature_names.extend([
            "macd", "macd_signal", "macd_histogram", 
            "macd_cross", "macd_cross_signal", "macd_positive"
        ])
        
        return features
    
    # ==================== TIME FEATURES ====================
    
    def _calc_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        features = pd.DataFrame(index=data.index)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return features
        
        features["hour"] = data.index.hour / 24.0
        features["day_of_week"] = data.index.dayofweek / 6.0
        features["is_weekend"] = (data.index.dayofweek >= 5).astype(float)
        features["is_us_session"] = ((data.index.hour >= 13) & (data.index.hour <= 21)).astype(float)
        features["is_asian_session"] = ((data.index.hour >= 0) & (data.index.hour <= 8)).astype(float)
        
        self._feature_names.extend([
            "hour", "day_of_week", "is_weekend", "is_us_session", "is_asian_session"
        ])
        
        return features
    
    # ==================== MAIN METHOD ====================
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features from price data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all features
        """
        self._feature_names = []  # Reset feature names
        
        # Calculate all feature groups
        feature_dfs = [
            self._calc_returns(data),
            self._calc_rsi(data),
            self._calc_ema_features(data),
            self._calc_adx(data),
            self._calc_slope(data),
            self._calc_bollinger(data),
            self._calc_zscore(data),
            self._calc_atr(data),
            self._calc_range_features(data),
            self._calc_volume_features(data),
            self._calc_macd(data),
            self._calc_time_features(data),
        ]
        
        # Combine all features
        features = pd.concat(feature_dfs, axis=1)
        
        # Remove any duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]
        
        return features
    
    def calculate_labels(
        self,
        data: pd.DataFrame,
        forward_bars: int = 12,
        profit_threshold: float = 0.005,
        loss_threshold: float = -0.005,
    ) -> pd.Series:
        """
        Calculate labels for ML training.
        
        Labels:
        - 1: Long signal (forward return > profit_threshold)
        - -1: Short signal (forward return < loss_threshold)
        - 0: No signal
        
        Args:
            data: DataFrame with 'close' column
            forward_bars: Number of bars to look ahead
            profit_threshold: Minimum return for long signal
            loss_threshold: Maximum return for short signal
            
        Returns:
            Series with labels
        """
        forward_return = data["close"].shift(-forward_bars) / data["close"] - 1
        
        labels = pd.Series(0, index=data.index)
        labels[forward_return > profit_threshold] = 1
        labels[forward_return < loss_threshold] = -1
        
        return labels
