"""
Base strategy class and parameter configuration.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd


@dataclass
class ParamConfig:
    """Configuration for a strategy parameter (for UI rendering)."""
    name: str
    label: str
    param_type: str  # "int", "float", "bool", "select"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list] = None  # For select type
    help_text: str = ""


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    Each strategy must implement:
    - name: Strategy display name
    - generate_signals: Add entry/exit signals to data
    - get_param_config: Define adjustable parameters for UI
    
    Signal columns added by generate_signals:
    - entry_signal: 1 for long entry, -1 for short entry, 0 for no signal
    - exit_signal: True when position should be closed by signal
    - stop_price: Stop-loss price for the trade
    - target_price: (optional) Take-profit price
    - trailing_stop_atr: (optional) Trailing stop distance in price units
    """
    
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.params = self._apply_defaults(params)
    
    def _apply_defaults(self, params: dict) -> dict:
        """Apply default values for missing parameters."""
        defaults = {p.name: p.default for p in self.get_param_config()}
        defaults.update(params)
        return defaults
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy display name."""
        pass
    
    @property
    def description(self) -> str:
        """Strategy description (optional)."""
        return ""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals on the data.
        
        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with added signal columns:
            - entry_signal: 1 (long), -1 (short), 0 (none)
            - exit_signal: True/False
            - stop_price: float
            - target_price: float (optional)
            - trailing_stop_atr: float (optional)
        """
        pass
    
    @abstractmethod
    def get_param_config(self) -> list[ParamConfig]:
        """
        Get parameter configuration for UI rendering.
        
        Returns:
            List of ParamConfig objects defining adjustable parameters
        """
        pass
    
    def get_indicator_info(self) -> list[dict]:
        """
        Get information about indicators to display on chart.
        
        Returns:
            List of dicts with keys: name, column, color, style
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.name}({self.params})"
