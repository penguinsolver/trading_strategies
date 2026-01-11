"""
Strategy: Z-Score Mean Reversion with Strict Regime Gate

Mean reversion strategy that only trades in ranging markets.
Entry: Z-score exceeds entry threshold in range regime (low ADX)
Exit: Z-score returns to exit threshold or crosses zero
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, adx, zscore, sma


class ZScoreReversionStrategy(Strategy):
    """Z-score mean reversion with strict regime filtering."""
    
    @property
    def name(self) -> str:
        return "Z-Score Reversion"
    
    @property
    def description(self) -> str:
        return "Mean reversion based on Z-score extremes. Only trades when ADX is low (ranging market)."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="lookback",
                label="Z-Score Lookback",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="Lookback period for mean and std calculation",
            ),
            ParamConfig(
                name="z_entry",
                label="Entry Z-Score",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="Z-score threshold for entry",
            ),
            ParamConfig(
                name="z_exit",
                label="Exit Z-Score",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.25,
                help_text="Z-score threshold for exit (closer to mean)",
            ),
            ParamConfig(
                name="adx_threshold",
                label="ADX Threshold",
                param_type="int",
                default=20,
                min_value=15,
                max_value=25,
                step=5,
                help_text="Maximum ADX for ranging market filter",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiplier for stop loss",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Z-score mean reversion signals."""
        # Parameters
        lookback = self.params.get("lookback", 20)
        z_entry = self.params.get("z_entry", 2.0)
        z_exit = self.params.get("z_exit", 0.5)
        adx_threshold = self.params.get("adx_threshold", 20)
        atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        
        # Calculate Z-score
        data["zscore"] = zscore(data["close"], lookback)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Calculate ADX for regime detection
        adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], 14)
        data["adx"] = adx_val
        
        # Range regime filter: ADX must be low
        data["range_regime"] = data["adx"] < adx_threshold
        
        # Mean reversion signals
        # Long: Z-score very negative (oversold) in range regime
        data["long_signal"] = (
            (data["zscore"] < -z_entry) &
            data["range_regime"]
        )
        
        # Short: Z-score very positive (overbought) in range regime
        data["short_signal"] = (
            (data["zscore"] > z_entry) &
            data["range_regime"]
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["long_signal"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - (
            data.loc[long_mask, "atr"] * atr_stop_mult
        )
        
        # Short entries
        short_mask = data["short_signal"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + (
            data.loc[short_mask, "atr"] * atr_stop_mult
        )
        
        # Exit signals: Z-score returns toward mean
        # Exit long when Z crosses above z_exit (recovering from oversold)
        data["exit_long"] = (data["zscore"] > z_exit) & (data["zscore"].shift(1) <= z_exit)
        # Exit short when Z crosses below -z_exit (recovering from overbought)
        data["exit_short"] = (data["zscore"] < -z_exit) & (data["zscore"].shift(1) >= -z_exit)
        
        data.loc[data["exit_long"] | data["exit_short"], "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "ADX", "column": "adx", "color": "purple", "style": "solid"},
        ]
