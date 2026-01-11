"""
Strategy: Bollinger Band Mean Reversion (Anti-Chop)

Mean reversion strategy that fades outer Bollinger Bands with strict regime gate.
Only trades in choppy/ranging markets (ADX low OR Choppiness Index high).
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, adx, bollinger_bands, choppiness_index, atr_slope, vwap


class BBMeanReversionStrategy(Strategy):
    """Bollinger Band mean reversion with regime gate."""
    
    @property
    def name(self) -> str:
        return "BB Mean Reversion"
    
    @property
    def description(self) -> str:
        return "Fade outer Bollinger Bands in choppy/ranging markets. Strict ADX/Choppiness regime gate."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="bb_period",
                label="BB Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="Bollinger Band period",
            ),
            ParamConfig(
                name="bb_std",
                label="BB Std Dev",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="Bollinger Band standard deviation",
            ),
            ParamConfig(
                name="adx_threshold",
                label="ADX Threshold",
                param_type="int",
                default=25,
                min_value=15,
                max_value=35,
                step=5,
                help_text="Max ADX for ranging market (below = range)",
            ),
            ParamConfig(
                name="chop_threshold",
                label="Choppiness Threshold",
                param_type="float",
                default=61.8,
                min_value=50.0,
                max_value=75.0,
                step=5.0,
                help_text="Min Choppiness for choppy market (above = choppy)",
            ),
            ParamConfig(
                name="exit_mode",
                label="Exit Mode",
                param_type="select",
                default="mid_band",
                options=["mid_band", "vwap", "opposite_band"],
                help_text="Target for mean reversion exit",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate BB mean reversion signals with regime gate."""
        # Parameters
        bb_period = self.params.get("bb_period", 20)
        bb_std = self.params.get("bb_std", 2.0)
        adx_threshold = self.params.get("adx_threshold", 25)
        chop_threshold = self.params.get("chop_threshold", 61.8)
        exit_mode = self.params.get("exit_mode", "mid_band")
        
        # Calculate indicators
        upper, lower, middle = bollinger_bands(data["close"], bb_period, bb_std)
        data["bb_upper"] = upper
        data["bb_lower"] = lower
        data["bb_mid"] = middle
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Calculate ADX for regime
        adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], 14)
        data["adx"] = adx_val
        
        # Calculate Choppiness Index
        data["choppiness"] = choppiness_index(data["high"], data["low"], data["close"], 14)
        
        # Calculate ATR slope (flat = ranging)
        data["atr_slope"] = atr_slope(data["high"], data["low"], data["close"], 14, 5)
        
        # Calculate VWAP for exit (vwap expects DataFrame with OHLCV)
        data["vwap"] = vwap(data)
        
        # Regime gate: Only trade in ranging/choppy conditions
        # ADX < threshold OR Choppiness > threshold AND ATR slope flat
        data["range_regime"] = (
            ((data["adx"] < adx_threshold) | (data["choppiness"] > chop_threshold)) &
            (data["atr_slope"].abs() < 1.0)  # Flat ATR slope
        )
        
        # Mean reversion signals (use shifted values to avoid lookahead)
        prev_close = data["close"].shift(1)
        prev_upper = data["bb_upper"].shift(1)
        prev_lower = data["bb_lower"].shift(1)
        
        # Long: Close below lower band in range regime (oversold)
        data["long_signal"] = (
            (data["close"] < data["bb_lower"]) &
            (prev_close >= prev_lower) &  # Just crossed below
            data["range_regime"]
        )
        
        # Short: Close above upper band in range regime (overbought)
        data["short_signal"] = (
            (data["close"] > data["bb_upper"]) &
            (prev_close <= prev_upper) &  # Just crossed above
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
        # Stop: 1x ATR below lower band
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "bb_lower"] - data.loc[long_mask, "atr"]
        
        # Short entries
        short_mask = data["short_signal"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        # Stop: 1x ATR above upper band
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "bb_upper"] + data.loc[short_mask, "atr"]
        
        # Exit signals based on exit mode
        if exit_mode == "mid_band":
            # Exit long when price reaches middle band
            data.loc[(data["high"] >= data["bb_mid"]), "exit_signal"] = True
            # Exit short when price reaches middle band
            data.loc[(data["low"] <= data["bb_mid"]), "exit_signal"] = True
        elif exit_mode == "vwap":
            # Exit at VWAP
            data.loc[(data["high"] >= data["vwap"]), "exit_signal"] = True
            data.loc[(data["low"] <= data["vwap"]), "exit_signal"] = True
        # opposite_band mode: no exit signal, rely on stop or opposite signal
        
        # Store diagnostics for debug
        data["_diag_regime_pass"] = data["range_regime"]
        data["_diag_bb_cross_up"] = data["close"] > data["bb_upper"]
        data["_diag_bb_cross_down"] = data["close"] < data["bb_lower"]
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "BB Upper", "column": "bb_upper", "color": "red", "style": "dashed"},
            {"name": "BB Lower", "column": "bb_lower", "color": "green", "style": "dashed"},
            {"name": "BB Mid", "column": "bb_mid", "color": "blue", "style": "solid"},
        ]
