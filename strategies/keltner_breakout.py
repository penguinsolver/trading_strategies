"""
Strategy: Keltner Channel Breakout

Breakout strategy using Keltner Channels (EMA + ATR bands).
More adaptive than Bollinger Bands due to ATR-based bands.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import ema, atr, adx


class KeltnerBreakoutStrategy(Strategy):
    """Keltner Channel breakout with trend confirmation."""
    
    @property
    def name(self) -> str:
        return "Keltner Breakout"
    
    @property
    def description(self) -> str:
        return "Trade breakouts from Keltner Channels with ATR-based volatility bands and ADX trend filter."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="ema_period",
                label="EMA Period",
                param_type="int",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                help_text="EMA period for channel midline",
            ),
            ParamConfig(
                name="atr_mult",
                label="ATR Multiplier",
                param_type="float",
                default=2.0,
                min_value=1.0,
                max_value=3.0,
                step=0.5,
                help_text="ATR multiplier for channel width",
            ),
            ParamConfig(
                name="adx_min",
                label="Min ADX",
                param_type="int",
                default=20,
                min_value=10,
                max_value=35,
                step=5,
                help_text="Minimum ADX for trend confirmation",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Keltner Channel breakout signals."""
        ema_period = self.params.get("ema_period", 20)
        atr_mult = self.params.get("atr_mult", 2.0)
        adx_min = self.params.get("adx_min", 20)
        
        # Keltner Channel calculation
        data["kc_mid"] = ema(data["close"], ema_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["kc_upper"] = data["kc_mid"] + (data["atr"] * atr_mult)
        data["kc_lower"] = data["kc_mid"] - (data["atr"] * atr_mult)
        
        # ADX for trend strength
        adx_val, _, _ = adx(data["high"], data["low"], data["close"], 14)
        data["adx"] = adx_val
        
        # Breakout signals with shift to avoid lookahead
        prev_close = data["close"].shift(1)
        prev_upper = data["kc_upper"].shift(1)
        prev_lower = data["kc_lower"].shift(1)
        
        # Long: Close breaks above upper band with strong ADX
        long_cond = (
            (data["close"] > data["kc_upper"]) &
            (prev_close <= prev_upper) &
            (data["adx"] > adx_min)
        )
        
        # Short: Close breaks below lower band with strong ADX
        short_cond = (
            (data["close"] < data["kc_lower"]) &
            (prev_close >= prev_lower) &
            (data["adx"] > adx_min)
        )
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "kc_mid"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "kc_mid"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit at midline
        data.loc[data["close"] < data["kc_mid"], "exit_signal"] = True  # Exit longs
        data.loc[data["close"] > data["kc_mid"], "exit_signal"] = True  # Exit shorts
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "KC Upper", "column": "kc_upper", "color": "red", "style": "dashed"},
            {"name": "KC Lower", "column": "kc_lower", "color": "green", "style": "dashed"},
            {"name": "KC Mid", "column": "kc_mid", "color": "blue", "style": "solid"},
        ]
