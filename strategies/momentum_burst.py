"""
Strategy: Momentum Burst

Trade explosive momentum moves with volume confirmation.
Only enter when both price and volume spike together.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, sma


class MomentumBurstStrategy(Strategy):
    """Momentum burst with volume spike."""
    
    @property
    def name(self) -> str:
        return "Momentum Burst"
    
    @property
    def description(self) -> str:
        return "Trade explosive price moves confirmed by volume spikes."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="price_mult",
                label="Price Move (ATR)",
                param_type="float",
                default=1.5,
                min_value=1.0,
                max_value=2.5,
                step=0.5,
                help_text="Min price move in ATR",
            ),
            ParamConfig(
                name="vol_mult",
                label="Volume Spike",
                param_type="float",
                default=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                help_text="Volume above average",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum burst signals."""
        price_mult = self.params.get("price_mult", 1.5)
        vol_mult = self.params.get("vol_mult", 2.0)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["vol_avg"] = sma(data["volume"], 20)
        
        # Price move size
        data["move"] = (data["close"] - data["close"].shift(1)).abs()
        
        # Conditions
        big_up = (data["close"] > data["close"].shift(1)) & (data["move"] > data["atr"] * price_mult)
        big_down = (data["close"] < data["close"].shift(1)) & (data["move"] > data["atr"] * price_mult)
        high_vol = data["volume"] > data["vol_avg"] * vol_mult
        
        long_cond = big_up & high_vol
        short_cond = big_down & high_vol
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.0
        
        # Exit on momentum fade (volume drops)
        vol_fade = data["volume"] < data["vol_avg"]
        data.loc[vol_fade.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
