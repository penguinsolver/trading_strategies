"""
Strategy: Strong Trend Only

Only trade when trend is exceptionally strong (ADX > 40).
High selectivity for best trends.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, adx, ema


class StrongTrendOnlyStrategy(Strategy):
    """Strong trend only strategy."""
    
    @property
    def name(self) -> str:
        return "Strong Trend Only"
    
    @property
    def description(self) -> str:
        return "Trade only when ADX > 40 indicates exceptionally strong trends."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="adx_threshold",
                label="Min ADX",
                param_type="int",
                default=40,
                min_value=30,
                max_value=50,
                step=5,
                help_text="Minimum ADX for strong trend",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate strong trend signals."""
        adx_threshold = self.params.get("adx_threshold", 40)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], 14)
        data["adx"] = adx_val
        data["plus_di"] = plus_di
        data["minus_di"] = minus_di
        data["ema20"] = ema(data["close"], 20)
        
        # Strong uptrend
        strong_up = (adx_val > adx_threshold) & (plus_di > minus_di)
        # Strong downtrend
        strong_down = (adx_val > adx_threshold) & (minus_di > plus_di)
        
        prev_strong_up = strong_up.shift(1).fillna(False)
        prev_strong_down = strong_down.shift(1).fillna(False)
        
        # Entry on strong trend start
        long_cond = strong_up & (~prev_strong_up)
        short_cond = strong_down & (~prev_strong_down)
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "ema20"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "ema20"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.0
        
        # Exit when ADX weakens
        weak_adx = adx_val < 25
        data.loc[weak_adx.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [{"name": "EMA 20", "column": "ema20", "color": "blue", "style": "solid"}]
