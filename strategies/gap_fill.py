"""
Strategy: Gap Fill

Trade gaps that fill back to previous close.
Gaps often retrace before continuing.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


class GapFillStrategy(Strategy):
    """Gap fill strategy."""
    
    @property
    def name(self) -> str:
        return "Gap Fill"
    
    @property
    def description(self) -> str:
        return "Trade gaps that are likely to fill back to previous close."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="min_gap",
                label="Min Gap (ATR)",
                param_type="float",
                default=0.5,
                min_value=0.25,
                max_value=1.5,
                step=0.25,
                help_text="Minimum gap size in ATR",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate gap fill signals."""
        min_gap_atr = self.params.get("min_gap", 0.5)
        
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Gap calculation
        data["gap"] = data["open"] - data["close"].shift(1)
        min_gap = data["atr"] * min_gap_atr
        
        # Gap up = short (expect fill), gap down = long
        gap_up = data["gap"] > min_gap
        gap_down = data["gap"] < -min_gap
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Short gap ups (fade the gap)
        short_mask = gap_up.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "open"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 0.5
        
        # Long gap downs (fade the gap)
        long_mask = gap_down.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "open"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 0.5
        
        # Exit when gap filled (close returns to prev close)
        prev_close = data["close"].shift(1)
        gap_filled_up = (data["close"] <= prev_close) & gap_up.shift(0)
        gap_filled_down = (data["close"] >= prev_close) & gap_down.shift(0)
        data.loc[(gap_filled_up | gap_filled_down).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return []
