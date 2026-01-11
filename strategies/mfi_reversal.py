"""
Strategy: MFI (Money Flow Index) Reversal

MFI is a volume-weighted RSI. Trade extremes for reversals.
Combines price and volume for better overbought/oversold signals.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Positive and negative money flow
    price_change = typical_price.diff()
    positive_flow = pd.Series(np.where(price_change > 0, raw_money_flow, 0), index=close.index)
    negative_flow = pd.Series(np.where(price_change < 0, raw_money_flow, 0), index=close.index)
    
    # Rolling sum
    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()
    
    # Money flow ratio
    mf_ratio = positive_sum / negative_sum.replace(0, np.nan)
    
    # MFI
    mfi_val = 100 - (100 / (1 + mf_ratio))
    return mfi_val


class MFIReversionStrategy(Strategy):
    """Money Flow Index reversal strategy."""
    
    @property
    def name(self) -> str:
        return "MFI Reversal"
    
    @property
    def description(self) -> str:
        return "Trade MFI extremes (>80 or <20) for volume-weighted overbought/oversold reversals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="period",
                label="MFI Period",
                param_type="int",
                default=14,
                min_value=7,
                max_value=21,
                step=2,
                help_text="MFI lookback period",
            ),
            ParamConfig(
                name="overbought",
                label="Overbought",
                param_type="int",
                default=80,
                min_value=70,
                max_value=90,
                step=5,
                help_text="Overbought level",
            ),
            ParamConfig(
                name="oversold",
                label="Oversold",
                param_type="int",
                default=20,
                min_value=10,
                max_value=30,
                step=5,
                help_text="Oversold level",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MFI signals."""
        period = self.params.get("period", 14)
        overbought = self.params.get("overbought", 80)
        oversold = self.params.get("oversold", 20)
        
        # Calculate MFI
        data["mfi"] = mfi(data["high"], data["low"], data["close"], data["volume"], period)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        prev_mfi = data["mfi"].shift(1)
        
        # Long: MFI rises above oversold
        long_cond = (data["mfi"] > oversold) & (prev_mfi <= oversold)
        
        # Short: MFI falls below overbought
        short_cond = (data["mfi"] < overbought) & (prev_mfi >= overbought)
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "low"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 1.5
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "high"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 1.5
        
        # Exit at midpoint
        mid = (overbought + oversold) / 2
        exit_long = (data["mfi"] > mid) & (prev_mfi <= mid)
        exit_short = (data["mfi"] < mid) & (prev_mfi >= mid)
        data.loc[(exit_long | exit_short).fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "MFI", "column": "mfi", "color": "teal", "style": "solid"},
        ]
