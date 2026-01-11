"""
Strategy: Anchored VWAP Trend Pullback

Trend pullback strategy using Anchored VWAP as support/resistance.
Entry: Pullback to AVWAP and reclaim in trend direction
Exit: Partial at 1R, trail remainder
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr, ema


class AVWAPPullbackStrategy(Strategy):
    """Anchored VWAP trend pullback strategy."""
    
    @property
    def name(self) -> str:
        return "AVWAP Pullback"
    
    @property
    def description(self) -> str:
        return "Trend pullback to Anchored VWAP. Enters on pullback + reclaim pattern, not mean reversion."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="anchor",
                label="Anchor Type",
                param_type="select",
                default="daily",
                options=["daily", "weekly"],
                help_text="Anchor VWAP to daily or weekly open",
            ),
            ParamConfig(
                name="reclaim_bars",
                label="Reclaim Bars",
                param_type="int",
                default=3,
                min_value=2,
                max_value=5,
                step=1,
                help_text="Bars required to confirm reclaim",
            ),
            ParamConfig(
                name="atr_stop_mult",
                label="ATR Stop Multiple",
                param_type="float",
                default=1.5,
                min_value=1.0,
                max_value=2.5,
                step=0.5,
                help_text="ATR multiplier for stop loss",
            ),
            ParamConfig(
                name="trend_ema",
                label="Trend EMA Period",
                param_type="int",
                default=50,
                min_value=20,
                max_value=100,
                step=10,
                help_text="EMA period for trend filter",
            ),
        ]
    
    def _calculate_avwap(self, data: pd.DataFrame, anchor: str) -> pd.Series:
        """Calculate Anchored VWAP based on anchor type."""
        if not isinstance(data.index, pd.DatetimeIndex):
            # Fallback to simple VWAP if no datetime index
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            return (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()
        
        # Calculate typical price * volume
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        tpv = typical_price * data["volume"]
        
        avwap = pd.Series(np.nan, index=data.index)
        
        if anchor == "daily":
            # Group by date
            data["date"] = data.index.date
            for date_val in data["date"].unique():
                mask = data["date"] == date_val
                cum_tpv = tpv[mask].cumsum()
                cum_vol = data.loc[mask, "volume"].cumsum()
                avwap.loc[mask] = cum_tpv / cum_vol
        else:  # weekly
            # Group by week
            data["week"] = data.index.isocalendar().week.astype(str) + "-" + data.index.year.astype(str)
            for week_val in data["week"].unique():
                mask = data["week"] == week_val
                cum_tpv = tpv[mask].cumsum()
                cum_vol = data.loc[mask, "volume"].cumsum()
                avwap.loc[mask] = cum_tpv / cum_vol
        
        return avwap
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate AVWAP pullback signals."""
        # Parameters
        anchor = self.params.get("anchor", "daily")
        reclaim_bars = self.params.get("reclaim_bars", 3)
        atr_stop_mult = self.params.get("atr_stop_mult", 1.5)
        trend_ema = self.params.get("trend_ema", 50)
        
        # Calculate AVWAP
        data["avwap"] = self._calculate_avwap(data, anchor)
        
        # Calculate trend EMA
        data["trend_ema"] = ema(data["close"], trend_ema)
        
        # Calculate ATR
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Trend filter
        data["uptrend"] = data["close"] > data["trend_ema"]
        data["downtrend"] = data["close"] < data["trend_ema"]
        
        # Detect pullback to AVWAP
        # For longs: Price was above AVWAP, dipped to touch AVWAP, now closing back above
        data["touched_avwap_from_above"] = (data["low"] <= data["avwap"]) & (data["close"].shift(1) > data["avwap"].shift(1))
        data["touched_avwap_from_below"] = (data["high"] >= data["avwap"]) & (data["close"].shift(1) < data["avwap"].shift(1))
        
        # Track recent touches
        data["recent_touch_above"] = data["touched_avwap_from_above"].rolling(window=reclaim_bars, min_periods=1).sum() > 0
        data["recent_touch_below"] = data["touched_avwap_from_below"].rolling(window=reclaim_bars, min_periods=1).sum() > 0
        
        # Reclaim: Close back above/below AVWAP after touch
        data["reclaim_long"] = (
            data["uptrend"] &
            data["recent_touch_above"] &
            (data["close"] > data["avwap"]) &
            (data["close"].shift(1) <= data["avwap"].shift(1))
        )
        
        data["reclaim_short"] = (
            data["downtrend"] &
            data["recent_touch_below"] &
            (data["close"] < data["avwap"]) &
            (data["close"].shift(1) >= data["avwap"].shift(1))
        )
        
        # Initialize signal columns
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = data["reclaim_long"].fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        # Stop below recent low or ATR
        recent_low = data["low"].rolling(window=reclaim_bars + 1).min()
        data.loc[long_mask, "stop_price"] = np.minimum(
            recent_low.loc[long_mask],
            data.loc[long_mask, "close"] - (data.loc[long_mask, "atr"] * atr_stop_mult)
        )
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * atr_stop_mult
        
        # Short entries
        short_mask = data["reclaim_short"].fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        recent_high = data["high"].rolling(window=reclaim_bars + 1).max()
        data.loc[short_mask, "stop_price"] = np.maximum(
            recent_high.loc[short_mask],
            data.loc[short_mask, "close"] + (data.loc[short_mask, "atr"] * atr_stop_mult)
        )
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * atr_stop_mult
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "AVWAP", "column": "avwap", "color": "purple", "style": "solid"},
            {"name": "Trend EMA", "column": "trend_ema", "color": "blue", "style": "dashed"},
        ]
