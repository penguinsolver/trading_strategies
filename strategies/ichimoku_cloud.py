"""
Strategy: Ichimoku Cloud Breakout

Classic Ichimoku Kinko Hyo cloud breakout strategy.
Uses Tenkan/Kijun crosses with cloud position confirmation.
"""
import pandas as pd
import numpy as np

from .base import Strategy, ParamConfig
from indicators import atr


def ichimoku_lines(high: pd.Series, low: pd.Series, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> dict:
    """Calculate Ichimoku lines."""
    # Tenkan-sen (Conversion Line)
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_b = ((high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    chikou = low.shift(-kijun_period)
    
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou
    }


class IchimokuCloudStrategy(Strategy):
    """Ichimoku Cloud breakout strategy."""
    
    @property
    def name(self) -> str:
        return "Ichimoku Cloud"
    
    @property
    def description(self) -> str:
        return "Trade Tenkan/Kijun crosses when price is above/below the Ichimoku cloud for strong trend signals."
    
    def get_param_config(self) -> list[ParamConfig]:
        return [
            ParamConfig(
                name="tenkan_period",
                label="Tenkan Period",
                param_type="int",
                default=9,
                min_value=7,
                max_value=14,
                step=1,
                help_text="Conversion line period",
            ),
            ParamConfig(
                name="kijun_period",
                label="Kijun Period",
                param_type="int",
                default=26,
                min_value=20,
                max_value=35,
                step=3,
                help_text="Base line period",
            ),
            ParamConfig(
                name="require_cloud",
                label="Require Cloud Break",
                param_type="bool",
                default=True,
                help_text="Only trade when price is above/below cloud",
            ),
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Ichimoku signals."""
        tenkan_period = self.params.get("tenkan_period", 9)
        kijun_period = self.params.get("kijun_period", 26)
        require_cloud = self.params.get("require_cloud", True)
        
        # Calculate Ichimoku
        ich = ichimoku_lines(data["high"], data["low"], tenkan_period, kijun_period, 52)
        data["tenkan"] = ich["tenkan"]
        data["kijun"] = ich["kijun"]
        data["senkou_a"] = ich["senkou_a"]
        data["senkou_b"] = ich["senkou_b"]
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        
        # Cloud boundaries
        cloud_top = data[["senkou_a", "senkou_b"]].max(axis=1)
        cloud_bottom = data[["senkou_a", "senkou_b"]].min(axis=1)
        
        # Cloud position
        above_cloud = data["close"] > cloud_top
        below_cloud = data["close"] < cloud_bottom
        
        # TK cross
        prev_tenkan = data["tenkan"].shift(1)
        prev_kijun = data["kijun"].shift(1)
        
        tk_cross_up = (data["tenkan"] > data["kijun"]) & (prev_tenkan <= prev_kijun)
        tk_cross_down = (data["tenkan"] < data["kijun"]) & (prev_tenkan >= prev_kijun)
        
        # Long condition
        if require_cloud:
            long_cond = tk_cross_up & above_cloud
            short_cond = tk_cross_down & below_cloud
        else:
            long_cond = tk_cross_up
            short_cond = tk_cross_down
        
        # Initialize signals
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        # Long entries
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "kijun"] - data.loc[long_mask, "atr"]
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2
        
        # Short entries
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "kijun"] + data.loc[short_mask, "atr"]
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2
        
        # Exit on opposite cross
        data.loc[tk_cross_down.fillna(False), "exit_signal"] = True
        data.loc[tk_cross_up.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "Tenkan", "column": "tenkan", "color": "blue", "style": "solid"},
            {"name": "Kijun", "column": "kijun", "color": "red", "style": "solid"},
            {"name": "Senkou A", "column": "senkou_a", "color": "green", "style": "dashed"},
            {"name": "Senkou B", "column": "senkou_b", "color": "red", "style": "dashed"},
        ]
