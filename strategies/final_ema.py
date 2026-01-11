"""Strategy: Final EMA Optimized - Optimized 10/30 EMA cross."""
import pandas as pd
import numpy as np
from .base import Strategy, ParamConfig
from indicators import atr, ema
from indicators.moving_averages import crossover, crossunder

class FinalEMAOptimizedStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Final EMA"
    
    @property
    def description(self) -> str:
        return "Optimized 10/30 EMA cross with momentum filter."
    
    def get_param_config(self) -> list[ParamConfig]:
        return []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)
        data["ema10"] = ema(data["close"], 10)
        data["ema30"] = ema(data["close"], 30)
        
        golden = crossover(data["ema10"], data["ema30"])
        death = crossunder(data["ema10"], data["ema30"])
        
        # Add momentum filter
        bullish_mom = data["close"] > data["close"].shift(3)
        bearish_mom = data["close"] < data["close"].shift(3)
        
        long_cond = golden & bullish_mom
        short_cond = death & bearish_mom
        
        data["entry_signal"] = 0
        data["exit_signal"] = False
        data["stop_price"] = np.nan
        data["trailing_stop_atr"] = np.nan
        
        long_mask = long_cond.fillna(False)
        data.loc[long_mask, "entry_signal"] = 1
        data.loc[long_mask, "stop_price"] = data.loc[long_mask, "close"] - data.loc[long_mask, "atr"] * 1.5
        data.loc[long_mask, "trailing_stop_atr"] = data.loc[long_mask, "atr"] * 2.0
        
        short_mask = short_cond.fillna(False)
        data.loc[short_mask, "entry_signal"] = -1
        data.loc[short_mask, "stop_price"] = data.loc[short_mask, "close"] + data.loc[short_mask, "atr"] * 1.5
        data.loc[short_mask, "trailing_stop_atr"] = data.loc[short_mask, "atr"] * 2.0
        
        data.loc[death.fillna(False), "exit_signal"] = True
        data.loc[golden.fillna(False), "exit_signal"] = True
        
        return data
    
    def get_indicator_info(self) -> list[dict]:
        return [
            {"name": "EMA 10", "column": "ema10", "color": "green", "style": "solid"},
            {"name": "EMA 30", "column": "ema30", "color": "red", "style": "solid"},
        ]
