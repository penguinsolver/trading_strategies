"""Trading strategies module."""
from .base import Strategy, ParamConfig
from .trend_pullback import TrendPullbackStrategy
from .breakout import BreakoutStrategy
from .vwap_reversion import VWAPReversionStrategy
from .ma_crossover import MACrossoverStrategy

# New strategies (batch 1)
from .supertrend import SupertrendStrategy
from .donchian_turtle import DonchianTurtleStrategy
from .rsi2_dip import RSI2DipStrategy
from .bb_squeeze import BBSqueezeStrategy
from .inside_bar import InsideBarStrategy
from .orb import ORBStrategy
from .breakout_retest import BreakoutRetestStrategy
from .regime_switcher import RegimeSwitcherStrategy

# New strategies (batch 2 - selective/low-frequency)
from .atr_channel import ATRChannelStrategy
from .volume_breakout import VolumeBreakoutStrategy
from .zscore_reversion import ZScoreReversionStrategy
from .chandelier_trend import ChandelierTrendStrategy
from .avwap_pullback import AVWAPPullbackStrategy
from .regression_slope import RegressionSlopeStrategy

# New strategies (batch 3 - anti-chop)
from .bb_mean_reversion import BBMeanReversionStrategy
from .prev_day_range import PrevDayRangeStrategy
from .ts_momentum import TSMomentumStrategy

# Wrappers
from .wrappers import TimeFilteredStrategy, VolatilityTargetedStrategy

# Registry of available strategies
STRATEGIES = {
    # Original strategies
    "trend_pullback": TrendPullbackStrategy,
    "breakout": BreakoutStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "ma_crossover": MACrossoverStrategy,
    # Batch 1: Diverse strategies
    "supertrend": SupertrendStrategy,
    "donchian_turtle": DonchianTurtleStrategy,
    "rsi2_dip": RSI2DipStrategy,
    "bb_squeeze": BBSqueezeStrategy,
    "inside_bar": InsideBarStrategy,
    "orb": ORBStrategy,
    "breakout_retest": BreakoutRetestStrategy,
    "regime_switcher": RegimeSwitcherStrategy,
    # Batch 2: Selective/low-frequency strategies
    "atr_channel": ATRChannelStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    "zscore_reversion": ZScoreReversionStrategy,
    "chandelier_trend": ChandelierTrendStrategy,
    "avwap_pullback": AVWAPPullbackStrategy,
    "regression_slope": RegressionSlopeStrategy,
    # Batch 3: Anti-chop strategies
    "bb_mean_reversion": BBMeanReversionStrategy,
    "prev_day_range": PrevDayRangeStrategy,
    "ts_momentum": TSMomentumStrategy,
}
