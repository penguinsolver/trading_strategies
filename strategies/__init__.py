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

# New strategies (batch 4 - promising)
from .keltner_breakout import KeltnerBreakoutStrategy
from .macd_divergence import MACDDivergenceStrategy
from .parabolic_sar import ParabolicSARStrategy
from .stochastic_momentum import StochasticMomentumStrategy
from .williams_r import WilliamsRStrategy
from .cci_momentum import CCIMomentumStrategy
from .ichimoku_cloud import IchimokuCloudStrategy
from .elder_ray import ElderRayStrategy
from .obv_divergence import OBVDivergenceStrategy

# New strategies (batch 5 - more promising)
from .pivot_point import PivotPointStrategy
from .trix_momentum import TRIXMomentumStrategy
from .aroon_trend import AroonTrendStrategy

# New strategies (batch 6)
from .force_index import ForceIndexStrategy
from .mfi_reversal import MFIReversionStrategy
from .ad_line import ADLineStrategy

# New strategies (batch 7)
from .ultimate_oscillator import UltimateOscillatorStrategy
from .dmi_cross import DMICrossStrategy
from .roc_momentum import ROCMomentumStrategy

# New strategies (batch 8)
from .hull_ma import HullMAStrategy
from .vortex import VortexStrategy
from .chaikin_oscillator import ChaikinOscillatorStrategy

# New strategies (batch 9-10 - final)
from .kst import KSTStrategy
from .coppock import CoppockCurveStrategy
from .ppo import PPOStrategy
from .macd_zero import MACDZeroCrossStrategy
from .rsi_divergence import RSIDivergenceStrategy
from .smi import SMIStrategy

# New strategies (batch 11 - optimized)
from .rsi_extreme import RSIExtremeBounceStrategy
from .tight_ema_scalp import TightEMAScalperStrategy
from .range_breakout import RangeBreakoutMomentumStrategy
from .ema_slope_momentum import EMASlopeMomentumStrategy
from .price_action import PriceActionReversalStrategy
from .momentum_burst import MomentumBurstStrategy

# New strategies (batch 12)
from .triple_ema import TripleEMAStrategy
from .candle_combo import CandlePatternComboStrategy
from .vwap_bounce import VWAPBounceStrategy
from .hl_breakout import HighLowBreakoutStrategy
from .rsi_bb_revert import MeanReversionRSIBBStrategy
from .quick_scalp import QuickMomentumScalpStrategy

# New strategies (batch 13)
from .atr_trend_rider import ATRTrendRiderStrategy
from .dual_tf_momentum import DualTimeframeMomentumStrategy
from .vol_contraction import VolatilityContractionStrategy
from .c2c_momentum import CloseToCloseMomentumStrategy
from .gap_fill import GapFillStrategy
from .range_revert import RangeReversionStrategy

# New strategies (batch 14)
from .strong_trend import StrongTrendOnlyStrategy
from .pullback_ema import PullbackToEMAStrategy
from .vol_weighted_trend import VolumeWeightedTrendStrategy
from .inside_bar_bo import InsideBarBreakoutStrategy
from .rsi_trending import RSITrendingStrategy
from .close_breakout import CloseBreakoutStrategy

# New strategies (batch 15)
from .quick_rsi_scalp import QuickRSIScalpStrategy
from .vol_spike import VolatilitySpikeStrategy
from .ema_ribbon import EMARibbonTrendStrategy
from .bounce_low import BounceFromLowStrategy
from .mom_continue import MomentumContinuationStrategy
from .simple_pa import SimplePriceActionStrategy

# New strategies (batch 16-17 - final)
from .fast_trend_scalp import FastTrendScalpStrategy
from .aggressive_bo import AggressiveBreakoutStrategy
from .micro_trend import MicroTrendStrategy
from .quick_reversal import QuickReversalStrategy
from .trend_simple import TrendFollowSimpleStrategy
from .doji_reversal import DojiReversalStrategy
from .bar_count import BarCountTrendStrategy
from .opening_move import OpeningMoveStrategy
from .fade_extreme import FadeExtremeStrategy
from .tight_range_break import TightRangeBreakStrategy
from .mom_filter import MomentumFilterStrategy
from .final_ema import FinalEMAOptimizedStrategy

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
    # Batch 4: Promising strategies
    "keltner_breakout": KeltnerBreakoutStrategy,
    "macd_divergence": MACDDivergenceStrategy,
    "parabolic_sar": ParabolicSARStrategy,
    "stochastic_momentum": StochasticMomentumStrategy,
    "williams_r": WilliamsRStrategy,
    "cci_momentum": CCIMomentumStrategy,
    "ichimoku_cloud": IchimokuCloudStrategy,
    "elder_ray": ElderRayStrategy,
    "obv_divergence": OBVDivergenceStrategy,
    # Batch 5: More promising
    "pivot_point": PivotPointStrategy,
    "trix_momentum": TRIXMomentumStrategy,
    "aroon_trend": AroonTrendStrategy,
    # Batch 6: Volume-based
    "force_index": ForceIndexStrategy,
    "mfi_reversal": MFIReversionStrategy,
    "ad_line": ADLineStrategy,
    # Batch 7: More momentum
    "ultimate_oscillator": UltimateOscillatorStrategy,
    "dmi_cross": DMICrossStrategy,
    "roc_momentum": ROCMomentumStrategy,
    # Batch 8: Trend indicators
    "hull_ma": HullMAStrategy,
    "vortex": VortexStrategy,
    "chaikin_oscillator": ChaikinOscillatorStrategy,
    # Batch 9-10: Final strategies
    "kst": KSTStrategy,
    "coppock": CoppockCurveStrategy,
    "ppo": PPOStrategy,
    "macd_zero": MACDZeroCrossStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
    "smi": SMIStrategy,
    # Batch 11: Optimized strategies
    "rsi_extreme": RSIExtremeBounceStrategy,
    "tight_ema_scalp": TightEMAScalperStrategy,
    "range_breakout": RangeBreakoutMomentumStrategy,
    "ema_slope_momentum": EMASlopeMomentumStrategy,
    "price_action": PriceActionReversalStrategy,
    "momentum_burst": MomentumBurstStrategy,
    # Batch 12: More strategies
    "triple_ema": TripleEMAStrategy,
    "candle_combo": CandlePatternComboStrategy,
    "vwap_bounce": VWAPBounceStrategy,
    "hl_breakout": HighLowBreakoutStrategy,
    "rsi_bb_revert": MeanReversionRSIBBStrategy,
    "quick_scalp": QuickMomentumScalpStrategy,
    # Batch 13: More strategies
    "atr_trend_rider": ATRTrendRiderStrategy,
    "dual_tf_momentum": DualTimeframeMomentumStrategy,
    "vol_contraction": VolatilityContractionStrategy,
    "c2c_momentum": CloseToCloseMomentumStrategy,
    "gap_fill": GapFillStrategy,
    "range_revert": RangeReversionStrategy,
    # Batch 14: Trend strategies
    "strong_trend": StrongTrendOnlyStrategy,
    "pullback_ema": PullbackToEMAStrategy,
    "vol_weighted_trend": VolumeWeightedTrendStrategy,
    "inside_bar_bo": InsideBarBreakoutStrategy,
    "rsi_trending": RSITrendingStrategy,
    "close_breakout": CloseBreakoutStrategy,
    # Batch 15: Mix strategies
    "quick_rsi_scalp": QuickRSIScalpStrategy,
    "vol_spike": VolatilitySpikeStrategy,
    "ema_ribbon": EMARibbonTrendStrategy,
    "bounce_low": BounceFromLowStrategy,
    "mom_continue": MomentumContinuationStrategy,
    "simple_pa": SimplePriceActionStrategy,
    # Batch 16-17: Final strategies
    "fast_trend_scalp": FastTrendScalpStrategy,
    "aggressive_bo": AggressiveBreakoutStrategy,
    "micro_trend": MicroTrendStrategy,
    "quick_reversal": QuickReversalStrategy,
    "trend_simple": TrendFollowSimpleStrategy,
    "doji_reversal": DojiReversalStrategy,
    "bar_count": BarCountTrendStrategy,
    "opening_move": OpeningMoveStrategy,
    "fade_extreme": FadeExtremeStrategy,
    "tight_range_break": TightRangeBreakStrategy,
    "mom_filter": MomentumFilterStrategy,
    "final_ema": FinalEMAOptimizedStrategy,
}

