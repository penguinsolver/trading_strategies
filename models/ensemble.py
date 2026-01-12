"""
Ensemble Strategy - Combines multiple strategies with voting logic.

Trades when a threshold of strategies agree on direction.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting."""
    
    # Signal mode: 
    # - "consensus": require threshold agreement
    # - "any": use any signal from min_signals strategies
    # - "best": use the best-performing strategy's signals directly (no voting)
    signal_mode: str = "consensus"  # "consensus", "any", or "best"
    
    # Voting threshold (fraction of strategies that must agree) - only for consensus mode
    threshold: float = 0.6  # 60% must agree
    
    # Weighting mode
    weight_mode: str = "equal"  # "equal", "performance", "custom"
    
    # If using custom weights, map strategy name to weight
    custom_weights: dict = field(default_factory=dict)
    
    # Minimum signals required to trade (even if threshold met)
    min_signals: int = 1


class EnsembleStrategy:
    """
    Meta-strategy that combines multiple strategies using voting.
    
    Generates entry signals when a threshold of constituent strategies
    agree on direction. This can reduce whipsaws and false signals.
    """
    
    def __init__(
        self,
        strategies: List["Strategy"],
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Initialize ensemble.
        
        Args:
            strategies: List of strategy instances to combine
            config: Ensemble configuration
        """
        self.strategies = strategies
        self.config = config or EnsembleConfig()
        self._weights: Optional[dict] = None
    
    @property
    def name(self) -> str:
        return f"Ensemble({len(self.strategies)})"
    
    @property
    def description(self) -> str:
        strategy_names = [s.name for s in self.strategies[:3]]
        if len(self.strategies) > 3:
            strategy_names.append(f"...+{len(self.strategies)-3} more")
        return f"Ensemble voting: {', '.join(strategy_names)}"
    
    def set_weights_from_performance(self, performance_dict: dict) -> None:
        """
        Set strategy weights based on historical performance.
        
        Args:
            performance_dict: Dict of strategy_name -> net_return
        """
        # Convert to positive weights (shift so minimum is 0)
        min_perf = min(performance_dict.values())
        shifted = {k: v - min_perf + 0.01 for k, v in performance_dict.items()}
        
        # Normalize to sum to 1
        total = sum(shifted.values())
        self._weights = {k: v / total for k, v in shifted.items()}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate consensus signals from all strategies.
        
        Signal is generated when threshold of strategies agree.
        """
        # Collect signals from all strategies
        signal_matrix = pd.DataFrame(index=data.index)
        strategy_dataframes = {}  # Store full DataFrames for "best" mode
        
        for strategy in self.strategies:
            try:
                df_copy = data.copy()
                df_with_signals = strategy.generate_signals(df_copy)
                signal_col = df_with_signals.get("entry_signal", pd.Series(0, index=data.index))
                signal_matrix[strategy.name] = signal_col
                strategy_dataframes[strategy.name] = df_with_signals  # Store full DataFrame
            except Exception as e:
                # Strategy failed, use zeros
                signal_matrix[strategy.name] = 0
                strategy_dataframes[strategy.name] = None
        
        # Get weights
        if self.config.weight_mode == "custom" and self.config.custom_weights:
            weights = self.config.custom_weights
        elif self.config.weight_mode == "performance" and self._weights:
            weights = self._weights
        else:
            # Equal weights
            weights = {s.name: 1.0 / len(self.strategies) for s in self.strategies}
        
        # Calculate weighted votes
        long_votes = pd.Series(0.0, index=data.index)
        short_votes = pd.Series(0.0, index=data.index)
        total_weight = pd.Series(0.0, index=data.index)
        
        for strategy in self.strategies:
            signal = signal_matrix[strategy.name]
            weight = weights.get(strategy.name, 1.0 / len(self.strategies))
            
            # Count votes
            long_votes += (signal > 0).astype(float) * weight
            short_votes += (signal < 0).astype(float) * weight
            
            # Track which strategies voted (for min_signals check)
            total_weight += ((signal != 0).astype(float)) * weight
        
        # Count raw signals for min_signals check
        n_long_signals = (signal_matrix > 0).sum(axis=1)
        n_short_signals = (signal_matrix < 0).sum(axis=1)
        
        # Generate final signal based on mode
        final_signal = pd.Series(0, index=data.index)
        
        if self.config.signal_mode == "best":
            # BEST mode: Use the best-performing (highest weight) strategy's signals
            # Find the strategy with highest weight
            best_strategy = max(weights.keys(), key=lambda k: weights.get(k, 0))
            
            # Get the full DataFrame from the best strategy
            if best_strategy in strategy_dataframes and strategy_dataframes[best_strategy] is not None:
                best_df = strategy_dataframes[best_strategy]
                
                # Copy ALL signal columns from the best strategy
                signal_columns = ['entry_signal', 'exit_signal', 'stop_price']
                for col in signal_columns:
                    if col in best_df.columns:
                        data[col] = best_df[col]
                
                final_signal = data.get('entry_signal', pd.Series(0, index=data.index))
            elif best_strategy in signal_matrix.columns:
                # Fallback to just entry_signal
                final_signal = signal_matrix[best_strategy].copy()
            
        elif self.config.signal_mode == "any":
            # ANY mode: Use signal from ANY strategy (weighted priority)
            # Take signal if at least min_signals agree, regardless of threshold
            long_condition = (n_long_signals >= self.config.min_signals)
            short_condition = (n_short_signals >= self.config.min_signals)
            
            # If both long and short have signals, use the one with more votes
            both = long_condition & short_condition
            long_wins = both & (n_long_signals > n_short_signals)
            short_wins = both & (n_short_signals >= n_long_signals)
            
            # Clear conflicts
            long_condition = long_condition & ~short_wins
            short_condition = short_condition & ~long_wins
            
            final_signal[long_condition] = 1
            final_signal[short_condition] = -1
            
        else:
            # CONSENSUS mode: Require threshold agreement (original behavior)
            # Long when:
            # 1. Weighted long votes >= threshold
            # 2. At least min_signals strategies signaled long
            long_condition = (
                (long_votes >= self.config.threshold) &
                (n_long_signals >= self.config.min_signals)
            )
            
            # Short when:
            # 1. Weighted short votes >= threshold
            # 2. At least min_signals strategies signaled short
            short_condition = (
                (short_votes >= self.config.threshold) &
                (n_short_signals >= self.config.min_signals)
            )
            
            # If both conditions met (shouldn't happen normally), use 0
            conflict = long_condition & short_condition
            long_condition = long_condition & ~conflict
            short_condition = short_condition & ~conflict
            
            final_signal[long_condition] = 1
            final_signal[short_condition] = -1
        
        # Add to data
        data["entry_signal"] = final_signal
        data["ensemble_long_votes"] = long_votes
        data["ensemble_short_votes"] = short_votes
        data["ensemble_agreement"] = (long_votes + short_votes).clip(0, 1)
        
        return data


class TopNEnsemble(EnsembleStrategy):
    """
    Ensemble that automatically selects top N strategies by performance.
    """
    
    def __init__(
        self,
        all_strategies: List["Strategy"],
        top_n: int = 5,
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Args:
            all_strategies: Full list of available strategies
            top_n: Number of top performers to include
            config: Ensemble configuration
        """
        self.all_strategies = all_strategies
        self.top_n = top_n
        self._performance_cache: dict = {}
        
        # Initialize with first N strategies (will be updated)
        super().__init__(all_strategies[:top_n], config)
    
    @property
    def name(self) -> str:
        return f"TopN_Ensemble({self.top_n})"
    
    def update_top_strategies(self, performance_dict: dict) -> None:
        """
        Update ensemble to use top N strategies by performance.
        
        Args:
            performance_dict: Dict of strategy_name -> net_return
        """
        # Sort by performance
        sorted_strategies = sorted(
            performance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top N names
        top_names = [name for name, _ in sorted_strategies[:self.top_n]]
        
        # Filter strategies
        name_to_strategy = {s.name: s for s in self.all_strategies}
        self.strategies = [
            name_to_strategy[name]
            for name in top_names
            if name in name_to_strategy
        ]
        
        # Set weights based on performance
        top_performance = {name: perf for name, perf in sorted_strategies[:self.top_n]}
        self.set_weights_from_performance(top_performance)


def create_ensemble_from_results(results: dict, top_n: int = 5, threshold: float = 0.6) -> EnsembleStrategy:
    """
    Create an ensemble from backtest results.
    
    Args:
        results: Dict of strategy_key -> BacktestResult
        top_n: Number of top performers to include
        threshold: Voting threshold
    
    Returns:
        EnsembleStrategy configured with top performers
    """
    from strategies import STRATEGIES
    
    # Extract performance
    performance = {}
    for key, result in results.items():
        if result is not None:
            performance[key] = result.metrics.net_return
    
    # Sort and get top N
    sorted_keys = sorted(performance.keys(), key=lambda k: performance[k], reverse=True)
    top_keys = sorted_keys[:top_n]
    
    # Create strategy instances
    strategies = []
    for key in top_keys:
        if key in STRATEGIES:
            strategies.append(STRATEGIES[key]())
    
    # Create ensemble
    config = EnsembleConfig(threshold=threshold)
    ensemble = EnsembleStrategy(strategies, config)
    
    # Set weights based on performance
    top_performance = {k: performance[k] for k in top_keys}
    ensemble.set_weights_from_performance(top_performance)
    
    return ensemble
