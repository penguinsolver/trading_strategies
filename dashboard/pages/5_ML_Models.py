"""
ML & Statistical Models page - Ensemble, Regime Detection, and ML strategies.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from data import CandleFetcher
from backtest import BacktestEngine, ExitEngineBacktester, ExitEngineConfig, CostModel
from strategies import STRATEGIES
from models.ensemble import EnsembleStrategy, EnsembleConfig, create_ensemble_from_results
from models.regime_filter import RegimeFilter, RegimeConfig, RegimeAwareStrategy

# Statistical & ML Models
try:
    from models.hmm_regime import HMMRegimeDetector, HMMConfig, HMMFilteredStrategy
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

try:
    from models.kalman_filter import KalmanTrendFilter, KalmanConfig, KalmanStrategy
    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False

try:
    from models.garch_sizing import GARCHVolatilitySizer, GARCHConfig, GARCHSizedStrategy
    HAS_GARCH = True
except ImportError:
    HAS_GARCH = False

try:
    from models.xgboost_classifier import XGBoostTradeClassifier, XGBConfig, XGBoostFilteredStrategy
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from models.ml_signal_generator import MLSignalGenerator, MLSignalConfig, MultiModelEnsemble
    HAS_ML_GENERATOR = True
except ImportError:
    HAS_ML_GENERATOR = False

try:
    from models.additional_strategies import (
        MomentumStrategy, ADXTrendStrategy, BreakoutStrategy,
        OptimizedMACrossover, DualMomentumStrategy, TrendFollowingSystem
    )
    from models.advanced_models import (
        StackingEnsemble, NeuralNetworkModel, VotingEnsembleModel,
        MeanReversionStrategy, HybridMACrossover, AdvancedConfig
    )
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


def render_ml_models_page():
    """Render the ML Models page."""
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">🧠 ML & Statistical Models</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Ensemble voting, Regime filtering, and ML-based strategies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different model types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Ensemble Voting", 
        "📈 Regime Filter", 
        "📉 Statistical Models",
        "🤖 ML Classifier",
        "🚀 ML Signal Generator",
        "🏆 Strategy Comparison"
    ])
    
    with tab1:
        render_ensemble_tab()
    
    with tab2:
        render_regime_tab()
    
    with tab3:
        render_statistical_tab()
    
    with tab4:
        render_ml_tab()
    
    with tab5:
        render_ml_signal_generator_tab()
    
    with tab6:
        render_strategy_comparison_tab()


def render_ensemble_tab():
    """Render the ensemble voting tab."""
    
    st.markdown("### 📊 Ensemble Voting Strategy")
    st.caption("Combine top-performing strategies with voting consensus")
    
    # Settings
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        window = st.selectbox(
            "📅 Time Window",
            options=["24h", "7d", "14d", "30d", "90d", "180d"],
            index=3,  # 30d
            key="ens_window",
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Candle Interval",
            options=["5m", "15m", "1h", "4h"],
            index=1,  # 15m
            key="ens_interval",
        )
    
    with col3:
        capital = st.number_input(
            "💰 Initial Capital ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            key="ens_capital",
        )
    
    with col4:
        risk = st.slider(
            "⚡ Risk per Trade (%)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            key="ens_risk",
        ) / 100
    
    # Ensemble settings
    st.markdown("---")
    st.markdown("### Ensemble Configuration")
    
    # Signal mode selection
    signal_mode = st.radio(
        "Signal Mode",
        options=["best", "consensus", "any"],
        index=0,  # Default to "best" for reliable trades
        format_func=lambda x: {
            "best": "🏆 Best Strategy (uses #1 performer)",
            "consensus": "🎯 Consensus (require agreement)",
            "any": "⚡ Any Signal (more trades)"
        }.get(x, x),
        horizontal=True,
        key="ens_signal_mode",
        help="**Best**: Use the top-performing strategy directly. **Consensus**: Require threshold % agreement. **Any**: Use any signal."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Top N Strategies", 3, 15, 5, key="ens_top_n")
    
    with col2:
        if signal_mode == "consensus":
            threshold = st.slider("Voting Threshold (%)", 30, 80, 50, 5, key="ens_threshold") / 100
        else:
            threshold = 0.0  # Not used in "best" or "any" mode
            st.caption(f"Threshold not used in '{signal_mode.title()}' mode")
    
    with col3:
        if signal_mode in ["consensus", "any"]:
            min_signals = st.slider("Min Signals Required", 1, 5, 1, key="ens_min_signals")
        else:
            min_signals = 1
            st.caption("Uses single best strategy")
    
    use_exit_engine = st.checkbox("Apply Exit Engine (stops + trailing)", value=True, key="ens_exit")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Run Ensemble Backtest", type="primary", use_container_width=True, key="ens_run"):
        run_ensemble_backtest(window, interval, capital, risk, top_n, threshold, min_signals, use_exit_engine, signal_mode)
    
    # Show results
    if "ensemble_results" in st.session_state and st.session_state.ensemble_results:
        display_ensemble_results()


def run_ensemble_backtest(window, interval, capital, risk, top_n, threshold, min_signals, use_exit_engine, signal_mode="consensus"):
    """Run ensemble backtest."""
    
    with st.spinner("📡 Fetching data..."):
        try:
            fetcher = CandleFetcher(coin="BTC", use_cache=True)
            data = fetcher.fetch_data(interval=interval, window=window)
            
            if data.empty:
                st.error("No data received.")
                return
            
            st.success(f"✅ Fetched {len(data)} candles")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return
    
    cost_model = CostModel(maker_fee=0.0001, taker_fee=0.00035, slippage_bps=1.0)
    
    # First, run all individual strategies to find top performers
    with st.spinner("🔍 Analyzing individual strategies to find top performers..."):
        individual_results = {}
        progress = st.progress(0)
        
        strategy_keys = list(STRATEGIES.keys())
        for i, key in enumerate(strategy_keys):
            progress.progress((i + 1) / len(strategy_keys))
            
            try:
                strategy = STRATEGIES[key]()
                engine = BacktestEngine(cost_model=cost_model)
                result = engine.run(strategy, data.copy(), capital, risk)
                individual_results[key] = result
            except:
                pass
        
        progress.empty()
    
    # Sort by performance and get top N
    performance = {k: v.metrics.net_return for k, v in individual_results.items() if v is not None}
    sorted_keys = sorted(performance.keys(), key=lambda k: performance[k], reverse=True)
    top_keys = sorted_keys[:top_n]
    
    # Show top performers
    st.markdown(f"### Top {top_n} Performers Selected:")
    cols = st.columns(min(top_n, 5))
    for i, key in enumerate(top_keys[:5]):
        with cols[i]:
            ret = performance[key]
            color = "#22c55e" if ret > 0 else "#ef4444"
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: white; border-radius: 8px; border: 2px solid {color};">
                <div style="font-weight: 600; font-size: 0.85rem;">{key.replace('_', ' ').title()}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {color};">{ret:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Create ensemble
    with st.spinner("🧠 Running ensemble backtest..."):
        top_strategies = [STRATEGIES[key]() for key in top_keys]
        
        config = EnsembleConfig(
            signal_mode=signal_mode,
            threshold=threshold,
            min_signals=min_signals,
            weight_mode="performance",  # Use performance-based weights for "best" mode
        )
        
        ensemble = EnsembleStrategy(top_strategies, config)
        
        # Set performance weights - use strategy.name to match signal_matrix columns
        # Map key to strategy.name for proper matching
        key_to_name = {key: STRATEGIES[key]().name for key in top_keys}
        top_performance_by_name = {key_to_name[k]: performance[k] for k in top_keys}
        ensemble.set_weights_from_performance(top_performance_by_name)
        
        # Run backtest
        if use_exit_engine:
            exit_config = ExitEngineConfig()
            engine = ExitEngineBacktester(config=exit_config, cost_model=cost_model)
        else:
            engine = BacktestEngine(cost_model=cost_model)
        
        ensemble_result = engine.run(ensemble, data.copy(), capital, risk)
    
    # Store results
    st.session_state.ensemble_results = {
        "ensemble": ensemble_result,
        "individual": individual_results,
        "top_keys": top_keys,
        "settings": {
            "window": window,
            "interval": interval,
            "capital": capital,
            "top_n": top_n,
            "threshold": threshold,
        }
    }
    st.rerun()


def display_ensemble_results():
    """Display ensemble backtest results."""
    
    results = st.session_state.ensemble_results
    ensemble_result = results["ensemble"]
    individual_results = results["individual"]
    top_keys = results["top_keys"]
    settings = results["settings"]
    
    st.markdown("---")
    st.markdown("### 🎯 Ensemble Results")
    
    m = ensemble_result.metrics
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        color = "#22c55e" if m.net_return > 0 else "#ef4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 2px solid {color};">
            <div style="font-size: 0.8rem; color: #64748b;">Net Return</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {color};">{m.net_return:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0;">
            <div style="font-size: 0.8rem; color: #64748b;">Win Rate</div>
            <div style="font-size: 1.5rem; font-weight: 700;">{m.win_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0;">
            <div style="font-size: 0.8rem; color: #64748b;">Total Trades</div>
            <div style="font-size: 1.5rem; font-weight: 700;">{m.total_trades}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pf = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "∞"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0;">
            <div style="font-size: 0.8rem; color: #64748b;">Profit Factor</div>
            <div style="font-size: 1.5rem; font-weight: 700;">{pf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 2px solid #ef4444;">
            <div style="font-size: 0.8rem; color: #64748b;">Max Drawdown</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{m.max_drawdown:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison with individual strategies
    st.markdown("### 📊 Ensemble vs Individual Strategies")
    
    comparison_data = []
    
    # Add ensemble
    comparison_data.append({
        "Strategy": "🧠 ENSEMBLE",
        "Return": f"{m.net_return:+.2f}%",
        "Win Rate": f"{m.win_rate:.1f}%",
        "Trades": m.total_trades,
        "Max DD": f"{m.max_drawdown:.1f}%",
        "PF": f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "∞",
    })
    
    # Add top individual strategies
    for key in top_keys:
        if key in individual_results and individual_results[key]:
            im = individual_results[key].metrics
            comparison_data.append({
                "Strategy": key.replace("_", " ").title(),
                "Return": f"{im.net_return:+.2f}%",
                "Win Rate": f"{im.win_rate:.1f}%",
                "Trades": im.total_trades,
                "Max DD": f"{im.max_drawdown:.1f}%",
                "PF": f"{im.profit_factor:.2f}" if im.profit_factor != float('inf') else "∞",
            })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Equity curve
    st.markdown("### 📈 Equity Curve")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Ensemble curve
    fig.add_trace(go.Scatter(
        x=ensemble_result.equity_curve.index,
        y=ensemble_result.equity_curve.values,
        name="Ensemble",
        line=dict(color="#10b981", width=3),
    ))
    
    # Add top 3 individual curves
    colors = ["#3b82f6", "#f59e0b", "#8b5cf6"]
    for i, key in enumerate(top_keys[:3]):
        if key in individual_results and individual_results[key]:
            result = individual_results[key]
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name=key.replace("_", " ").title(),
                line=dict(color=colors[i], width=1.5, dash="dot"),
            ))
    
    fig.add_hline(y=settings["capital"], line_dash="dash", line_color="#94a3b8")
    
    fig.update_layout(
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        xaxis_title="",
        yaxis_title="Equity ($)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_regime_tab():
    """Render the regime filter tab."""
    
    st.markdown("### 📈 Regime-Aware Trading")
    st.caption("Filter signals based on market regime (trending vs ranging)")
    
    st.info("🚧 **Coming Soon**: Run strategies with ADX-based regime filtering. Only trade when market conditions match strategy type.")
    
    # Preview of regime detection
    st.markdown("#### How It Works:")
    st.markdown("""
    1. **ADX > 25**: Trending market → Use trend-following strategies
    2. **ADX < 20**: Ranging market → Use mean-reversion strategies
    3. **Volatility Expanding**: High volatility → Reduce position size
    """)


def render_statistical_tab():
    """Render the statistical models tab."""
    
    st.markdown("### 📉 Statistical Models")
    st.caption("HMM Regime Detection, Kalman Filter, and GARCH Volatility Sizing")
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window = st.selectbox(
            "📅 Time Window",
            options=["30d", "60d", "90d", "180d"],
            index=2,  # 90d
            key="stat_window",
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Candle Interval",
            options=["15m", "1h", "4h"],
            index=1,  # 1h
            key="stat_interval",
        )
    
    with col3:
        base_strategy_key = st.selectbox(
            "📈 Base Strategy",
            options=list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x]().name,
            key="stat_base_strategy",
        )
    
    # Model selection
    st.markdown("#### Select Models to Apply:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_hmm = st.checkbox(
            "🔮 HMM Regime Detection",
            value=True,
            help="Use Hidden Markov Model to detect market regimes",
            disabled=not HAS_HMM,
        )
    
    with col2:
        use_kalman = st.checkbox(
            "📐 Kalman Trend Filter",
            value=True,
            help="Use Kalman filter for adaptive trend detection",
            disabled=not HAS_KALMAN,
        )
    
    with col3:
        use_garch = st.checkbox(
            "📊 GARCH Volatility Sizing",
            value=True,
            help="Use GARCH model for dynamic position sizing",
            disabled=not HAS_GARCH,
        )
    
    if st.button("🚀 Run Statistical Models Backtest", key="stat_run"):
        run_statistical_backtest(window, interval, base_strategy_key, use_hmm, use_kalman, use_garch)
    
    # Display results
    if "stat_results" in st.session_state:
        display_statistical_results()


def run_statistical_backtest(window, interval, base_strategy_key, use_hmm, use_kalman, use_garch):
    """Run backtest with statistical model enhancements."""
    
    with st.spinner("📡 Fetching data..."):
        try:
            fetcher = CandleFetcher(coin="BTC", use_cache=True)
            data = fetcher.fetch_data(interval=interval, window=window)
            
            if data.empty:
                st.error("No data received.")
                return
            
            st.success(f"✅ Fetched {len(data)} candles")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return
    
    capital = 10000
    risk = 2.0
    
    # Get base strategy
    base_strategy = STRATEGIES[base_strategy_key]()
    
    results = {}
    cost_model = CostModel()
    engine = BacktestEngine(cost_model=cost_model)
    
    # Run base strategy
    with st.spinner(f"Running {base_strategy.name}..."):
        base_result = engine.run(base_strategy, data.copy(), capital, risk)
        results["base"] = {"name": base_strategy.name, "result": base_result}
    
    # HMM Regime filtered
    if use_hmm and HAS_HMM:
        with st.spinner("Running HMM Regime Detection..."):
            try:
                hmm_detector = HMMRegimeDetector(HMMConfig(
                    trade_in_bull=True,
                    trade_in_sideways=True,
                    trade_in_bear=False,
                ))
                hmm_strategy = HMMFilteredStrategy(STRATEGIES[base_strategy_key](), hmm_detector)
                hmm_result = engine.run(hmm_strategy, data.copy(), capital, risk)
                results["hmm"] = {"name": f"HMM_{base_strategy.name}", "result": hmm_result}
            except Exception as e:
                st.warning(f"HMM failed: {e}")
    
    # Kalman filtered
    if use_kalman and HAS_KALMAN:
        with st.spinner("Running Kalman Trend Filter..."):
            try:
                kalman_strategy = KalmanStrategy(STRATEGIES[base_strategy_key](), filter_mode=True)
                kalman_result = engine.run(kalman_strategy, data.copy(), capital, risk)
                results["kalman"] = {"name": f"Kalman_{base_strategy.name}", "result": kalman_result}
            except Exception as e:
                st.warning(f"Kalman failed: {e}")
    
    # GARCH sized
    if use_garch and HAS_GARCH:
        with st.spinner("Running GARCH Volatility Sizing..."):
            try:
                garch_strategy = GARCHSizedStrategy(STRATEGIES[base_strategy_key]())
                garch_result = engine.run(garch_strategy, data.copy(), capital, risk)
                results["garch"] = {"name": f"GARCH_{base_strategy.name}", "result": garch_result}
            except Exception as e:
                st.warning(f"GARCH failed: {e}")
    
    st.session_state.stat_results = results
    st.rerun()


def display_statistical_results():
    """Display statistical model results."""
    results = st.session_state.stat_results
    
    st.markdown("### 📊 Statistical Model Results")
    
    # Build comparison table
    rows = []
    for key, data in results.items():
        result = data["result"]
        metrics = result.metrics
        rows.append({
            "Strategy": data["name"],
            "Net Return": f"{metrics.net_return:+.2f}%",
            "Total Trades": metrics.total_trades,
            "Win Rate": f"{metrics.win_rate:.1f}%",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Max Drawdown": f"{metrics.max_drawdown:.2f}%",
        })
    
    df = pd.DataFrame(rows)
    
    # Highlight best ROI
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Check for 15% target
    for key, data in results.items():
        roi = data["result"].metrics.net_return
        if roi >= 15:  # Already in percentage (e.g., 15 means 15%)
            st.success(f"🎯 **{data['name']}** achieves {roi:.1f}% ROI - meets 15% target!")


def render_ml_tab():
    """Render the ML models tab."""
    
    st.markdown("### 🤖 XGBoost ML Classifier")
    st.caption("Predict profitable trades using strategy signals and market features")
    
    if not HAS_XGBOOST:
        st.error("❌ XGBoost not installed. Run: `pip install xgboost scikit-learn`")
        return
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window = st.selectbox(
            "📅 Time Window",
            options=["30d", "60d", "90d", "180d"],
            index=2,  # 90d
            key="ml_window",
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Candle Interval",
            options=["15m", "1h", "4h"],
            index=1,  # 1h
            key="ml_interval",
        )
    
    with col3:
        base_strategy_key = st.selectbox(
            "📈 Base Strategy",
            options=list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x]().name,
            key="ml_base_strategy",
        )
    
    # ML parameters
    st.markdown("#### XGBoost Parameters:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_proba = st.slider(
            "Min Probability",
            min_value=0.5,
            max_value=0.8,
            value=0.55,
            step=0.05,
            help="Only take trades where P(profit) >= this threshold",
        )
    
    with col2:
        n_estimators = st.slider(
            "# Estimators",
            min_value=50,
            max_value=300,
            value=100,
            step=50,
        )
    
    with col3:
        forward_bars = st.slider(
            "Look-Ahead Bars",
            min_value=6,
            max_value=24,
            value=12,
            help="How many bars to look ahead for profitability target",
        )
    
    # Number of strategies to use as features
    n_strategies = st.slider(
        "# Strategies for Features",
        min_value=5,
        max_value=30,
        value=10,
        help="How many strategy signals to use as features",
    )
    
    if st.button("🚀 Train & Backtest XGBoost", key="ml_run"):
        run_xgboost_backtest(window, interval, base_strategy_key, min_proba, n_estimators, forward_bars, n_strategies)
    
    # Display results
    if "ml_results" in st.session_state:
        display_ml_results()


def run_xgboost_backtest(window, interval, base_strategy_key, min_proba, n_estimators, forward_bars, n_strategies):
    """Run backtest with XGBoost ML filtering."""
    
    with st.spinner("📡 Fetching data..."):
        try:
            fetcher = CandleFetcher(coin="BTC", use_cache=True)
            data = fetcher.fetch_data(interval=interval, window=window)
            
            if data.empty:
                st.error("No data received.")
                return
            
            st.success(f"✅ Fetched {len(data)} candles")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return
    
    capital = 10000
    risk = 2.0
    
    # Get base strategy
    base_strategy = STRATEGIES[base_strategy_key]()
    
    results = {}
    cost_model = CostModel()
    engine = BacktestEngine(cost_model=cost_model)
    
    # Run base strategy
    with st.spinner(f"Running {base_strategy.name}..."):
        base_result = engine.run(base_strategy, data.copy(), capital, risk)
        results["base"] = {"name": base_strategy.name, "result": base_result}
    
    # Get subset of strategies for features
    strategy_keys = list(STRATEGIES.keys())[:n_strategies]
    all_strategies = [STRATEGIES[k]() for k in strategy_keys]
    
    # Train XGBoost
    with st.spinner("Training XGBoost classifier..."):
        try:
            config = XGBConfig(
                min_probability=min_proba,
                n_estimators=n_estimators,
                forward_return_bars=forward_bars,
            )
            classifier = XGBoostTradeClassifier(config)
            metrics = classifier.train(data.copy(), all_strategies)
            
            st.info(f"📈 Training Accuracy: {metrics['train_accuracy']:.1%}, Validation Accuracy: {metrics['val_accuracy']:.1%}")
        except Exception as e:
            st.error(f"Training failed: {e}")
            return
    
    # Run XGBoost filtered strategy
    with st.spinner("Running XGBoost-filtered strategy..."):
        try:
            xgb_strategy = XGBoostFilteredStrategy(
                STRATEGIES[base_strategy_key](),
                classifier,
                all_strategies,
                auto_train=False,
            )
            xgb_strategy._is_trained = True  # Already trained
            xgb_result = engine.run(xgb_strategy, data.copy(), capital, risk)
            results["xgboost"] = {"name": f"XGB_{base_strategy.name}", "result": xgb_result}
        except Exception as e:
            st.error(f"XGBoost backtest failed: {e}")
    
    # Store feature importance
    st.session_state.ml_results = results
    st.session_state.ml_feature_importance = classifier.get_feature_importance(top_n=15)
    st.session_state.ml_training_metrics = metrics
    st.rerun()


def display_ml_results():
    """Display ML model results."""
    results = st.session_state.ml_results
    
    st.markdown("### 📊 XGBoost Results")
    
    # Build comparison table
    rows = []
    for key, data in results.items():
        result = data["result"]
        metrics = result.metrics
        rows.append({
            "Strategy": data["name"],
            "Net Return": f"{metrics.net_return:+.2f}%",
            "Total Trades": metrics.total_trades,
            "Win Rate": f"{metrics.win_rate:.1f}%",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Max Drawdown": f"{metrics.max_drawdown:.2f}%",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Check for 15% target
    for key, data in results.items():
        roi = data["result"].metrics.net_return
        if roi >= 15:  # Already in percentage (e.g., 15 means 15%)
            st.success(f"🎯 **{data['name']}** achieves {roi:.1f}% ROI - meets 15% target!")
    
    # Feature importance
    if "ml_feature_importance" in st.session_state:
        st.markdown("#### 📊 Top Feature Importance")
        importance = st.session_state.ml_feature_importance
        
        if importance:
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in importance.items()
            ])
            st.bar_chart(imp_df.set_index("Feature"), height=300)


def render_ml_signal_generator_tab():
    """Render the ML Signal Generator tab."""
    
    st.markdown("### 🚀 ML Signal Generator")
    st.caption("Generate trading signals directly using ML models (not just filtering)")
    
    if not HAS_ML_GENERATOR:
        st.error("❌ ML Signal Generator not available. Check dependencies.")
        return
    
    # Info box
    st.info("""
    **This is different from ML Classifier:**
    - ML Classifier: Filters existing strategy signals
    - ML Signal Generator: Generates signals directly from 50+ technical features
    """)
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window = st.selectbox(
            "📅 Time Window",
            options=["60d", "90d", "120d", "180d"],
            index=1,  # 90d
            key="mlgen_window",
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Candle Interval",
            options=["15m", "1h", "4h"],
            index=1,  # 1h
            key="mlgen_interval",
        )
    
    with col3:
        model_type = st.selectbox(
            "🤖 Model Type",
            options=["xgboost", "random_forest", "ensemble"],
            format_func=lambda x: {"xgboost": "XGBoost", "random_forest": "Random Forest", "ensemble": "Multi-Model Ensemble"}[x],
            key="mlgen_model_type",
        )
    
    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            forward_bars = st.slider(
                "Look-Ahead Bars",
                min_value=6,
                max_value=24,
                value=12,
                help="Bars to look ahead for profitability label",
                key="mlgen_forward_bars",
            )
            
            profit_threshold = st.slider(
                "Profit Threshold (%)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Minimum forward return for 'profitable' label",
                key="mlgen_profit_threshold",
            )
        
        with col2:
            min_probability = st.slider(
                "Min Probability",
                min_value=0.50,
                max_value=0.75,
                value=0.55,
                step=0.05,
                help="Minimum probability to generate signal",
                key="mlgen_min_prob",
            )
            
            n_estimators = st.slider(
                "# Estimators",
                min_value=50,
                max_value=300,
                value=200,
                step=50,
                key="mlgen_n_estimators",
            )
    
    if st.button("🚀 Train & Backtest ML Signal Generator", key="mlgen_run"):
        run_ml_signal_generator(
            window, interval, model_type,
            forward_bars, profit_threshold / 100,  # Convert to decimal
            min_probability, n_estimators
        )
    
    # Display results
    if "mlgen_results" in st.session_state:
        display_ml_generator_results()


def run_ml_signal_generator(window, interval, model_type, forward_bars, profit_threshold, min_probability, n_estimators):
    """Run ML Signal Generator training and backtest."""
    
    with st.spinner("📡 Fetching data..."):
        try:
            fetcher = CandleFetcher(coin="BTC", use_cache=True)
            data = fetcher.fetch_data(interval=interval, window=window)
            
            if data.empty:
                st.error("No data received.")
                return
            
            st.success(f"✅ Fetched {len(data)} candles")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return
    
    capital = 10000
    risk = 2.0
    
    results = {}
    cost_model = CostModel()
    engine = BacktestEngine(cost_model=cost_model)
    
    # Train/test split
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.copy()  # Use all data, model only trained on first 70%
    
    # Create ML Signal Generator
    with st.spinner(f"Training {model_type.upper()} Signal Generator..."):
        try:
            config = MLSignalConfig(
                model_type=model_type,
                forward_bars=forward_bars,
                profit_threshold=profit_threshold,
                loss_threshold=-profit_threshold,
                min_probability=min_probability,
                n_estimators=n_estimators,
            )
            
            if model_type == "ensemble":
                ml_strategy = MultiModelEnsemble(
                    model_types=["xgboost", "random_forest"],
                    min_agreement=2,
                    config=config,
                )
            else:
                ml_strategy = MLSignalGenerator(config)
            
            # Train
            training_metrics = ml_strategy.train(train_data)
            
            st.info(f"📈 Walk-Forward Validation Accuracy: {training_metrics.get('val_accuracy_mean', 0):.1%}")
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Run backtest
    with st.spinner("Running ML Signal Generator backtest..."):
        try:
            ml_result = engine.run(ml_strategy, test_data.copy(), capital, risk)
            results["ml_generator"] = {"name": ml_strategy.name, "result": ml_result}
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Compare with MA Crossover
    with st.spinner("Running MA Crossover for comparison..."):
        try:
            ma_strategy = STRATEGIES["ma_crossover"]()
            ma_result = engine.run(ma_strategy, data.copy(), capital, risk)
            results["ma_crossover"] = {"name": ma_strategy.name, "result": ma_result}
        except Exception as e:
            st.warning(f"MA Crossover comparison failed: {e}")
    
    # Store results
    st.session_state.mlgen_results = results
    st.session_state.mlgen_training_metrics = training_metrics
    
    if hasattr(ml_strategy, 'get_feature_importance'):
        st.session_state.mlgen_feature_importance = ml_strategy.get_feature_importance(top_n=15)
    
    st.rerun()


def display_ml_generator_results():
    """Display ML Signal Generator results."""
    results = st.session_state.mlgen_results
    
    st.markdown("### 📊 ML Signal Generator Results")
    
    # Build comparison table
    rows = []
    for key, data in results.items():
        result = data["result"]
        metrics = result.metrics
        rows.append({
            "Strategy": data["name"],
            "Net Return": f"{metrics.net_return:+.2f}%",
            "Total Trades": metrics.total_trades,
            "Win Rate": f"{metrics.win_rate:.1f}%",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Max Drawdown": f"{metrics.max_drawdown:.2f}%",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Compare with MA Crossover
    ml_roi = results.get("ml_generator", {}).get("result", None)
    ma_roi = results.get("ma_crossover", {}).get("result", None)
    
    if ml_roi and ma_roi:
        ml_return = ml_roi.metrics.net_return
        ma_return = ma_roi.metrics.net_return
        
        if ml_return > ma_return:
            diff = ml_return - ma_return
            st.success(f"🎉 **ML Signal Generator beats MA Crossover by {diff:.2f}%!**")
        else:
            diff = ma_return - ml_return
            st.warning(f"⚠️ MA Crossover is still {diff:.2f}% ahead. Try tuning parameters.")
    
    # Check for 15% target
    for key, data in results.items():
        roi = data["result"].metrics.net_return
        if roi >= 15:
            st.success(f"🎯 **{data['name']}** achieves {roi:.1f}% ROI - meets 15% target!")
    
    # Training metrics
    if "mlgen_training_metrics" in st.session_state:
        st.markdown("#### 📈 Training Metrics")
        metrics = st.session_state.mlgen_training_metrics
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Validation Accuracy", f"{metrics.get('val_accuracy_mean', 0):.1%}")
        with col2:
            st.metric("# Features", metrics.get('n_features', 0))
        with col3:
            st.metric("# Training Samples", metrics.get('n_samples', 0))
    
    # Feature importance
    if "mlgen_feature_importance" in st.session_state:
        st.markdown("#### 📊 Top Feature Importance")
        importance = st.session_state.mlgen_feature_importance
        
        if importance:
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in importance.items()
            ])
            st.bar_chart(imp_df.set_index("Feature"), height=300)


def render_strategy_comparison_tab():
    """Render the strategy comparison tab with all models."""
    
    st.markdown("### 🏆 Strategy Comparison")
    st.caption("Compare all ML, Statistical, and Technical strategies")
    
    if not HAS_ADVANCED:
        st.warning("⚠️ Advanced models not available. Install required packages.")
        return
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window = st.selectbox("Time Window", ["30d", "60d", "90d", "180d"], index=2, key="strat_window")
    
    with col2:
        interval = st.selectbox("Candle Interval", ["1h", "4h", "1d"], index=0, key="strat_interval")
    
    with col3:
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5, key="strat_risk") / 100
    
    if st.button("🚀 Run Strategy Comparison", type="primary", key="run_strat_comparison"):
        with st.spinner("Running all strategies..."):
            run_strategy_comparison(window, interval, risk)
    
    # Display results
    if "strat_comparison_results" in st.session_state:
        display_strategy_comparison_results()


def run_strategy_comparison(window: str, interval: str, risk: float):
    """Run all strategies and store results."""
    import warnings
    warnings.filterwarnings("ignore")
    
    fetcher = CandleFetcher(coin="BTC", use_cache=True)
    data = fetcher.fetch_data(interval=interval, window=window)
    
    if data.empty:
        st.error("Failed to fetch data")
        return
    
    engine = BacktestEngine(cost_model=CostModel())
    capital = 10000
    results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    # All strategies to test
    all_strategies = []
    
    # Built-in technical strategies
    for key, strat_class in STRATEGIES.items():
        all_strategies.append((f"📈 {strat_class().name}", strat_class()))
    
    # Additional strategies
    if HAS_ADVANCED:
        all_strategies.extend([
            ("🔥 Momentum", MomentumStrategy()),
            ("📊 ADX Trend", ADXTrendStrategy()),
            ("🚀 Breakout", BreakoutStrategy()),
            ("⚡ Optimized MA", OptimizedMACrossover()),
            ("📉 Dual Momentum", DualMomentumStrategy()),
            ("📈 Trend System", TrendFollowingSystem()),
            ("🔀 Mean Reversion", MeanReversionStrategy()),
            ("🤝 Hybrid MA+ML", HybridMACrossover()),
        ])
    
    # ML models (train on first 70%)
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    
    if HAS_ADVANCED:
        try:
            stacking = StackingEnsemble(AdvancedConfig(min_probability=0.45))
            stacking.train(train_data)
            all_strategies.append(("🧠 Stacking Ensemble", stacking))
        except:
            pass
        
        try:
            nn = NeuralNetworkModel(AdvancedConfig(min_probability=0.45))
            nn.train(train_data)
            all_strategies.append(("🤖 Neural Network", nn))
        except:
            pass
        
        try:
            voting = VotingEnsembleModel(AdvancedConfig(min_probability=0.45))
            voting.train(train_data)
            all_strategies.append(("🗳️ Voting Ensemble", voting))
        except:
            pass
    
    total = len(all_strategies)
    
    for i, (name, strat) in enumerate(all_strategies):
        status.text(f"Testing {name}...")
        progress.progress((i + 1) / total)
        
        try:
            result = engine.run(strat, data.copy(), capital, risk)
            results.append({
                "Strategy": name,
                "ROI %": result.metrics.net_return,
                "Trades": result.metrics.total_trades,
                "Win Rate %": result.metrics.win_rate,
                "Profit Factor": result.metrics.profit_factor,
                "Max DD %": result.metrics.max_drawdown,
            })
        except Exception as e:
            results.append({
                "Strategy": name,
                "ROI %": 0,
                "Trades": 0,
                "Win Rate %": 0,
                "Profit Factor": 0,
                "Max DD %": 0,
            })
    
    progress.empty()
    status.empty()
    
    # Sort by ROI
    results.sort(key=lambda x: x["ROI %"], reverse=True)
    st.session_state.strat_comparison_results = results


def display_strategy_comparison_results():
    """Display strategy comparison results."""
    results = st.session_state.strat_comparison_results
    
    if not results:
        return
    
    # Find baseline (MA Crossover)
    baseline_roi = 0
    for r in results:
        if "Ma Crossover" in r["Strategy"] or "MA_CROSSOVER" in r["Strategy"].upper():
            baseline_roi = r["ROI %"]
            break
    
    st.markdown("#### 📊 Results Ranked by ROI")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add beat indicator
    df["Beat MA?"] = df["ROI %"].apply(
        lambda x: "✅ YES" if x > baseline_roi else ("🎯 BASELINE" if x == baseline_roi else "❌")
    )
    
    # Format for display
    display_df = df.copy()
    display_df["ROI %"] = display_df["ROI %"].apply(lambda x: f"{x:+.2f}%")
    display_df["Win Rate %"] = display_df["Win Rate %"].apply(lambda x: f"{x:.1f}%")
    display_df["Profit Factor"] = display_df["Profit Factor"].apply(lambda x: f"{x:.2f}")
    display_df["Max DD %"] = display_df["Max DD %"].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Highlight winners
    winners = [r for r in results if r["ROI %"] > baseline_roi]
    
    if winners:
        st.success(f"🎉 **{len(winners)} strategies beat MA Crossover!**")
        
        cols = st.columns(min(len(winners), 3))
        for i, w in enumerate(winners[:3]):
            with cols[i]:
                diff = w["ROI %"] - baseline_roi
                st.metric(
                    w["Strategy"],
                    f"{w['ROI %']:+.2f}%",
                    f"+{diff:.2f}% vs MA"
                )
    else:
        st.info(f"MA Crossover ({baseline_roi:+.2f}%) remains the top performer")
    
    # Top 5 bar chart
    st.markdown("#### 📊 Top 5 Strategies")
    top5 = pd.DataFrame(results[:5])
    st.bar_chart(top5.set_index("Strategy")["ROI %"])


# Run the page
render_ml_models_page()
