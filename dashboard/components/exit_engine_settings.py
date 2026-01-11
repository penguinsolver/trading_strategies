"""
Exit Engine Settings component for Streamlit dashboard.
"""
import streamlit as st
from backtest import ExitEngineConfig


def render_exit_engine_settings() -> ExitEngineConfig:
    """
    Render Exit Engine settings panel and return config.
    
    Returns:
        ExitEngineConfig with user-selected parameters
    """
    with st.expander("⚙️ Exit Engine Settings", expanded=False):
        st.caption("Configure standardized exit rules applied to all strategies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Stop Settings**")
            atr_period = st.number_input(
                "ATR Period",
                min_value=5,
                max_value=50,
                value=14,
                step=1,
                help="ATR lookback period for volatility calculation",
            )
            hard_stop_mult = st.number_input(
                "Hard Stop (× ATR)",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.25,
                help="Hard stop distance as multiple of ATR",
            )
            cooldown_bars = st.number_input(
                "Cooldown (bars)",
                min_value=0,
                max_value=20,
                value=4,
                step=1,
                help="Bars to wait after stop-out before new entry",
            )
        
        with col2:
            st.markdown("**Trailing Stop**")
            trailing_type = st.selectbox(
                "Trailing Type",
                options=["chandelier_atr", "atr", "percent"],
                index=0,
                help="How trailing stop is calculated",
            )
            trailing_mult = st.number_input(
                "Trailing (× ATR)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Trailing stop distance as multiple of ATR",
            )
            trail_activation_r = st.number_input(
                "Activate Trail at (+R)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="R-multiple profit needed to activate trailing",
            )
        
        with col3:
            st.markdown("**Take Profit & Breakeven**")
            partial_tp_enabled = st.checkbox(
                "Partial TP (50% at +1R)",
                value=True,
                help="Take 50% profit at +1R",
            )
            breakeven_enabled = st.checkbox(
                "Move to Breakeven at +1R",
                value=True,
                help="Move stop to entry price at +1R profit",
            )
            
            st.markdown("**Time Stop (Optional)**")
            time_stop_bars = st.number_input(
                "Time Stop (bars)",
                min_value=0,
                max_value=100,
                value=0,
                step=10,
                help="Exit after X bars if unrealized < threshold (0 = disabled)",
            )
    
    return ExitEngineConfig(
        atr_period=atr_period,
        hard_stop_mult=hard_stop_mult,
        trailing_type=trailing_type,
        trailing_mult=trailing_mult,
        trail_activation_r=trail_activation_r,
        partial_tp_enabled=partial_tp_enabled,
        breakeven_enabled=breakeven_enabled,
        time_stop_bars=time_stop_bars,
        cooldown_bars=cooldown_bars,
    )


def display_config_summary(config: ExitEngineConfig) -> None:
    """Display a compact summary of the exit engine config."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 0.5rem 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
        <strong>Exit Engine:</strong> 
        Stop {config.hard_stop_mult}× ATR | Trail {config.trailing_mult}× ATR ({config.trailing_type}) | 
        {"Partial TP ✓" if config.partial_tp_enabled else "No Partial"} | 
        {"BE ✓" if config.breakeven_enabled else "No BE"} | 
        Cooldown {config.cooldown_bars}
    </div>
    """, unsafe_allow_html=True)
