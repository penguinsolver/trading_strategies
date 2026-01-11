"""
BTC Active Trading Lab - Home Page

A polished, professional Streamlit dashboard with clean UI.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st


# Page configuration
st.set_page_config(
    page_title="BTC Active Trading Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS styling
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: #f8fafc;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1e293b;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .hero h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .hero p {
        margin: 0.75rem 0 0 0;
        font-size: 1.15rem;
        opacity: 0.9;
    }
    
    /* Stat cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: #ffffff;
        padding: 1.75rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
        border-color: #667eea;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Strategy cards */
    .strategy-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .strategy-card {
        background: #ffffff;
        padding: 1.75rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .strategy-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border-color: #667eea;
    }
    
    .strategy-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.25rem;
        color: #1e293b;
    }
    
    .strategy-card p {
        margin: 0 0 1rem 0;
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-green {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-yellow {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-blue {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .badge-purple {
        background: #f3e8ff;
        color: #7c3aed;
    }
    
    .badge-red {
        background: #fee2e2;
        color: #dc2626;
    }
    
    .badge-gray {
        background: #f1f5f9;
        color: #475569;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        margin: 2.5rem 0 1.5rem 0;
    }
    
    .section-header h2 {
        margin: 0;
        font-size: 1.5rem;
        color: #1e293b;
        font-weight: 700;
    }
    
    .section-header .icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
    }
    
    /* Quick start */
    .step-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .step-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1e293b;
    }
    
    .step-card p {
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Button improvements */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button[kind="secondary"] {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        color: #1e293b;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: #667eea;
        color: #667eea;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main home page."""
    
    # Sidebar navigation
    st.sidebar.markdown("### 📍 Navigation")
    st.sidebar.page_link("app.py", label="🏠 Home")
    st.sidebar.page_link("pages/1_Trading_Dashboard.py", label="📈 Trading Dashboard")
    st.sidebar.page_link("pages/2_Strategy_Guide.py", label="📚 Strategy Guide")
    st.sidebar.page_link("pages/3_Compare_Strategies.py", label="⚔️ Compare All")
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### 💡 Quick Tips")
    st.sidebar.info("""
    **New here?**
    1. Start with the Strategy Guide
    2. Try MA Crossover first
    3. Use 7d window for robust results
    """)
    
    # Hero section
    st.markdown("""
    <div class="hero">
        <h1>📈 BTC Active Trading Lab</h1>
        <p>Backtest and compare trading strategies on Hyperliquid BTC perpetual</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Start Trading", type="primary", use_container_width=True):
            st.switch_page("pages/1_Trading_Dashboard.py")
    
    with col2:
        if st.button("⚔️ Compare All Strategies", type="secondary", use_container_width=True):
            st.switch_page("pages/3_Compare_Strategies.py")
    
    with col3:
        if st.button("📚 Strategy Guide", type="secondary", use_container_width=True):
            st.switch_page("pages/2_Strategy_Guide.py")
    
    # Stats section
    st.markdown("""
    <div class="section-header">
        <span class="icon">🎯</span>
        <h2>What You Can Do</h2>
    </div>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">Strategies</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">2</div>
            <div class="stat-label">Time Windows</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">15+</div>
            <div class="stat-label">Metrics</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">∞</div>
            <div class="stat-label">Backtests</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategies section
    st.markdown("""
    <div class="section-header">
        <span class="icon">📊</span>
        <h2>Available Strategies</h2>
    </div>
    """, unsafe_allow_html=True)
    
    strat_col1, strat_col2 = st.columns(2)
    
    with strat_col1:
        st.markdown("""
        <div class="strategy-card">
            <h3>📈 Trend Pullback</h3>
            <p>Multi-timeframe trend following with pullback entries. Follow the big trend, enter on retracements.</p>
            <div>
                <span class="badge badge-green">Trending Markets</span>
                <span class="badge badge-yellow">⭐ Recommended</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try Trend Pullback →", key="trend", use_container_width=True):
            st.switch_page("pages/1_Trading_Dashboard.py")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="strategy-card">
            <h3>🔄 VWAP Reversion</h3>
            <p>Mean reversion to Volume-Weighted Average Price in ranging markets. Fade extremes.</p>
            <div>
                <span class="badge badge-purple">Ranging Markets</span>
                <span class="badge badge-red">⚠️ Counter-trend</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try VWAP Reversion →", key="vwap", use_container_width=True):
            st.switch_page("pages/1_Trading_Dashboard.py")
    
    with strat_col2:
        st.markdown("""
        <div class="strategy-card">
            <h3>🚀 Breakout</h3>
            <p>Donchian channel breakout with volatility filter. Catch momentum when price breaks range.</p>
            <div>
                <span class="badge badge-blue">Volatile Markets</span>
                <span class="badge badge-purple">High Momentum</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try Breakout →", key="breakout", use_container_width=True):
            st.switch_page("pages/1_Trading_Dashboard.py")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="strategy-card">
            <h3>📊 MA Crossover</h3>
            <p>Simple EMA crossover baseline. Classic golden cross / death cross for learning.</p>
            <div>
                <span class="badge badge-gray">Baseline</span>
                <span class="badge badge-green">Learning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try MA Crossover →", key="ma", use_container_width=True):
            st.switch_page("pages/1_Trading_Dashboard.py")
    
    # Quick start section
    st.markdown("""
    <div class="section-header">
        <span class="icon">🚀</span>
        <h2>Quick Start</h2>
    </div>
    """, unsafe_allow_html=True)
    
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">1</div>
            <h4>Choose a Strategy</h4>
            <p>Pick based on your market outlook. Trending → Trend Pullback. Ranging → VWAP Reversion. Learning → MA Crossover.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step_col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">2</div>
            <h4>Configure & Run</h4>
            <p>Set your capital, risk per trade, and time window. Click Run Backtest and watch the analysis unfold.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step_col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">3</div>
            <h4>Analyze Results</h4>
            <p>Review metrics, equity curve, and trade ledger. Compare strategies using the comparison tool.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <strong>BTC Active Trading Lab</strong> • Data from Hyperliquid Public API • For educational purposes only<br>
        <small>Past performance does not guarantee future results. Always do your own research.</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
