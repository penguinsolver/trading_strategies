"""
Strategy Guide page with explanations and parameter recommendations.
"""
import streamlit as st


def render_strategy_guide():
    """Render the strategy guide content."""
    
    st.title("📚 Strategy Guide")
    st.caption("Learn about each strategy and get started with recommended parameters")
    
    st.divider()
    
    # Quick Start Section
    st.header("🚀 Quick Start")
    st.markdown("""
    **New to active trading?** Here's the recommended path:
    
    | Level | Strategy | Why |
    |-------|----------|-----|
    | 1️⃣ Beginner | MA Crossover | Simple baseline to understand the backtester |
    | 2️⃣ Intermediate | Trend Pullback | Good risk management, works in trends |
    | 3️⃣ Advanced | Regime Switcher | Adapts to market conditions |
    | 4️⃣ Selective | Chandelier Trend | Low frequency, high quality setups |
    
    > 💡 **Tip**: Start with 7-day backtest, then adjust one parameter at a time.
    """)
    
    st.divider()
    
    # ========================================
    # SECTION 1: ORIGINAL STRATEGIES
    # ========================================
    st.header("📦 Original Strategies")
    
    # Trend Pullback
    with st.expander("📈 Trend Pullback", expanded=False):
        st.markdown("""
        **Concept**: Follow the dominant trend, enter on pullbacks rather than chasing.
        
        **How it works**:
        - Uses higher-timeframe EMA to determine trend direction
        - Waits for price to pull back toward slower EMA
        - Enters when price bounces back (fast EMA crosses slow EMA)
        - Uses swing lows/highs for stop + trailing stop
        
        **Best for**: Trending markets, clear bull/bear runs  
        **Avoid when**: Choppy or ranging markets
        
        | Parameter | Default | Aggressive |
        |-----------|---------|------------|
        | HTF EMA | 50 | 30 |
        | LTF Slow EMA | 20 | 15 |
        | LTF Fast EMA | 10 | 5 |
        | ATR Multiplier | 2.0 | 1.5 |
        """)
    
    # Breakout
    with st.expander("🚀 Breakout", expanded=False):
        st.markdown("""
        **Concept**: Enter when price breaks out of consolidation range.
        
        **How it works**:
        - Uses Donchian Channels (highest high / lowest low)
        - LONG when price breaks above upper channel
        - SHORT when price breaks below lower channel
        - Volatility filter: Only trades when ATR is expanding
        
        **Best for**: After consolidation periods, news-driven moves  
        **Avoid when**: Market is already trending (late entries)
        
        | Parameter | Default | Conservative |
        |-----------|---------|--------------|
        | Donchian Period | 20 | 30 |
        | Volatility Threshold | 1.2 | 1.5 |
        """)
    
    # VWAP Reversion
    with st.expander("🔄 VWAP Reversion", expanded=False):
        st.markdown("""
        **Concept**: Fade extreme moves away from VWAP, bet on mean reversion.
        
        **How it works**:
        - Only activates when ADX < 25 (ranging market)
        - LONG when price significantly below VWAP + RSI oversold
        - SHORT when price significantly above VWAP + RSI overbought
        - Target is the VWAP itself
        
        ⚠️ **Warning**: Counter-trend strategy. Will fail in strong trends.
        
        **Best for**: Ranging markets, consolidation, weekends  
        **Avoid when**: Strong trend in place
        """)
    
    # MA Crossover
    with st.expander("📊 MA Crossover (Baseline)", expanded=False):
        st.markdown("""
        **Concept**: Simplest trend-following system. Enter on MA crossovers.
        
        **How it works**:
        - Golden Cross (fast > slow) → LONG
        - Death Cross (fast < slow) → SHORT
        - Exit on opposite crossover
        
        **Purpose**: Baseline for comparison, learning tool
        
        | Parameter | Default | Longer-term |
        |-----------|---------|-------------|
        | Fast EMA | 10 | 20 |
        | Slow EMA | 30 | 50 |
        """)
    
    st.divider()
    
    # ========================================
    # SECTION 2: DIVERSE STRATEGIES (BATCH 1)
    # ========================================
    st.header("⚡ Diverse Strategies")
    
    # Supertrend
    with st.expander("⚡ Supertrend", expanded=False):
        st.markdown("""
        **Concept**: Trend-following indicator that flips between bullish/bearish states.
        
        **How it works**:
        - Calculates Supertrend line based on ATR bands
        - LONG when price crosses above Supertrend (bullish flip)
        - SHORT when price crosses below Supertrend (bearish flip)
        - Stays in position until opposite flip
        
        **Best for**: Trending markets, momentum plays  
        **Trades**: Moderate frequency
        
        | Parameter | Default | Tighter |
        |-----------|---------|---------|
        | ATR Period | 10 | 7 |
        | Multiplier | 3.0 | 2.0 |
        """)
    
    # Donchian Turtle
    with st.expander("🐢 Donchian Turtle", expanded=False):
        st.markdown("""
        **Concept**: Classic Turtle Trading system - buy 55-day highs, sell 20-day lows.
        
        **How it works**:
        - Entry: Break of N-period high/low (typically 20-55 bars)
        - Exit: Break of shorter M-period level (typically 10-20 bars)
        - Trailing stop with ATR
        
        **Best for**: Strong trends, breakout environments  
        **History**: Made famous by the Turtle Traders in the 1980s
        
        | Parameter | Default |
        |-----------|---------|
        | Entry N | 20 |
        | Exit N | 10 |
        | ATR Stop Mult | 2.0 |
        """)
    
    # RSI-2 Dip
    with st.expander("📉 RSI-2 Dip", expanded=False):
        st.markdown("""
        **Concept**: Buy short-term oversold conditions in an uptrend.
        
        **How it works**:
        - Trend filter: Price > EMA(200) for longs
        - Entry: RSI(2) < oversold threshold (very short-term extreme)
        - Exit: RSI crosses back above exit threshold
        
        **Best for**: Quick mean reversion within trend  
        **Note**: Uses very short RSI period (2) for high sensitivity
        
        | Parameter | Default |
        |-----------|---------|
        | EMA Period | 200 |
        | RSI Period | 2 |
        | RSI Oversold | 10 |
        | RSI Exit | 50 |
        """)
    
    # BB Squeeze
    with st.expander("🎯 BB Squeeze", expanded=False):
        st.markdown("""
        **Concept**: Trade breakouts after Bollinger Band squeeze (low volatility compression).
        
        **How it works**:
        - Detect squeeze: Bandwidth in bottom percentile of recent history
        - Wait for price to break above/below bands after squeeze
        - Trades the volatility expansion that follows compression
        
        **Best for**: Pre-breakout positioning  
        **Key insight**: Low volatility → High volatility transition
        
        | Parameter | Default |
        |-----------|---------|
        | BB Period | 20 |
        | Squeeze Percentile | 20 |
        | Lookback | 50 |
        """)
    
    # Inside Bar
    with st.expander("📦 Inside Bar / NR7", expanded=False):
        st.markdown("""
        **Concept**: Trade breakouts of compression patterns (inside bars, narrow range bars).
        
        **How it works**:
        - Inside Bar: Today's range is completely inside yesterday's range
        - NR7: Narrowest range of the last 7 bars
        - Enter on break of pattern high/low
        - Stop at opposite side of pattern
        
        **Best for**: Low-risk pattern-based entries  
        **Trades**: Very selective in choppy markets
        
        | Parameter | Default |
        |-----------|---------|
        | Pattern Type | inside |
        | NR Lookback | 7 |
        | Target R | 2.0 |
        """)
    
    # Opening Range
    with st.expander("🌅 Opening Range Breakout (ORB)", expanded=False):
        st.markdown("""
        **Concept**: Trade breakouts of the first hour's range during a session.
        
        **How it works**:
        - Define opening range (first 60 mins of session)
        - LONG if price breaks above range high
        - SHORT if price breaks below range low
        - Exit at session end or R-target
        
        **Best for**: Session-based intraday trading  
        **Note**: Requires datetime index for session detection
        
        | Parameter | Default |
        |-----------|---------|
        | Session Start (UTC) | 14 |
        | Session End (UTC) | 21 |
        | Range Minutes | 60 |
        """)
    
    # Breakout Retest
    with st.expander("🔁 Breakout Retest", expanded=False):
        st.markdown("""
        **Concept**: Higher quality entries - wait for breakout THEN retest of level.
        
        **How it works**:
        - Identify breakout level (Donchian or prior day high/low)
        - Wait for price to retest the breakout level
        - Enter when price reclaims the level (confirms support/resistance flip)
        - Stop below retest low
        
        **Best for**: Avoiding false breakouts, better R/R entries  
        **Trades**: Fewer than pure breakout, but higher quality
        
        | Parameter | Default |
        |-----------|---------|
        | Level Source | donchian |
        | Retest Window | 5 bars |
        | ATR Stop Mult | 1.5 |
        """)
    
    # Regime Switcher
    with st.expander("🔀 Regime Switcher (Meta)", expanded=False):
        st.markdown("""
        **Concept**: Automatically switch between trend and range strategies based on ADX.
        
        **How it works**:
        - Measures ADX to determine market regime
        - ADX > threshold → Trending → Uses trend strategy
        - ADX < threshold → Ranging → Uses range strategy
        - Avoids choppy "no-trade" zones
        
        **Best for**: All-weather performance, reduces regime mismatch  
        **Key insight**: Adapts strategy to current conditions
        
        | Parameter | Default |
        |-----------|---------|
        | ADX Period | 14 |
        | ADX Trend Threshold | 25 |
        | Trend Strategy | supertrend |
        | Range Strategy | rsi2_dip |
        """)
    
    st.divider()
    
    # ========================================
    # SECTION 3: SELECTIVE STRATEGIES (BATCH 2)
    # ========================================
    st.header("💎 Selective Strategies (Low Frequency)")
    st.caption("These strategies trade less often but aim for higher-quality setups")
    
    # ATR Channel
    with st.expander("📡 ATR Channel Breakout", expanded=False):
        st.markdown("""
        **Concept**: EMA channel with ATR-based bands, only trade when volatility is expanding.
        
        **How it works**:
        - Channel: EMA ± ATR × multiplier
        - Entry: Close breaks beyond channel AND ATR% > threshold
        - Exit: Opposite channel or ATR trailing stop
        
        **Selectivity**: ATR% filter ensures only trading during vol expansion  
        **Trades**: Very few - highly selective
        
        | Parameter | Default |
        |-----------|---------|
        | EMA Period | 20 |
        | ATR Multiplier | 2.0 |
        | ATR% Threshold | 2.0 |
        """)
    
    # Volume Breakout
    with st.expander("📢 Volume-Confirmed Breakout", expanded=False):
        st.markdown("""
        **Concept**: Only take breakouts when volume confirms the move.
        
        **How it works**:
        - Breakout level: Donchian or prior day high/low
        - Volume filter: Volume > SMA(volume) × multiplier
        - Only enter when BOTH conditions pass
        
        **Selectivity**: Volume confirmation reduces false breakouts  
        **Key insight**: True breakouts typically have above-average volume
        
        | Parameter | Default |
        |-----------|---------|
        | Volume Lookback | 20 |
        | Volume Multiplier | 1.5 |
        | Breakout Source | donchian |
        """)
    
    # Z-Score Reversion
    with st.expander("📐 Z-Score Mean Reversion", expanded=False):
        st.markdown("""
        **Concept**: Trade extreme z-score readings, but ONLY in ranging regime.
        
        **How it works**:
        - Z-score = (close - SMA) / std dev
        - Regime gate: ADX must be below threshold (ranging market)
        - Entry: |z| > entry threshold (extreme reading)
        - Exit: |z| < exit threshold (returning to mean)
        
        **Selectivity**: Strict ADX gate prevents trading in trends  
        **Key insight**: Mean reversion only works in ranges
        
        | Parameter | Default |
        |-----------|---------|
        | Lookback | 20 |
        | Z Entry | 2.0 |
        | Z Exit | 0.5 |
        | ADX Threshold | 20 |
        """)
    
    # Chandelier Trend
    with st.expander("💎 Chandelier Trend", expanded=False):
        st.markdown("""
        **Concept**: Trend-following with Chandelier Exit trailing stop + EMA slope filter.
        
        **How it works**:
        - Trend filter: EMA(200) slope must be positive (longs) or negative (shorts)
        - Entry: Supertrend flip in direction of EMA slope
        - Stop: Chandelier Exit = Highest High(N) - ATR × mult
        
        **Selectivity**: Dual filter (EMA slope + Supertrend alignment)  
        **Best performer**: Often shows best risk-adjusted returns
        
        | Parameter | Default |
        |-----------|---------|
        | EMA Period | 200 |
        | Chandelier N | 22 |
        | Chandelier Mult | 3.0 |
        """)
    
    # AVWAP Pullback
    with st.expander("⚓ AVWAP Pullback", expanded=False):
        st.markdown("""
        **Concept**: Trade pullbacks to Anchored VWAP (NOT mean reversion).
        
        **How it works**:
        - Compute AVWAP anchored to daily or weekly open
        - Wait for uptrend + pullback to AVWAP
        - Enter when price reclaims AVWAP (bounces back above)
        - Stop below pullback swing low
        
        **Key difference**: This is trend-following, not mean reversion  
        **Selectivity**: Requires specific pullback + reclaim pattern
        
        | Parameter | Default |
        |-----------|---------|
        | Anchor | daily |
        | Reclaim Bars | 3 |
        | ATR Stop Mult | 1.5 |
        """)
    
    # Regression Slope
    with st.expander("📉 Regression Slope Trend", expanded=False):
        st.markdown("""
        **Concept**: Only trade when linear regression slope confirms strong trend.
        
        **How it works**:
        - Calculate rolling linear regression slope
        - Only trade when |slope| > threshold (confirmed trend)
        - Entry: Pullback to EMA in direction of slope
        - Exit: Slope weakens significantly
        
        **Selectivity**: Slope threshold filters out weak/choppy trends  
        **Trades**: Very few - only in strong directional moves
        
        | Parameter | Default |
        |-----------|---------|
        | Slope Lookback | 50 |
        | Slope Threshold | 0.5 |
        | Pullback EMA | 20 |
        """)
    
    st.divider()
    
    # ========================================
    # SECTION 4: ANTI-CHOP STRATEGIES (BATCH 3)
    # ========================================
    st.header("🔙 Anti-Chop Strategies")
    st.caption("Strategies designed to survive choppy markets and reduce overtrading")
    
    # BB Mean Reversion
    with st.expander("🔙 BB Mean Reversion", expanded=False):
        st.markdown("""
        **Concept**: Fade outer Bollinger Bands in choppy/ranging markets only.
        
        **How it works**:
        - Regime gate: Only trade when ADX < threshold OR Choppiness > threshold
        - Entry: Close crosses below lower band → LONG, above upper → SHORT
        - Exit: Mid-band (mean reversion target)
        - Stop: 1x ATR beyond band
        
        **Anti-chop design**: Strict regime gate ensures NO trades in trending markets
        
        | Parameter | Default |
        |-----------|---------|
        | BB Period | 20 |
        | BB Std Dev | 2.0 |
        | ADX Threshold | 25 |
        | Choppiness | 61.8 |
        """)
    
    # Prev Day Range
    with st.expander("📅 Previous Day Range Breakout", expanded=False):
        st.markdown("""
        **Concept**: Low-frequency daily breakout strategy with trade limiting.
        
        **How it works**:
        - Levels: Yesterday's high and low
        - Entry: Breakout above/below daily levels
        - Max 1 trade per day to reduce churn
        - Stop: Opposite side of range or ATR
        
        **Anti-chop design**: Daily trade limit prevents overtrading in noisy conditions
        
        | Parameter | Default |
        |-----------|---------|
        | Trades Per Day | 1 |
        | Retest Required | False |
        | Stop Mode | range |
        """)
    
    # TS Momentum
    with st.expander("📊 Time-Series Momentum", expanded=False):
        st.markdown("""
        **Concept**: Higher timeframe momentum following with SMA slope confirmation.
        
        **How it works**:
        - Long: Close > SMA(50) AND SMA slope positive
        - Short: Close < SMA(50) AND SMA slope negative  
        - Exit: Cross back or ATR trailing stop
        
        **Anti-chop design**: Uses longer periods naturally filters noise
        
        | Parameter | Default |
        |-----------|---------|
        | SMA Period | 50 |
        | Slope Lookback | 10 |
        | ATR Trail Mult | 2.0 |
        """)
    
    st.divider()
    
    # ========================================
    # GENERAL TIPS
    # ========================================
    st.header("💡 General Tips")
    
    st.markdown("""
    ### Position Sizing
    | Risk Tolerance | Risk/Trade | Notes |
    |----------------|------------|-------|
    | Conservative | 0.5% | Slower growth, smaller drawdowns |
    | **Moderate** | **1%** | **Balanced (default)** |
    | Aggressive | 2% | Faster growth, larger drawdowns |
    
    ### Key Metrics to Watch
    - **Profit Factor** > 1.2 is decent, > 1.5 is good
    - **Win Rate** matters less than avg win/loss ratio
    - **Max Drawdown** should be within your tolerance
    - **Total Trades** - more trades = more statistical significance
    
    ### Common Mistakes
    - ❌ Over-optimizing on 24h of data
    - ❌ Ignoring trading costs
    - ❌ Using trend strategies in ranging markets
    - ❌ Using mean reversion in trending markets
    - ✅ Test on 7-day window before conclusions
    - ✅ Use the Compare Strategies page to benchmark
    """)
    
    st.divider()
    
    # Back to trading button
    st.markdown("### Ready to trade?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Trading Dashboard", type="primary", use_container_width=True):
            st.switch_page("pages/1_Trading_Dashboard.py")
    with col2:
        if st.button("📊 Compare Strategies", use_container_width=True):
            st.switch_page("pages/3_Compare_Strategies.py")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Strategy Guide - BTC Trading Lab",
        page_icon="📚",
        layout="wide",
    )
    render_strategy_guide()
