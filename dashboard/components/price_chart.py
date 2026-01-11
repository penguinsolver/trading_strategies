"""Price chart component with candlesticks, indicators, and trade markers."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from backtest import BacktestResult


def render_price_chart(result: BacktestResult) -> go.Figure:
    """
    Render an interactive price chart with:
    - Candlesticks
    - Strategy indicators
    - Entry/exit markers
    - Volume subplot
    """
    data = result.data
    trades = result.trades
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
        subplot_titles=("", "Volume"),
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="BTC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )
    
    # Add indicator lines based on strategy
    indicator_colors = {
        "htf_ema": "#FFA726",
        "slow_ema": "#42A5F5",
        "fast_ema": "#66BB6A",
        "ema_fast": "#66BB6A",
        "ema_slow": "#EF5350",
        "vwap": "#AB47BC",
        "dc_upper": "#26a69a",
        "dc_lower": "#ef5350",
        "dc_mid": "#9E9E9E",
    }
    
    for col in indicator_colors:
        if col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    name=col.replace("_", " ").title(),
                    line=dict(color=indicator_colors[col], width=1),
                    opacity=0.8,
                ),
                row=1, col=1,
            )
    
    # Add trade markers
    if trades:
        # Long entries
        long_entries = [t for t in trades if t.side == "long"]
        if long_entries:
            fig.add_trace(
                go.Scatter(
                    x=[t.entry_time for t in long_entries],
                    y=[t.entry_price for t in long_entries],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#26a69a",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="Long Entry<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
                ),
                row=1, col=1,
            )
        
        # Short entries
        short_entries = [t for t in trades if t.side == "short"]
        if short_entries:
            fig.add_trace(
                go.Scatter(
                    x=[t.entry_time for t in short_entries],
                    y=[t.entry_price for t in short_entries],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#ef5350",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="Short Entry<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
                ),
                row=1, col=1,
            )
        
        # All exits
        fig.add_trace(
            go.Scatter(
                x=[t.exit_time for t in trades],
                y=[t.exit_price for t in trades],
                mode="markers",
                name="Exit",
                marker=dict(
                    symbol="x",
                    size=10,
                    color=["#26a69a" if t.pnl_net > 0 else "#ef5350" for t in trades],
                    line=dict(width=2),
                ),
                hovertemplate="Exit<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
            ),
            row=1, col=1,
        )
        
        # Draw lines connecting entries to exits
        for trade in trades:
            color = "#26a69a" if trade.pnl_net > 0 else "#ef5350"
            opacity = 0.3
            
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time, trade.exit_time],
                    y=[trade.entry_price, trade.exit_price],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )
    
    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350" 
              for c, o in zip(data["close"], data["open"])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.5,
        ),
        row=2, col=1,
    )
    
    # Layout
    fig.update_layout(
        title=f"BTC Price Chart - {result.strategy_name}",
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
    )
    
    return fig
