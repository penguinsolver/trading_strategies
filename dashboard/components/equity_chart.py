"""Equity curve and drawdown chart component."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from backtest import BacktestResult


def render_equity_chart(result: BacktestResult) -> go.Figure:
    """
    Render equity curve with drawdown visualization.
    
    - Main: Equity line
    - Subplot: Drawdown percentage fill
    """
    equity = result.equity_curve
    drawdown = result.drawdown_curve
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Account Equity", "Drawdown"),
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Equity",
            line=dict(color="#42A5F5", width=2),
            fill="tozeroy",
            fillcolor="rgba(66, 165, 245, 0.1)",
            hovertemplate="Equity: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Add horizontal line for initial capital
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Initial",
        row=1, col=1,
    )
    
    # Mark trade points on equity curve
    if result.trades:
        trade_times = [t.exit_time for t in result.trades]
        trade_equities = []
        for t in result.trades:
            if t.exit_time in equity.index:
                trade_equities.append(equity.loc[t.exit_time])
            else:
                # Find closest timestamp
                idx = equity.index.get_indexer([t.exit_time], method="nearest")[0]
                trade_equities.append(equity.iloc[idx])
        
        colors = ["#26a69a" if t.pnl_net > 0 else "#ef5350" for t in result.trades]
        
        fig.add_trace(
            go.Scatter(
                x=trade_times,
                y=trade_equities,
                mode="markers",
                name="Trades",
                marker=dict(
                    size=8,
                    color=colors,
                    symbol="circle",
                ),
                hovertemplate="Trade Exit<br>Equity: $%{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    
    # Drawdown fill chart
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown",
            line=dict(color="#ef5350", width=1),
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.3)",
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    
    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    
    fig.add_annotation(
        x=max_dd_idx,
        y=max_dd_val,
        text=f"Max DD: {max_dd_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef5350",
        font=dict(color="#ef5350"),
        row=2, col=1,
    )
    
    # Layout
    fig.update_layout(
        height=400,
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
    
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    
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
