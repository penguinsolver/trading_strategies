"""Dashboard utility functions."""
import pandas as pd


def format_currency(value: float) -> str:
    """Format as currency."""
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def format_percent(value: float) -> str:
    """Format as percentage."""
    return f"{value:.2f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format as number."""
    return f"{value:,.{decimals}f}"


def color_pnl(value: float) -> str:
    """Return color based on P&L value."""
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "gray"
