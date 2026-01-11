# BTC Active Trading Lab

A local backtesting and strategy comparison dashboard for **BTC perpetual trading** on Hyperliquid. Built for learning and experimenting with active trading strategies.

![Dashboard Screenshot](docs/dashboard.png)

## Features

- **4 Trading Strategies** to compare:
  - 📈 **Trend Pullback**: Multi-timeframe trend following with pullback entries
  - 🚀 **Breakout**: Donchian channel breakout with volatility filter
  - 🔄 **VWAP Reversion**: Mean reversion to VWAP in ranging markets
  - 📊 **MA Crossover**: Simple EMA crossover baseline

- **Realistic Backtesting** with:
  - Trading fees (maker/taker configurable)
  - Slippage estimation
  - Funding rate impact
  - Position sizing based on risk percentage

- **Interactive Dashboard**:
  - Price charts with trade markers and indicators
  - Equity curve with drawdown visualization
  - Comprehensive performance metrics
  - Filterable trade ledger with export (CSV/JSON)

- **Time Windows**: 24-hour and 7-day backtests (extensible to longer periods)

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone or navigate to the project
cd project_hyperliquid_bot

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
python run.py
```

Or directly with Streamlit:

```bash
streamlit run dashboard/app.py
```

The dashboard will open at **http://localhost:8501**

## Usage

1. **Select a Strategy** from the sidebar
2. **Choose a Time Window** (24h or 7d)
3. **Adjust Parameters** using the sliders
4. Click **▶️ Run Backtest**
5. Analyze results:
   - View trades on the price chart
   - Check equity curve and drawdown
   - Review metrics and trade ledger
   - Export trades to CSV/JSON

## Project Structure

```
project_hyperliquid_bot/
├── run.py                    # Launch script
├── requirements.txt          # Dependencies
├── config/
│   └── settings.py           # Configuration
├── data/
│   ├── api_client.py         # Hyperliquid API client
│   ├── candle_fetcher.py     # Data fetching logic
│   └── cache.py              # Local Parquet cache
├── indicators/
│   ├── moving_averages.py    # SMA, EMA
│   ├── vwap.py               # VWAP calculation
│   ├── volatility.py         # ATR, Donchian, Bollinger
│   └── trend.py              # RSI, ADX, swing detection
├── backtest/
│   ├── engine.py             # Core backtest loop
│   ├── costs.py              # Fee/slippage model
│   ├── position.py           # Trade management
│   └── metrics.py            # Performance calculations
├── strategies/
│   ├── base.py               # Strategy interface
│   ├── trend_pullback.py     # Trend following
│   ├── breakout.py           # Donchian breakout
│   ├── vwap_reversion.py     # Mean reversion
│   └── ma_crossover.py       # Simple baseline
├── dashboard/
│   ├── app.py                # Main Streamlit app
│   └── components/           # UI components
└── tests/                    # Unit tests
```

## Strategies

### Trend Pullback (Recommended for Learning)

- **Concept**: Follow the trend, enter on pullbacks
- **Higher TF**: EMA determines trend direction
- **Lower TF**: Wait for price to pull back to slow EMA, enter when fast EMA crossed
- **Exit**: Trailing stop using ATR

### Breakout

- **Concept**: Enter on volatility expansion
- **Signal**: Price breaks Donchian channel high/low
- **Filter**: Only trade when ATR ratio shows volatility expansion
- **Exit**: Trail using ATR or Donchian mid

### VWAP Reversion

- **Concept**: Mean reversion in ranging markets
- **Filter**: Low ADX indicates ranging condition
- **Entry**: Price deviates from VWAP + RSI confirmation
- **Target**: Return to VWAP

### MA Crossover (Baseline)

- **Concept**: Simple EMA crossover
- **Purpose**: Validate that the backtest engine works
- **Long**: Fast EMA crosses above slow EMA
- **Short**: Fast EMA crosses below slow EMA

## Backtest Engine Details

### Execution Model

- **Entry Timing**: Signal on bar close → Execute at next bar open
- **Stop-Loss**: Checked against bar high/low, fills at stop price
- **Trailing**: Updates each bar based on price action
- **Costs**: Applied to all entries and exits

### Cost Model (Defaults)

| Cost Type | Default Value |
|-----------|---------------|
| Maker Fee | 0.01% (1 bp) |
| Taker Fee | 0.035% (3.5 bp) |
| Slippage | 1 bp |
| Funding | 0.01% per 8h |

## Data Source

Data is fetched from **Hyperliquid's public API**:
- Endpoint: `https://api.hyperliquid.xyz/info`
- Maximum 5000 candles available per timeframe
- Automatic pagination for larger requests
- Local caching in Parquet format

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `config/settings.py` to customize:
- API endpoints (mainnet/testnet)
- Default capital and risk settings
- Cache directory
- Fee structure

## Limitations

- **Historical Data**: Limited to ~5000 most recent candles per timeframe
- **No Live Trading**: This is a backtest-only tool
- **Funding Rates**: Estimated (not from historical data)
- **Execution**: Bar-based simulation, not tick-level

## License

MIT License - See LICENSE file

## Disclaimer

This tool is for **educational purposes only**. Past performance does not guarantee future results. Always do your own research before trading.
