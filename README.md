# BTC Active Trading Lab

A comprehensive backtesting and strategy comparison platform for **BTC perpetual trading** on Hyperliquid. Built for learning, experimenting, and finding winning trading strategies.

## 🏆 Best Performing Strategy

**BREAKOUT** strategy achieved **+13.32% ROI** over 90 days, beating the MA Crossover baseline (+12.39%).

## Features

### 📊 90+ Trading Strategies
- **Technical Strategies**: MA Crossover, Trend Pullback, Breakout, VWAP Reversion, OBV Divergence, Chandelier Trend, and more
- **ML Models**: XGBoost, Random Forest, Neural Network, Stacking Ensemble, Voting Ensemble
- **Statistical Models**: HMM Regime Detection, Kalman Filter, GARCH Volatility Sizing
- **Advanced Strategies**: Momentum, ADX Trend, Dual Momentum, Mean Reversion

### 🤖 ML & Statistical Models
- **Feature Engineering**: 50+ technical indicators automatically calculated
- **Walk-Forward Training**: Prevents overfitting with time-series cross-validation
- **Ensemble Methods**: Stacking, Voting, and Multi-Model ensembles
- **Regime Detection**: HMM-based market state classification

### 📈 Interactive Dashboard (6 Pages)
1. **Trading Dashboard**: Main backtest interface with charts and metrics
2. **Strategy Guide**: Documentation for all strategies
3. **Technical Indicators**: Visualize indicators
4. **Technical Exit Engine**: Advanced exit rules
5. **ML Models**: 6 tabs for ML/Statistical/Ensemble strategies
6. **Strategy Comparison**: Compare ALL strategies side-by-side

### 💰 Realistic Backtesting
- Trading fees (maker/taker configurable)
- Slippage estimation
- Funding rate impact
- Position sizing based on risk percentage
- ATR-based stop losses

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
streamlit run dashboard/app.py
```

The dashboard will open at **http://localhost:8501**

## Usage

### Basic Backtesting
1. Navigate to **Trading Dashboard**
2. Select a strategy from the sidebar
3. Choose time window (24h to 180d)
4. Adjust parameters using sliders
5. Click **▶️ Run Backtest**
6. Analyze results and export trades

### ML Models
1. Navigate to **ML Models** page
2. Choose from 6 tabs:
   - 📊 Ensemble Voting - Combine top strategies
   - 📈 Regime Filter - Market state detection
   - 📉 Statistical Models - HMM, Kalman, GARCH
   - 🤖 ML Classifier - XGBoost signal filtering
   - 🚀 ML Signal Generator - Direct ML signals
   - 🏆 **Strategy Comparison** - Compare ALL strategies

### Strategy Comparison
The best way to find winning strategies:
1. Go to **ML Models** → **🏆 Strategy Comparison** tab
2. Select time window and risk level
3. Click **🚀 Run Strategy Comparison**
4. View ranked results to find top performers

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
├── strategies/               # 15+ technical strategies
│   ├── base.py               # Strategy interface
│   ├── ma_crossover.py       # Simple baseline
│   ├── trend_pullback.py     # Trend following
│   ├── breakout.py           # Donchian breakout
│   └── ...                   # More strategies
├── models/                   # ML & Statistical models
│   ├── ensemble.py           # Ensemble voting
│   ├── ml_signal_generator.py # ML signal generation
│   ├── advanced_models.py    # Stacking, NN, Voting
│   ├── additional_strategies.py # Momentum, ADX, etc.
│   ├── hmm_regime.py         # HMM regime detection
│   ├── kalman_filter.py      # Kalman trend filter
│   └── garch_sizing.py       # GARCH volatility sizing
├── dashboard/
│   ├── app.py                # Main Streamlit app
│   ├── pages/                # Dashboard pages
│   └── components/           # UI components
├── scripts/                  # Testing scripts
└── tests/                    # Unit tests
```

## Top Strategies (90-day, 1h candles)

| Strategy | ROI | Trades | Type |
|----------|-----|--------|------|
| 🚀 **Breakout** | +13.32% | 10 | Technical |
| 📈 MA Crossover | +12.39% | 6 | Technical |
| 📈 Trend System | +11.78% | 4 | Technical |
| 📊 ADX Trend | +9.28% | 5 | Technical |
| 🤝 Hybrid MA+ML | +9.55% | 4 | Hybrid |

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

## Limitations

- **Historical Data**: Limited to ~5000 most recent candles per timeframe
- **No Live Trading**: This is a backtest-only tool
- **Funding Rates**: Estimated (not from historical data)
- **Execution**: Bar-based simulation, not tick-level

## License

MIT License - See LICENSE file

## Disclaimer

This tool is for **educational purposes only**. Past performance does not guarantee future results. Always do your own research before trading.
