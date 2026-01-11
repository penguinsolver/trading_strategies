"""
Configuration settings for the BTC Active Trading Lab.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CostSettings:
    """Trading cost configuration."""
    maker_fee: float = 0.0001      # 0.01% (1 bp)
    taker_fee: float = 0.00035     # 0.035% (3.5 bp)
    slippage_bps: float = 1.0      # 1 basis point slippage estimate
    funding_rate_8h: float = 0.0001  # 0.01% per 8h default estimate


@dataclass
class BacktestSettings:
    """Backtest configuration."""
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    

@dataclass
class Settings:
    """Main application settings."""
    # API Configuration
    api_base_url: str = "https://api.hyperliquid.xyz/info"
    api_testnet_url: str = "https://api.hyperliquid-testnet.xyz/info"
    ws_url: str = "wss://api.hyperliquid.xyz/ws"
    ws_testnet_url: str = "wss://api.hyperliquid-testnet.xyz/ws"
    use_testnet: bool = False
    
    # Trading asset
    coin: str = "BTC"
    
    # Data settings
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    
    # Rate limiting
    requests_per_second: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Backtest defaults
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    costs: CostSettings = field(default_factory=CostSettings)
    
    # Timeframe windows (in hours)
    timeframe_windows: dict = field(default_factory=lambda: {
        "24h": 24,
        "7d": 168,
    })
    
    @property
    def active_api_url(self) -> str:
        """Get the active API URL based on testnet setting."""
        return self.api_testnet_url if self.use_testnet else self.api_base_url
    
    @property
    def active_ws_url(self) -> str:
        """Get the active WebSocket URL based on testnet setting."""
        return self.ws_testnet_url if self.use_testnet else self.ws_url


# Global settings instance
SETTINGS = Settings()
