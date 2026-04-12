from __future__ import annotations
"""
Configuration settings for the AI Trading Agent system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Load .env file if present (before any os.getenv calls below)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class KrakenConfig:
    api_key: str = os.getenv("KRAKEN_API_KEY", "")
    api_secret: str = os.getenv("KRAKEN_API_SECRET", "")
    cli_path: str = os.getenv("KRAKEN_CLI_PATH", "kraken")
    paper_trading: bool = True
    default_pair: str = "BTC/USD"
    websocket_url: str = "wss://ws.kraken.com"
    rest_url: str = "https://api.kraken.com"


@dataclass
class OnChainConfig:
    rpc_url: str = os.getenv("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")
    private_key: str = os.getenv("AGENT_PRIVATE_KEY", "")
    chain_id: int = 84532  # Base Sepolia
    identity_registry_address: str = os.getenv(
        "IDENTITY_REGISTRY_ADDRESS", "0x8004A818BFB912233c491871b3d84c89A494BD9e"
    )
    reputation_registry_address: str = os.getenv(
        "REPUTATION_REGISTRY_ADDRESS", "0x8004B663056A597Dffe9eCcC1965A193B7388713"
    )
    validation_registry_address: str = os.getenv("VALIDATION_REGISTRY_ADDRESS", "")
    agent_uri: str = ""  # IPFS or HTTPS URI for Agent Registration JSON
    gas_limit: int = 800_000


@dataclass
class RiskConfig:
    max_position_pct: float = 0.60       # Max 60% of portfolio per position
    max_drawdown_pct: float = 0.25       # Max 25% drawdown before halt
    daily_loss_limit_pct: float = 0.10   # Max 10% daily loss
    volatility_threshold: float = 0.08   # Reject if volatility > 8%
    consecutive_loss_halt: int = 5       # Halt after 5 consecutive losses
    max_leverage: float = 1.0            # No leverage by default


@dataclass
class StrategyConfig:
    short_ma_period: int = 12
    long_ma_period: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    signal_weights: dict = field(default_factory=lambda: {
        "trend": 0.35,
        "momentum": 0.35,
        "funding": 0.10,      # Mean-reversion: crowded longs → contrarian short
        "sentiment": 0.10,    # F&G + headlines (contrarian on F&G)
        "onchain": 0.10,      # Mempool / fee / hashrate / difficulty
    })
    use_fundamentals: bool = True   # Pull funding / sentiment / on-chain agents
    use_llm_manager: bool = False   # If True, Manager calls Claude to break ties


@dataclass
class PortfolioConfig:
    initial_balance: float = 10000.0
    base_currency: str = "USD"
    position_sizing_method: str = "fixed_fraction"  # "fixed_fraction" or "kelly"
    fixed_fraction: float = 0.10  # 10% cash per trade (industry-realistic; with 5x leverage = 50% notional)
    leverage: float = 1.0          # 1x = spot. Kraken Margin supports up to 5x on BTC/USD.
    slippage_bps: float = 3.0      # 3 bps = 0.03% adverse fill (realistic for Kraken BTC/USD majors)


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-20250514"


@dataclass
class Settings:
    kraken: KrakenConfig = field(default_factory=KrakenConfig)
    onchain: OnChainConfig = field(default_factory=OnChainConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trading_pair: str = "BTC/USD"
    loop_interval_seconds: int = 60
    mode: str = "paper"  # "paper", "live", "backtest"


# Global settings instance
settings = Settings()
