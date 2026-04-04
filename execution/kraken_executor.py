from __future__ import annotations
"""
Kraken CLI executor: wraps Kraken CLI commands for trade execution.
Supports paper trading (built-in sandbox) and live trading.

Kraken CLI paper trading commands (tested with v0.3.0):
  kraken paper init                    — Initialize $10,000 paper account
  kraken paper buy BTC/USD 0.001 -o json  — Market buy
  kraken paper sell BTC/USD 0.001 -o json — Market sell
  kraken paper status -o json          — Account summary with PnL
  kraken paper history -o json         — Trade history
  kraken paper balance -o json         — Current balances
  kraken ticker BTC/USD -o json        — Real-time ticker
  kraken ohlc BTC/USD -o json          — OHLC candle data
"""

import json
import os
import subprocess
from typing import Optional

from config.settings import settings
from utils.logger import logger


class KrakenExecutor:
    def __init__(self):
        self.cli_path = settings.kraken.cli_path
        self.paper_mode = settings.kraken.paper_trading

    def _run(self, *args: str, timeout: int = 30) -> Optional[dict | list]:
        """Execute a Kraken CLI command and return parsed JSON."""
        cmd = [self.cli_path] + list(args)
        env = os.environ.copy()
        env["PATH"] = os.path.expanduser("~/bin") + ":" + env.get("PATH", "")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env,
            )
            if result.returncode != 0:
                logger.error(f"Kraken CLI: {result.stderr.strip()}")
                return None
            if not result.stdout.strip():
                return {}
            return json.loads(result.stdout)
        except FileNotFoundError:
            logger.warning("Kraken CLI not found at: " + self.cli_path)
            return None
        except subprocess.TimeoutExpired:
            logger.error("Kraken CLI timed out")
            return None
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON output: {result.stdout[:200]}")
            return None

    # --- Paper Trading ---

    def paper_init(self) -> Optional[dict]:
        """Initialize paper trading account with $10,000."""
        return self._run("paper", "init", "-o", "json")

    def paper_buy(self, pair: str, volume: float) -> Optional[dict]:
        """Place a paper market buy order."""
        return self._run("paper", "buy", pair, str(volume), "-o", "json")

    def paper_sell(self, pair: str, volume: float) -> Optional[dict]:
        """Place a paper market sell order."""
        return self._run("paper", "sell", pair, str(volume), "-o", "json")

    def paper_status(self) -> Optional[dict]:
        """Get paper account summary with PnL."""
        return self._run("paper", "status", "-o", "json")

    def paper_history(self) -> Optional[dict | list]:
        """Get paper trade history."""
        return self._run("paper", "history", "-o", "json")

    def paper_balance(self) -> Optional[dict | list]:
        """Get paper account balances."""
        return self._run("paper", "balance", "-o", "json")

    def paper_reset(self) -> Optional[dict]:
        """Reset paper account to starting balance."""
        return self._run("paper", "reset", "--yes", "-o", "json")

    # --- Live Trading ---

    def place_market_order(self, pair: str, side: str, volume: float) -> Optional[dict]:
        """Place a live market order."""
        return self._run("order", "create", pair, "market", side, str(volume), "-o", "json")

    def cancel_order(self, order_id: str) -> Optional[dict]:
        return self._run("order", "cancel", order_id, "-o", "json")

    def get_open_orders(self) -> Optional[dict]:
        return self._run("open-orders", "-o", "json")

    # --- Market Data (works without API key) ---

    def get_ticker(self, pair: str) -> Optional[dict]:
        """Get real-time ticker data."""
        return self._run("ticker", pair, "-o", "json")

    def get_ohlc(self, pair: str, interval: int = 60) -> Optional[list]:
        """Get OHLC candle data."""
        return self._run("ohlc", pair, "--interval", str(interval), "-o", "json")

    def get_balance(self) -> Optional[dict]:
        return self._run("balance", "-o", "json")

    def get_trade_history(self) -> Optional[dict]:
        return self._run("trades", "-o", "json")

    # --- Utility ---

    def is_available(self) -> bool:
        """Check if Kraken CLI is installed and accessible."""
        try:
            env = os.environ.copy()
            env["PATH"] = os.path.expanduser("~/bin") + ":" + env.get("PATH", "")
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True, text=True, timeout=5, env=env,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
