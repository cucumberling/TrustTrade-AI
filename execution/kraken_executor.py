from __future__ import annotations
"""
Kraken CLI executor: wraps Kraken CLI commands for trade execution.
Supports paper trading (built-in sandbox) and live trading.
"""

import json
import subprocess
from typing import Optional

from config.settings import settings
from utils.logger import logger


class KrakenExecutor:
    def __init__(self):
        self.cli_path = settings.kraken.cli_path
        self.paper_mode = settings.kraken.paper_trading

    def _run(self, *args: str, timeout: int = 30) -> Optional[dict]:
        cmd = [self.cli_path] + list(args)
        if self.paper_mode:
            cmd.append("--paper")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                logger.error(f"Kraken CLI: {result.stderr}")
                return None
            return json.loads(result.stdout) if result.stdout.strip() else {}
        except FileNotFoundError:
            logger.warning("Kraken CLI not found")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Kraken CLI timed out")
            return None
        except json.JSONDecodeError:
            logger.error(f"Cannot parse output: {result.stdout[:100]}")
            return None

    def place_market_order(self, pair: str, side: str, volume: float) -> Optional[dict]:
        """Place a market order."""
        return self._run("order", "create", pair, "market", side, str(volume))

    def place_limit_order(
        self, pair: str, side: str, volume: float, price: float
    ) -> Optional[dict]:
        """Place a limit order."""
        return self._run("order", "create", pair, "limit", side, str(volume), str(price))

    def cancel_order(self, order_id: str) -> Optional[dict]:
        return self._run("order", "cancel", order_id)

    def get_open_orders(self) -> Optional[dict]:
        return self._run("orders", "open")

    def get_balance(self) -> Optional[dict]:
        return self._run("balance")

    def get_trade_history(self) -> Optional[dict]:
        return self._run("trades", "history")

    def is_available(self) -> bool:
        """Check if Kraken CLI is installed and accessible."""
        result = self._run("--version")
        return result is not None
