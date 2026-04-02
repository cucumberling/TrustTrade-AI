from __future__ import annotations
"""
Kraken data feed via CLI and REST API.
Supports both Kraken CLI (MCP) and direct REST API calls.
"""

import json
import subprocess
from typing import Optional

from config.settings import settings
from utils.logger import logger


class KrakenFeed:
    def __init__(self):
        self.cli_path = settings.kraken.cli_path
        self.pair = settings.trading_pair

    def _run_cli(self, *args: str) -> Optional[dict]:
        """Execute a Kraken CLI command and return parsed JSON output."""
        cmd = [self.cli_path] + list(args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"Kraken CLI error: {result.stderr}")
                return None
            return json.loads(result.stdout) if result.stdout.strip() else None
        except FileNotFoundError:
            logger.warning("Kraken CLI not found. Install it or set KRAKEN_CLI_PATH.")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Kraken CLI command timed out")
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Kraken CLI output: {result.stdout[:200]}")
            return None

    def get_ticker(self, pair: Optional[str] = None) -> Optional[dict]:
        """Get current ticker data for a trading pair."""
        pair = pair or self.pair
        return self._run_cli("ticker", pair)

    def get_ohlc(
        self,
        pair: Optional[str] = None,
        interval: int = 60,
        count: int = 100,
    ) -> Optional[list[dict]]:
        """Get OHLC candlestick data."""
        pair = pair or self.pair
        result = self._run_cli(
            "ohlc", pair,
            "--interval", str(interval),
            "--count", str(count),
        )
        if result and isinstance(result, list):
            return result
        return None

    def get_orderbook(self, pair: Optional[str] = None, depth: int = 10) -> Optional[dict]:
        """Get order book data."""
        pair = pair or self.pair
        return self._run_cli("orderbook", pair, "--depth", str(depth))

    def get_recent_trades(self, pair: Optional[str] = None) -> Optional[list]:
        """Get recent trades."""
        pair = pair or self.pair
        return self._run_cli("trades", pair)

    def get_price_series(self, pair: Optional[str] = None, count: int = 50) -> list[float]:
        """Get a series of closing prices. Falls back to mock data if CLI unavailable."""
        ohlc = self.get_ohlc(pair=pair, count=count)
        if ohlc:
            return [candle.get("close", candle.get("c", 0)) for candle in ohlc]

        # Fallback to REST API
        prices = self._fetch_prices_rest(pair or self.pair, count)
        if prices:
            return prices

        # Final fallback to mock data
        logger.warning("Using mock data — Kraken API unavailable")
        from data.mock_data import get_sample_btc_prices
        return get_sample_btc_prices()[:count]

    def _fetch_prices_rest(self, pair: str, count: int) -> Optional[list[float]]:
        """Fetch prices via Kraken REST API as fallback."""
        try:
            import urllib.request
            # Convert pair format: "BTC/USD" -> "XBTUSD"
            api_pair = pair.replace("BTC", "XBT").replace("/", "")
            url = f"{settings.kraken.rest_url}/0/public/OHLC?pair={api_pair}&interval=60"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("error"):
                    logger.error(f"Kraken REST error: {data['error']}")
                    return None
                result = data.get("result", {})
                # Get the first (and usually only) pair data
                for key in result:
                    if key != "last":
                        candles = result[key]
                        # OHLC format: [time, open, high, low, close, vwap, volume, count]
                        prices = [float(c[4]) for c in candles[-count:]]
                        return prices
        except Exception as e:
            logger.warning(f"Kraken REST API unavailable: {e}")
        return None


kraken_feed = KrakenFeed()
