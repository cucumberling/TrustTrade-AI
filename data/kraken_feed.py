from __future__ import annotations
"""
Kraken data feed via CLI and REST API.
3-tier fallback: Kraken CLI → Kraken REST API → Mock data.
"""

import json
import os
import subprocess
from typing import Optional

from config.settings import settings
from utils.logger import logger


class KrakenFeed:
    def __init__(self):
        self.cli_path = settings.kraken.cli_path
        self.pair = settings.trading_pair

    def _run_cli(self, *args: str) -> Optional[dict | list]:
        """Execute a Kraken CLI command and return parsed JSON output."""
        cmd = [self.cli_path] + list(args) + ["-o", "json"]
        env = os.environ.copy()
        env["PATH"] = os.path.expanduser("~/bin") + ":" + env.get("PATH", "")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, env=env,
            )
            if result.returncode != 0:
                logger.error(f"Kraken CLI error: {result.stderr.strip()}")
                return None
            if not result.stdout.strip():
                return None
            return json.loads(result.stdout)
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            logger.error("Kraken CLI command timed out")
            return None
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON CLI output: {result.stdout[:100]}")
            return None

    def get_ticker(self, pair: Optional[str] = None) -> Optional[dict | list]:
        """Get current ticker data for a trading pair."""
        pair = pair or self.pair
        return self._run_cli("ticker", pair)

    def get_ohlc(self, pair: Optional[str] = None, interval: int = 60) -> Optional[list]:
        """Get OHLC candlestick data from Kraken CLI."""
        pair = pair or self.pair
        result = self._run_cli("ohlc", pair, "--interval", str(interval))
        if result and isinstance(result, list):
            return result
        return None

    def get_current_price(self, pair: Optional[str] = None) -> Optional[float]:
        """Get the current price for a pair."""
        ticker = self.get_ticker(pair)
        if ticker and isinstance(ticker, list) and len(ticker) > 0:
            return float(ticker[0].get("last", 0))
        if ticker and isinstance(ticker, dict):
            return float(ticker.get("last", ticker.get("c", [0])[0]))
        return None

    def get_price_series(self, pair: Optional[str] = None, count: int = 50) -> list[float]:
        """Get closing prices. 3-tier fallback: CLI → REST → Mock."""
        # Tier 1: Kraken CLI
        ohlc = self.get_ohlc(pair=pair)
        if ohlc:
            prices = []
            for candle in ohlc:
                close = candle.get("close", candle.get("c", 0))
                if isinstance(close, str):
                    close = float(close)
                prices.append(close)
            if prices:
                logger.info(f"Got {len(prices)} prices from Kraken CLI")
                return prices[-count:]

        # Tier 2: Kraken REST API
        prices = self._fetch_prices_rest(pair or self.pair, count)
        if prices:
            logger.info(f"Got {len(prices)} prices from Kraken REST API")
            return prices

        # Tier 3: Mock data
        logger.warning("Using mock data — Kraken unavailable")
        from data.mock_data import get_sample_btc_prices
        return get_sample_btc_prices()[:count]

    def _fetch_prices_rest(self, pair: str, count: int) -> Optional[list[float]]:
        """Fetch prices via Kraken REST API as fallback."""
        try:
            import urllib.request
            api_pair = pair.replace("BTC", "XBT").replace("/", "")
            url = f"{settings.kraken.rest_url}/0/public/OHLC?pair={api_pair}&interval=60"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("error"):
                    return None
                result = data.get("result", {})
                for key in result:
                    if key != "last":
                        candles = result[key]
                        prices = [float(c[4]) for c in candles[-count:]]
                        return prices
        except Exception as e:
            logger.warning(f"Kraken REST unavailable: {e}")
        return None


kraken_feed = KrakenFeed()
