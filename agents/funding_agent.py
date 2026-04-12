from __future__ import annotations
"""
Funding Rate Agent: reads Kraken Futures perpetual funding rate to gauge market positioning.

Theory:
- Funding rate is paid from long holders to short holders (when positive) or vice versa.
- A persistently positive funding rate means longs are crowded → contrarian bearish signal.
- A persistently negative funding rate means shorts are crowded → contrarian bullish signal.
- Empirically one of the strongest mean-reversion signals in crypto markets.

Output: signal in [-1.0, +1.0] where positive = expect rally, negative = expect drop.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import urllib.request
import urllib.error

from utils.logger import logger


@dataclass
class FundingSignal:
    signal: float                       # -1.0 (strong bearish) to +1.0 (strong bullish)
    current_rate: float                 # Most recent funding rate (relative, 8h equivalent)
    avg_rate_24h: float                 # 24h average
    avg_rate_7d: float                  # 7d average
    regime: str                         # "extreme_long", "long", "neutral", "short", "extreme_short"
    reason: str
    raw_rates: list[float] = field(default_factory=list)


class FundingAgent:
    """Pulls Kraken Futures funding rate and converts it into a trading signal."""

    # Kraken Futures perpetual symbols (not the spot pair)
    SYMBOL_MAP = {
        "BTC/USD": "PF_XBTUSD",
        "ETH/USD": "PF_ETHUSD",
    }

    BASE_URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"

    def __init__(
        self,
        extreme_long_threshold: float = 0.0001,    # 0.01% per hour ≈ 0.08% per 8h funding period (extreme)
        long_threshold: float = 0.00003,           # 0.003% per hour ≈ 0.024% per 8h (elevated)
        cache_ttl_seconds: int = 600,              # Cache for 10 minutes to avoid hammering API
    ):
        self.extreme_long_threshold = extreme_long_threshold
        self.long_threshold = long_threshold
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    def _fetch_funding_rates(self, pair: str) -> Optional[list[dict]]:
        """Fetch raw funding rates from Kraken Futures (with caching)."""
        symbol = self.SYMBOL_MAP.get(pair)
        if not symbol:
            logger.warning(f"FundingAgent: unsupported pair {pair}")
            return None

        # Cache check
        import time
        now = time.time()
        if symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < self.cache_ttl:
                return cached

        url = f"{self.BASE_URL}?symbol={symbol}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TrustTrade-AI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("result") != "success":
                    logger.warning(f"FundingAgent: API error for {symbol}")
                    return None
                rates = data.get("rates", [])
                self._cache[symbol] = (now, rates)
                logger.info(f"FundingAgent: fetched {len(rates)} funding rates for {symbol}")
                return rates
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            logger.warning(f"FundingAgent: fetch failed: {e}")
            return None

    def analyze(self, pair: str = "BTC/USD") -> FundingSignal:
        """Analyze funding rate and return a contrarian signal."""
        rates = self._fetch_funding_rates(pair)
        if not rates or len(rates) < 24:
            return FundingSignal(
                signal=0.0, current_rate=0.0, avg_rate_24h=0.0, avg_rate_7d=0.0,
                regime="unknown", reason="Insufficient funding data — neutral",
            )

        # Use relativeFundingRate (per-period rate, comparable across time)
        relative_rates = [
            float(r.get("relativeFundingRate", 0.0))
            for r in rates if r.get("relativeFundingRate") is not None
        ]

        if not relative_rates:
            return FundingSignal(
                signal=0.0, current_rate=0.0, avg_rate_24h=0.0, avg_rate_7d=0.0,
                regime="unknown", reason="No relative funding rate data",
            )

        current_rate = relative_rates[-1]
        # Kraken Futures funding is hourly. 24h = last 24 entries, 7d = last 168 entries.
        avg_24h = sum(relative_rates[-24:]) / min(24, len(relative_rates))
        avg_7d = sum(relative_rates[-168:]) / min(168, len(relative_rates))

        # Use the 24h average as the primary regime indicator (smoother than spot reading)
        primary = avg_24h

        # Map funding rate to contrarian signal:
        # High positive funding → longs over-leveraged → expect drop → SELL signal (negative)
        # High negative funding → shorts over-leveraged → expect rally → BUY signal (positive)
        if primary >= self.extreme_long_threshold:
            regime = "extreme_long"
            signal = -0.9
            reason = f"Funding extremely positive ({primary*100:.4f}%) — longs over-leveraged, contrarian SHORT"
        elif primary >= self.long_threshold:
            regime = "long"
            # Linear scale between long and extreme_long thresholds
            scale = (primary - self.long_threshold) / (self.extreme_long_threshold - self.long_threshold)
            signal = -0.3 - 0.6 * scale
            reason = f"Funding elevated ({primary*100:.4f}%) — longs crowded, mild SHORT bias"
        elif primary <= -self.extreme_long_threshold:
            regime = "extreme_short"
            signal = 0.9
            reason = f"Funding extremely negative ({primary*100:.4f}%) — shorts over-leveraged, contrarian LONG"
        elif primary <= -self.long_threshold:
            regime = "short"
            scale = (-primary - self.long_threshold) / (self.extreme_long_threshold - self.long_threshold)
            signal = 0.3 + 0.6 * scale
            reason = f"Funding negative ({primary*100:.4f}%) — shorts crowded, mild LONG bias"
        else:
            regime = "neutral"
            # Soft signal proportional to small deviations
            signal = -primary / self.long_threshold * 0.3
            reason = f"Funding near neutral ({primary*100:.4f}%) — no strong positioning"

        signal = max(-1.0, min(1.0, signal))

        result = FundingSignal(
            signal=round(signal, 4),
            current_rate=round(current_rate, 8),
            avg_rate_24h=round(avg_24h, 8),
            avg_rate_7d=round(avg_7d, 8),
            regime=regime,
            reason=reason,
            raw_rates=relative_rates[-24:],
        )

        logger.info(
            f"Funding Agent: {regime}",
            signal=result.signal,
            avg_24h=f"{avg_24h*100:.4f}%",
        )

        return result


# Singleton
funding_agent = FundingAgent()
