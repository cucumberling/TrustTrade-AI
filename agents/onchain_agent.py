from __future__ import annotations
"""
On-Chain Agent: pulls free Bitcoin network metrics from mempool.space.

Signals derived:
1. Mempool congestion — high pending tx count vs baseline → bullish demand
2. Fee pressure — fastest-fee level (sat/vB) signals settlement urgency
3. Hashrate momentum — 3-day vs 7-day hashrate trend → miner conviction
4. Difficulty adjustment — recent positive adjustment = network growing

Output: signal in [-1.0, +1.0] where positive = bullish on-chain conditions.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time
import urllib.request
import urllib.error

from utils.logger import logger


@dataclass
class OnChainSignal:
    signal: float                       # Aggregate -1..+1
    mempool_tx_count: int               # Current pending tx count
    mempool_signal: float               # Component signal
    fastest_fee: int                    # sat/vB for next-block confirm
    fee_signal: float                   # Component signal
    hashrate_change_pct: float          # 3d vs prior period
    hashrate_signal: float              # Component signal
    difficulty_change_pct: float        # Pending difficulty adjustment %
    difficulty_signal: float            # Component signal
    reason: str
    components: dict = field(default_factory=dict)


class OnChainAgent:
    BASE = "https://mempool.space/api"

    # Empirical baselines (rough — can be tuned)
    MEMPOOL_LOW = 5_000        # Quiet
    MEMPOOL_HIGH = 80_000      # Congested
    FEE_LOW = 2                # sat/vB — quiet
    FEE_HIGH = 50              # sat/vB — congested

    def __init__(self, cache_ttl_seconds: int = 600):
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, object]] = {}

    # ──────────────────────────────────────────────────────────────────
    # HTTP helper
    # ──────────────────────────────────────────────────────────────────
    def _get_json(self, path: str) -> Optional[dict]:
        now = time.time()
        if path in self._cache:
            ts, data = self._cache[path]
            if now - ts < self.cache_ttl:
                return data  # type: ignore
        url = f"{self.BASE}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TrustTrade-AI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                self._cache[path] = (now, data)
                return data
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            logger.warning(f"OnChainAgent: fetch failed for {path}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────
    # Component scoring
    # ──────────────────────────────────────────────────────────────────
    def _score_mempool(self, tx_count: int) -> float:
        """High mempool count → high settlement demand → bullish."""
        if tx_count <= self.MEMPOOL_LOW:
            return -0.4
        if tx_count >= self.MEMPOOL_HIGH:
            return 0.8
        # Linear interpolation between low and high
        scale = (tx_count - self.MEMPOOL_LOW) / (self.MEMPOOL_HIGH - self.MEMPOOL_LOW)
        return -0.4 + scale * 1.2  # -0.4 → +0.8

    def _score_fees(self, sat_vb: int) -> float:
        """Higher fees → bullish demand."""
        if sat_vb <= self.FEE_LOW:
            return -0.3
        if sat_vb >= self.FEE_HIGH:
            return 0.7
        scale = (sat_vb - self.FEE_LOW) / (self.FEE_HIGH - self.FEE_LOW)
        return -0.3 + scale * 1.0

    def _score_hashrate(self, change_pct: float) -> float:
        """Rising hashrate = miner conviction = bullish."""
        # ±5% over the period is meaningful
        return max(-1.0, min(1.0, change_pct / 5.0))

    def _score_difficulty(self, change_pct: float) -> float:
        """Positive pending difficulty adjustment = network growing = bullish."""
        # ±5% adjustment is significant
        return max(-1.0, min(1.0, change_pct / 5.0))

    # ──────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────
    def analyze(self) -> OnChainSignal:
        # ── Mempool
        mempool = self._get_json("/mempool") or {}
        tx_count = int(mempool.get("count", 0))
        mempool_signal = self._score_mempool(tx_count)

        # ── Fees
        fees = self._get_json("/v1/fees/recommended") or {}
        fastest_fee = int(fees.get("fastestFee", 1))
        fee_signal = self._score_fees(fastest_fee)

        # ── Hashrate trend (3d window)
        hr = self._get_json("/v1/mining/hashrate/3d") or {}
        hashrates = hr.get("hashrates", [])
        hashrate_change_pct = 0.0
        if len(hashrates) >= 2:
            first = float(hashrates[0].get("avgHashrate", 0))
            last = float(hashrates[-1].get("avgHashrate", 0))
            if first > 0:
                hashrate_change_pct = (last - first) / first * 100
        hashrate_signal = self._score_hashrate(hashrate_change_pct)

        # ── Difficulty adjustment
        diff = self._get_json("/v1/difficulty-adjustment") or {}
        difficulty_change_pct = float(diff.get("difficultyChange", 0.0))
        difficulty_signal = self._score_difficulty(difficulty_change_pct)

        # ── Aggregate (weighted)
        # Mempool + fees are direct demand (60%); hashrate + difficulty are
        # miner conviction (40%, slower-moving).
        combined = (
            0.30 * mempool_signal +
            0.30 * fee_signal +
            0.20 * hashrate_signal +
            0.20 * difficulty_signal
        )
        combined = max(-1.0, min(1.0, combined))

        # Reason string
        parts = []
        parts.append(f"Mempool {tx_count:,} tx ({mempool_signal:+.2f})")
        parts.append(f"Fee {fastest_fee} sat/vB ({fee_signal:+.2f})")
        parts.append(f"Hashrate {hashrate_change_pct:+.1f}% ({hashrate_signal:+.2f})")
        parts.append(f"Difficulty {difficulty_change_pct:+.1f}% ({difficulty_signal:+.2f})")
        reason = " | ".join(parts)

        result = OnChainSignal(
            signal=round(combined, 4),
            mempool_tx_count=tx_count,
            mempool_signal=round(mempool_signal, 4),
            fastest_fee=fastest_fee,
            fee_signal=round(fee_signal, 4),
            hashrate_change_pct=round(hashrate_change_pct, 2),
            hashrate_signal=round(hashrate_signal, 4),
            difficulty_change_pct=round(difficulty_change_pct, 2),
            difficulty_signal=round(difficulty_signal, 4),
            reason=reason,
            components={
                "mempool": mempool_signal,
                "fees": fee_signal,
                "hashrate": hashrate_signal,
                "difficulty": difficulty_signal,
            },
        )

        logger.info(
            f"OnChain Agent: signal={result.signal}",
            mempool=tx_count,
            fee=fastest_fee,
            hashrate_pct=hashrate_change_pct,
        )

        return result


# Singleton
onchain_agent = OnChainAgent()
