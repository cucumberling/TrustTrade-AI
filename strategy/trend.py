from __future__ import annotations
"""
Trend-following strategy based on Moving Average crossover.
"""

from dataclasses import dataclass


@dataclass
class TrendSignal:
    signal: float       # -1.0 (strong sell) to +1.0 (strong buy)
    short_ma: float
    long_ma: float
    reason: str


def compute_ma(prices: list[float], period: int) -> float:
    if len(prices) < period:
        return sum(prices) / len(prices)
    return sum(prices[-period:]) / period


def analyze_trend(
    prices: list[float],
    short_period: int = 5,
    long_period: int = 20,
) -> TrendSignal:
    if len(prices) < 2:
        return TrendSignal(signal=0.0, short_ma=0.0, long_ma=0.0, reason="Insufficient data")

    short_ma = compute_ma(prices, short_period)
    long_ma = compute_ma(prices, long_period)

    if long_ma == 0:
        return TrendSignal(signal=0.0, short_ma=short_ma, long_ma=long_ma, reason="Long MA is zero")

    # Normalized distance between MAs, scaled by price volatility
    spread = (short_ma - long_ma) / long_ma

    # Compute recent price volatility (std dev of returns) for adaptive scaling
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    recent = returns[-long_period:] if len(returns) >= long_period else returns
    mean_r = sum(recent) / len(recent) if recent else 0
    stdev = (sum((r - mean_r) ** 2 for r in recent) / len(recent)) ** 0.5 if recent else 0

    # 1-sigma move → signal ~0.5, 2-sigma → 1.0
    if stdev > 0:
        signal = max(-1.0, min(1.0, spread / (2 * stdev)))
    else:
        signal = max(-1.0, min(1.0, spread * 3))  # conservative fallback

    if signal > 0.1:
        reason = f"Bullish: short MA ({short_ma:.2f}) > long MA ({long_ma:.2f}), spread={spread:.4f}"
    elif signal < -0.1:
        reason = f"Bearish: short MA ({short_ma:.2f}) < long MA ({long_ma:.2f}), spread={spread:.4f}"
    else:
        reason = f"Neutral: MAs converging, spread={spread:.4f}"

    return TrendSignal(signal=signal, short_ma=short_ma, long_ma=long_ma, reason=reason)
