from __future__ import annotations
"""
Momentum strategy based on RSI and MACD.
"""

from dataclasses import dataclass


@dataclass
class MomentumSignal:
    signal: float       # -1.0 to +1.0
    rsi: float
    macd_line: float
    signal_line: float
    reason: str


def compute_rsi(prices: list[float], period: int = 14) -> float:
    """Wilder's smoothed RSI — the industry standard formula."""
    if len(prices) < period + 1:
        return 50.0  # Neutral when insufficient data

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Seed avg_gain / avg_loss with SMA of first `period` changes
    first = changes[:period]
    avg_gain = sum(max(c, 0) for c in first) / period
    avg_loss = sum(max(-c, 0) for c in first) / period

    # Wilder's smoothing: EMA with alpha = 1/period
    for c in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(c, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-c, 0)) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ema(prices: list[float], period: int) -> list[float]:
    if not prices:
        return []
    multiplier = 2.0 / (period + 1)
    ema_values = [prices[0]]
    for price in prices[1:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def compute_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram)."""
    if len(prices) < slow:
        return 0.0, 0.0, 0.0

    fast_ema = compute_ema(prices, fast)
    slow_ema = compute_ema(prices, slow)

    macd_line_series = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line_series = compute_ema(macd_line_series, signal_period)

    macd_val = macd_line_series[-1]
    signal_val = signal_line_series[-1]
    histogram = macd_val - signal_val

    return macd_val, signal_val, histogram


def analyze_momentum(
    prices: list[float],
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> MomentumSignal:
    if len(prices) < 2:
        return MomentumSignal(
            signal=0.0, rsi=50.0, macd_line=0.0, signal_line=0.0,
            reason="Insufficient data",
        )

    rsi = compute_rsi(prices, rsi_period)
    macd_line, signal_line, histogram = compute_macd(prices, macd_fast, macd_slow, macd_signal)

    # RSI signal: oversold = buy, overbought = sell
    if rsi <= rsi_oversold:
        rsi_signal = 1.0
    elif rsi >= rsi_overbought:
        rsi_signal = -1.0
    else:
        # Linear interpolation between oversold and overbought
        rsi_signal = 1.0 - 2.0 * (rsi - rsi_oversold) / (rsi_overbought - rsi_oversold)

    # MACD signal: normalize histogram by price volatility
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    mean_r = sum(returns) / len(returns)
    stdev = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
    norm = stdev * prices[-1] if stdev > 0 else prices[-1] * 0.01
    macd_signal_val = max(-1.0, min(1.0, histogram / norm))

    # Combine RSI and MACD equally
    combined = 0.5 * rsi_signal + 0.5 * macd_signal_val
    combined = max(-1.0, min(1.0, combined))

    reasons = []
    if rsi <= rsi_oversold:
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi >= rsi_overbought:
        reasons.append(f"RSI overbought ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    if histogram > 0:
        reasons.append(f"MACD bullish (hist={histogram:.4f})")
    elif histogram < 0:
        reasons.append(f"MACD bearish (hist={histogram:.4f})")
    else:
        reasons.append("MACD neutral")

    return MomentumSignal(
        signal=combined,
        rsi=rsi,
        macd_line=macd_line,
        signal_line=signal_line,
        reason="; ".join(reasons),
    )
