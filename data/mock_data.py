from __future__ import annotations
"""
Mock market data for local testing and backtesting.
"""

import math
import random
from typing import List, Optional


def generate_price_series(
    base_price: float = 60000.0,
    num_points: int = 100,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: Optional[int] = None,
) -> list[float]:
    """Generate a realistic-looking price series with trend and volatility."""
    if seed is not None:
        random.seed(seed)

    prices = [base_price]
    for i in range(1, num_points):
        change = random.gauss(trend, volatility)
        # Add some mean reversion
        deviation = (prices[-1] - base_price) / base_price
        change -= deviation * 0.01
        prices.append(prices[-1] * (1 + change))

    return prices


def generate_ohlcv(
    base_price: float = 60000.0,
    num_candles: int = 100,
    volatility: float = 0.02,
    seed: Optional[int] = None,
) -> List[dict]:
    """Generate OHLCV candlestick data."""
    if seed is not None:
        random.seed(seed)

    candles = []
    price = base_price

    for i in range(num_candles):
        open_price = price
        high = open_price * (1 + abs(random.gauss(0, volatility)))
        low = open_price * (1 - abs(random.gauss(0, volatility)))
        close = open_price * (1 + random.gauss(0, volatility))
        volume = random.uniform(100, 10000)

        candles.append({
            "open": round(open_price, 2),
            "high": round(max(high, open_price, close), 2),
            "low": round(min(low, open_price, close), 2),
            "close": round(close, 2),
            "volume": round(volume, 2),
        })
        price = close

    return candles


def get_sample_btc_prices() -> list[float]:
    """Return a deterministic sample BTC price series for demo.

    Uses a composite of trend + cycle to produce clear buy/sell signals,
    making the demo visually compelling.
    """
    import math
    base = 65000.0
    n = 80
    random.seed(42)
    prices = []
    for i in range(n):
        # Composite: uptrend → pullback → rally → crash → recovery
        t = i / n
        trend = base * (1 + 0.08 * t)                        # gentle uptrend
        cycle = base * 0.03 * math.sin(2 * math.pi * t * 3)  # 3 cycles
        noise = random.gauss(0, base * 0.005)
        prices.append(trend + cycle + noise)
    return prices
