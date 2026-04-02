from __future__ import annotations
"""
Signal Agent: generates trading signals from multiple strategies.
Each strategy outputs a signal in [-1.0, +1.0].
The agent combines them using configurable weights.
"""

from dataclasses import dataclass

from config.settings import settings
from strategy.trend import analyze_trend
from strategy.momentum import analyze_momentum
from utils.logger import logger


@dataclass
class AggregatedSignal:
    combined_signal: float      # -1.0 to +1.0
    direction: str              # "BUY", "SELL", "HOLD"
    confidence: float           # 0.0 to 1.0 (abs of combined signal)
    strategy_signals: dict      # Individual strategy outputs
    reasons: list[str]          # Explanation from each strategy


class SignalAgent:
    def __init__(self):
        self.weights = settings.strategy.signal_weights
        self.hold_threshold = 0.15  # Below this abs signal → HOLD

    def analyze(self, prices: list[float]) -> AggregatedSignal:
        """Run all strategies and combine signals."""
        strategy_signals = {}
        reasons = []

        # Trend strategy
        trend = analyze_trend(
            prices,
            short_period=settings.strategy.short_ma_period,
            long_period=settings.strategy.long_ma_period,
        )
        strategy_signals["trend"] = trend.signal
        reasons.append(f"[Trend] {trend.reason}")

        # Momentum strategy
        momentum = analyze_momentum(
            prices,
            rsi_period=settings.strategy.rsi_period,
            rsi_overbought=settings.strategy.rsi_overbought,
            rsi_oversold=settings.strategy.rsi_oversold,
            macd_fast=settings.strategy.macd_fast,
            macd_slow=settings.strategy.macd_slow,
            macd_signal=settings.strategy.macd_signal,
        )
        strategy_signals["momentum"] = momentum.signal
        reasons.append(f"[Momentum] {momentum.reason}")

        # Weighted combination
        combined = 0.0
        total_weight = 0.0
        for name, signal in strategy_signals.items():
            w = self.weights.get(name, 0.0)
            combined += signal * w
            total_weight += w

        if total_weight > 0:
            combined /= total_weight

        combined = max(-1.0, min(1.0, combined))

        # Determine direction
        if abs(combined) < self.hold_threshold:
            direction = "HOLD"
        elif combined > 0:
            direction = "BUY"
        else:
            direction = "SELL"

        confidence = abs(combined)

        result = AggregatedSignal(
            combined_signal=round(combined, 4),
            direction=direction,
            confidence=round(confidence, 4),
            strategy_signals={k: round(v, 4) for k, v in strategy_signals.items()},
            reasons=reasons,
        )

        logger.info(
            f"Signal Agent: {direction}",
            combined=result.combined_signal,
            confidence=result.confidence,
            strategies=result.strategy_signals,
        )

        return result
