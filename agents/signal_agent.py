from __future__ import annotations
"""
Signal Agent: generates trading signals from multiple strategies.
Each strategy outputs a signal in [-1.0, +1.0].
The agent combines them using configurable weights.
"""

from dataclasses import dataclass, field
from typing import Optional

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
    fundamentals: dict = field(default_factory=dict)  # Snapshot of fundamental agents


class SignalAgent:
    def __init__(self):
        self.weights = settings.strategy.signal_weights
        self.hold_threshold = 0.05  # Below this abs signal → HOLD
        # Fundamental agent snapshots (lazy: fetched once per session)
        self._fund_signal: Optional[float] = None
        self._sent_signal: Optional[float] = None
        self._chain_signal: Optional[float] = None
        self._fund_snapshot: dict = {}
        self._sent_snapshot: dict = {}
        self._chain_snapshot: dict = {}

    def refresh_fundamentals(self, pair: str = "BTC/USD") -> None:
        """Fetch funding/sentiment/on-chain snapshots ONCE per run.
        These signals move on hour-to-day timescales, not per-bar."""
        try:
            from agents.funding_agent import funding_agent
            f = funding_agent.analyze(pair)
            self._fund_signal = f.signal
            self._fund_snapshot = {
                "signal": f.signal, "regime": f.regime,
                "avg_24h_pct": round(f.avg_rate_24h * 100, 4),
                "reason": f.reason,
            }
        except Exception as e:
            logger.warning(f"SignalAgent: funding fetch failed: {e}")
            self._fund_signal = 0.0
        try:
            from agents.sentiment_agent import sentiment_agent
            s = sentiment_agent.analyze()
            self._sent_signal = s.signal
            self._sent_snapshot = {
                "signal": s.signal,
                "fear_greed": f"{s.fear_greed_value} ({s.fear_greed_label})",
                "headline_signal": s.headline_signal,
                "headline_count": s.headline_count,
                "used_llm": s.used_llm,
                "reason": s.reason,
            }
        except Exception as e:
            logger.warning(f"SignalAgent: sentiment fetch failed: {e}")
            self._sent_signal = 0.0
        try:
            from agents.onchain_agent import onchain_agent
            c = onchain_agent.analyze()
            self._chain_signal = c.signal
            self._chain_snapshot = {
                "signal": c.signal,
                "mempool_tx": c.mempool_tx_count,
                "fee_sat_vb": c.fastest_fee,
                "hashrate_pct": c.hashrate_change_pct,
                "difficulty_pct": c.difficulty_change_pct,
                "reason": c.reason,
            }
        except Exception as e:
            logger.warning(f"SignalAgent: on-chain fetch failed: {e}")
            self._chain_signal = 0.0

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

        # Fundamental agents (snapshot — refreshed externally via refresh_fundamentals)
        if settings.strategy.use_fundamentals:
            if self._fund_signal is None and self._sent_signal is None and self._chain_signal is None:
                # Lazy first fetch if user forgot to call refresh
                self.refresh_fundamentals(settings.trading_pair)
            if self._fund_signal is not None:
                strategy_signals["funding"] = self._fund_signal
                reasons.append(f"[Funding] {self._fund_snapshot.get('reason','')}")
            if self._sent_signal is not None:
                strategy_signals["sentiment"] = self._sent_signal
                reasons.append(f"[Sentiment] {self._sent_snapshot.get('reason','')}")
            if self._chain_signal is not None:
                strategy_signals["onchain"] = self._chain_signal
                reasons.append(f"[OnChain] {self._chain_snapshot.get('reason','')}")

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

        # AGREEMENT FILTER: only trade when trend AND momentum agree on direction.
        # Research shows MA crossover alone has 57-76% false signal rate in ranging markets;
        # requiring confirmation from RSI/MACD dramatically improves win rate.
        # NOTE: only the fast technical signals participate in the agreement filter —
        # fundamentals are slow-moving regime context, not bar-by-bar confirmation.
        tech_signals = [
            strategy_signals.get("trend", 0.0),
            strategy_signals.get("momentum", 0.0),
        ]
        signals = list(strategy_signals.values())
        agree_long  = all(s >  0.05 for s in tech_signals)
        agree_short = all(s < -0.05 for s in tech_signals)

        if abs(combined) < self.hold_threshold:
            direction = "HOLD"
            reasons.append("[Filter] Combined signal below HOLD threshold")
        elif agree_long:
            direction = "BUY"
        elif agree_short:
            direction = "SELL"
        else:
            direction = "HOLD"
            reasons.append("[Filter] Trend & Momentum disagree → HOLD (avoid choppy market)")

        # Confidence from strategy agreement + magnitude
        same_direction = agree_long or agree_short
        avg_magnitude = sum(abs(s) for s in signals) / len(signals) if signals else 0
        confidence = min(1.0, avg_magnitude * (1.0 if same_direction else 0.3))

        result = AggregatedSignal(
            combined_signal=round(combined, 4),
            direction=direction,
            confidence=round(confidence, 4),
            strategy_signals={k: round(v, 4) for k, v in strategy_signals.items()},
            reasons=reasons,
            fundamentals={
                "funding": self._fund_snapshot,
                "sentiment": self._sent_snapshot,
                "onchain": self._chain_snapshot,
            },
        )

        logger.info(
            f"Signal Agent: {direction}",
            combined=result.combined_signal,
            confidence=result.confidence,
            strategies=result.strategy_signals,
        )

        return result
