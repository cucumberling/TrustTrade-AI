from __future__ import annotations
"""
Manager Agent: final decision maker.
Combines signals from Signal Agent, Risk Agent, and Portfolio Agent.
Resolves conflicts and produces the final TradeIntent.
Generates validation artifacts for ERC-8004 compliance.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import hashlib
import json

from agents.signal_agent import AggregatedSignal
from agents.risk_agent import RiskDecision
from agents.portfolio_agent import PortfolioAgent, TradeIntent
from portfolio.tracker import PortfolioTracker
from config.settings import settings
from utils.logger import logger
import re


@dataclass
class ManagerDecision:
    trade_intent: TradeIntent
    signal_summary: dict
    risk_summary: dict
    final_reasoning: str
    validation_artifact: dict    # For ERC-8004 validation
    llm_used: bool = False
    llm_verdict: Optional[str] = None  # raw LLM output (for transparency)


class ManagerAgent:
    def __init__(self, portfolio: PortfolioTracker):
        self.portfolio = portfolio
        self.portfolio_agent = PortfolioAgent()
        self.decision_history: list[ManagerDecision] = []
        self._llm_client = None  # Lazy
        self.llm_call_count = 0

    def decide(
        self,
        signal: AggregatedSignal,
        risk: RiskDecision,
        prices: list[float],
        current_price: float,
    ) -> ManagerDecision:
        """Make the final trading decision based on all agent inputs."""

        # First check stop loss / take profit
        stop_trade = self.portfolio_agent.check_stop_loss(self.portfolio, current_price)
        if stop_trade:
            decision = ManagerDecision(
                trade_intent=stop_trade,
                signal_summary={"direction": signal.direction, "confidence": signal.confidence},
                risk_summary={"action": risk.action, "risk_score": risk.risk_score},
                final_reasoning=stop_trade.reasoning,
                validation_artifact=self._create_validation_artifact(
                    stop_trade, signal, risk, current_price,
                ),
            )
            self.decision_history.append(decision)
            logger.info("Manager: Stop/TP triggered", action=stop_trade.direction)
            return decision

        # Apply risk decision
        if risk.action == "REJECT":
            trade = TradeIntent(
                pair=settings.trading_pair,
                direction="HOLD",
                quantity=0,
                entry_price=current_price,
                stop_loss=None,
                take_profit=None,
                position_pct=0,
                reasoning=f"Risk Agent rejected: {'; '.join(risk.reasons)}",
            )
            reasoning = f"HOLD — Risk rejected. Reasons: {'; '.join(risk.reasons)}"

        elif signal.direction == "HOLD":
            trade = TradeIntent(
                pair=settings.trading_pair,
                direction="HOLD",
                quantity=0,
                entry_price=current_price,
                stop_loss=None,
                take_profit=None,
                position_pct=0,
                reasoning="Signal is HOLD — no strong conviction",
            )
            reasoning = "HOLD — Signal below threshold"

        else:
            # Optional LLM gate: ask Claude to confirm/veto/resize the trade
            llm_verdict_text: Optional[str] = None
            llm_used = False
            chosen_direction = signal.direction
            confidence_mult = 1.0

            if settings.strategy.use_llm_manager and settings.llm.api_key:
                verdict = self._llm_review(signal, risk, current_price)
                if verdict is not None:
                    llm_used = True
                    llm_verdict_text = verdict.get("raw", "")
                    decision_word = verdict.get("decision", "").upper()
                    if decision_word == "VETO":
                        chosen_direction = "HOLD"
                    elif decision_word == "REDUCE":
                        confidence_mult = 0.5
                    # CONFIRM = leave alone

            if chosen_direction == "HOLD":
                trade = TradeIntent(
                    pair=settings.trading_pair,
                    direction="HOLD",
                    quantity=0,
                    entry_price=current_price,
                    stop_loss=None,
                    take_profit=None,
                    position_pct=0,
                    reasoning="LLM Manager VETO" if llm_used else "Signal HOLD",
                )
                reasoning = "HOLD — LLM Manager vetoed trade" if llm_used else "HOLD — Signal below threshold"
            else:
                # Construct trade via Portfolio Agent
                trade = self.portfolio_agent.construct_trade(
                    direction=chosen_direction,
                    confidence=signal.confidence * confidence_mult,
                    current_price=current_price,
                    portfolio=self.portfolio,
                    max_position_pct=risk.max_position_pct,
                    prices=prices,
                )
                reasoning = (
                    f"{trade.direction} — Signal: {signal.combined_signal:.3f} "
                    f"({signal.confidence:.1%} confidence), "
                    f"Risk: {risk.action} (score={risk.risk_score:.2f}), "
                    f"Size: {trade.position_pct:.1%}"
                )
                if llm_used:
                    reasoning += f" | LLM: {llm_verdict_text[:80] if llm_verdict_text else 'OK'}"

        decision = ManagerDecision(
            trade_intent=trade,
            signal_summary={
                "direction": signal.direction,
                "combined_signal": signal.combined_signal,
                "confidence": signal.confidence,
                "strategies": signal.strategy_signals,
            },
            risk_summary={
                "action": risk.action,
                "risk_score": risk.risk_score,
                "max_position_pct": risk.max_position_pct,
                "reasons": risk.reasons,
            },
            final_reasoning=reasoning,
            validation_artifact=self._create_validation_artifact(
                trade, signal, risk, current_price,
            ),
            llm_used=locals().get("llm_used", False),
            llm_verdict=locals().get("llm_verdict_text"),
        )

        self.decision_history.append(decision)

        logger.info(
            f"Manager: {trade.direction}",
            reasoning=reasoning,
        )

        return decision

    # ──────────────────────────────────────────────────────────────────
    # LLM Review (Claude as final reasoner)
    # ──────────────────────────────────────────────────────────────────
    def _llm_review(
        self,
        signal: AggregatedSignal,
        risk: RiskDecision,
        current_price: float,
    ) -> Optional[dict]:
        """Ask Claude to review the proposed trade. Returns dict with
        keys 'decision' (CONFIRM/REDUCE/VETO) and 'raw' (full reply),
        or None if call failed."""
        try:
            if self._llm_client is None:
                from anthropic import Anthropic
                self._llm_client = Anthropic(api_key=settings.llm.api_key)

            fund = signal.fundamentals.get("funding", {}) or {}
            sent = signal.fundamentals.get("sentiment", {}) or {}
            chain = signal.fundamentals.get("onchain", {}) or {}

            tech_lines = "\n".join(
                f"  - {k}: {v:+.3f}" for k, v in signal.strategy_signals.items()
            )
            prompt = (
                "You are the chief risk officer of a quantitative crypto trading desk. "
                "A multi-agent system has proposed a trade. Review the evidence and decide:\n"
                "  CONFIRM  → execute as planned\n"
                "  REDUCE   → execute at half size (mixed signals)\n"
                "  VETO     → skip this trade (red flag)\n\n"
                f"PROPOSED: {signal.direction} BTC at ${current_price:,.0f}\n"
                f"Combined signal: {signal.combined_signal:+.3f} (confidence {signal.confidence:.0%})\n\n"
                "AGENT SIGNALS:\n"
                f"{tech_lines}\n\n"
                "FUNDAMENTAL CONTEXT:\n"
                f"  Funding rate: {fund.get('regime','?')} ({fund.get('avg_24h_pct','?')}%/8h) — {fund.get('reason','')}\n"
                f"  Sentiment:    {sent.get('fear_greed','?')} — {sent.get('reason','')}\n"
                f"  On-chain:     {chain.get('reason','')}\n\n"
                f"RISK: {risk.action}, score={risk.risk_score:.2f}, max_pos={risk.max_position_pct:.0%}\n\n"
                "Reply with ONE word on the first line (CONFIRM, REDUCE, or VETO), "
                "then a single-sentence rationale on the second line."
            )

            resp = self._llm_client.messages.create(
                model=settings.llm.model,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            self.llm_call_count += 1

            # Parse first word for decision
            first_token = re.split(r"[\s\n,.]+", text, maxsplit=1)[0].upper()
            if first_token not in ("CONFIRM", "REDUCE", "VETO"):
                # Loose match — find the first one that appears
                for word in ("VETO", "REDUCE", "CONFIRM"):
                    if word in text.upper():
                        first_token = word
                        break
                else:
                    first_token = "CONFIRM"
            return {"decision": first_token, "raw": text}
        except Exception as e:
            logger.warning(f"Manager: LLM review failed: {e}")
            return None

    def _create_validation_artifact(
        self,
        trade: TradeIntent,
        signal: AggregatedSignal,
        risk: RiskDecision,
        current_price: float,
    ) -> dict:
        """Create a validation artifact for ERC-8004 compliance.
        This is a structured record of the decision that can be signed with EIP-712.
        """
        artifact = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_type": "manager",
            "action": trade.direction,
            "pair": trade.pair,
            "price": current_price,
            "quantity": trade.quantity,
            "signal_score": signal.combined_signal,
            "risk_score": risk.risk_score,
            "risk_action": risk.action,
            "portfolio_state": self.portfolio.get_state(current_price),
        }

        # Create a deterministic hash for the artifact
        artifact_json = json.dumps(artifact, sort_keys=True, default=str)
        artifact["hash"] = hashlib.sha256(artifact_json.encode()).hexdigest()

        return artifact

    def get_decision_summary(self) -> list[dict]:
        """Return summary of all decisions made."""
        return [
            {
                "action": d.trade_intent.direction,
                "reasoning": d.final_reasoning,
                "signal": d.signal_summary,
                "risk": d.risk_summary,
                "artifact_hash": d.validation_artifact.get("hash", ""),
            }
            for d in self.decision_history
        ]
