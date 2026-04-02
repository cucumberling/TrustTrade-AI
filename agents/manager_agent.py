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


@dataclass
class ManagerDecision:
    trade_intent: TradeIntent
    signal_summary: dict
    risk_summary: dict
    final_reasoning: str
    validation_artifact: dict    # For ERC-8004 validation


class ManagerAgent:
    def __init__(self, portfolio: PortfolioTracker):
        self.portfolio = portfolio
        self.portfolio_agent = PortfolioAgent()
        self.decision_history: list[ManagerDecision] = []

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
            # Construct trade via Portfolio Agent
            trade = self.portfolio_agent.construct_trade(
                direction=signal.direction,
                confidence=signal.confidence,
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
        )

        self.decision_history.append(decision)

        logger.info(
            f"Manager: {trade.direction}",
            reasoning=reasoning,
        )

        return decision

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
