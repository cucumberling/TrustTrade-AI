from __future__ import annotations
"""
LLM-powered explanation module.
Generates natural language trading reports and decision explanations.
Falls back to template-based explanation when API is unavailable.
"""

from typing import Optional

from config.settings import settings
from utils.logger import logger


class Explainer:
    def __init__(self):
        self.api_key = settings.llm.api_key
        self.model = settings.llm.model
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return True
        if not self.api_key:
            return False
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            return True
        except ImportError:
            logger.warning("anthropic package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to init Anthropic client: {e}")
            return False

    def explain_decision(
        self,
        signal_summary: dict,
        risk_summary: dict,
        trade_direction: str,
        portfolio_state: dict,
        final_reasoning: str,
    ) -> str:
        """Generate a natural language explanation for the trading decision."""
        # Try LLM first
        llm_explanation = self._llm_explain(
            signal_summary, risk_summary, trade_direction,
            portfolio_state, final_reasoning,
        )
        if llm_explanation:
            return llm_explanation

        # Fallback to template
        return self._template_explain(
            signal_summary, risk_summary, trade_direction,
            portfolio_state, final_reasoning,
        )

    def _llm_explain(
        self,
        signal_summary: dict,
        risk_summary: dict,
        trade_direction: str,
        portfolio_state: dict,
        final_reasoning: str,
    ) -> Optional[str]:
        if not self._init_client():
            return None

        prompt = f"""You are a trading agent explainer. Generate a concise, professional explanation for this trading decision.

Decision: {trade_direction}
Signal: {signal_summary}
Risk Assessment: {risk_summary}
Portfolio State: {portfolio_state}
Reasoning: {final_reasoning}

Provide a 3-4 sentence explanation covering:
1. Market trend assessment
2. Risk evaluation
3. Final decision rationale
Keep it factual and concise."""

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return None

    def _template_explain(
        self,
        signal_summary: dict,
        risk_summary: dict,
        trade_direction: str,
        portfolio_state: dict,
        final_reasoning: str,
    ) -> str:
        """Template-based explanation fallback."""
        lines = ["=== Trading Decision Explanation ==="]

        # Signal assessment
        direction = signal_summary.get("direction", "UNKNOWN")
        confidence = signal_summary.get("confidence", 0)
        strategies = signal_summary.get("strategies", {})
        lines.append(f"\n[Signal Analysis]")
        lines.append(f"  Direction: {direction} (confidence: {confidence:.1%})")
        for name, score in strategies.items():
            lines.append(f"  - {name}: {score:+.4f}")

        # Risk assessment
        risk_action = risk_summary.get("action", "UNKNOWN")
        risk_score = risk_summary.get("risk_score", 0)
        lines.append(f"\n[Risk Assessment]")
        lines.append(f"  Decision: {risk_action} (risk score: {risk_score:.2f})")
        for reason in risk_summary.get("reasons", []):
            lines.append(f"  - {reason}")

        # Portfolio state
        lines.append(f"\n[Portfolio]")
        lines.append(f"  Balance: ${portfolio_state.get('balance', 0):,.2f}")
        lines.append(f"  Total Value: ${portfolio_state.get('total_value', 0):,.2f}")
        lines.append(f"  Realized PnL: ${portfolio_state.get('realized_pnl', 0):,.2f}")
        lines.append(f"  Drawdown: {portfolio_state.get('drawdown', 0):.2%}")

        # Final decision
        lines.append(f"\n[Final Decision]")
        lines.append(f"  Action: {trade_direction}")
        lines.append(f"  Reasoning: {final_reasoning}")

        return "\n".join(lines)

    def generate_session_report(
        self,
        decision_history: list[dict],
        risk_report: dict,
        portfolio_state: dict,
    ) -> str:
        """Generate an end-of-session summary report."""
        lines = [
            "=" * 50,
            "  AI TRADING AGENT — SESSION REPORT",
            "=" * 50,
            "",
            f"Total Decisions: {len(decision_history)}",
            "",
        ]

        # Action breakdown
        actions = {}
        for d in decision_history:
            action = d.get("action", "UNKNOWN")
            actions[action] = actions.get(action, 0) + 1
        lines.append("[Action Breakdown]")
        for action, count in actions.items():
            lines.append(f"  {action}: {count}")

        # Risk metrics
        lines.append(f"\n[Risk Metrics]")
        for key, value in risk_report.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        # Portfolio
        lines.append(f"\n[Final Portfolio]")
        for key, value in portfolio_state.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            elif isinstance(value, dict):
                lines.append(f"  {key}: {len(value)} positions")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)


explainer = Explainer()
