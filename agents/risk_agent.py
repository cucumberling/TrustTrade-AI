from __future__ import annotations
"""
Risk Agent: evaluates whether a proposed trade should be executed.
Checks position limits, drawdown, daily loss, volatility, and consecutive losses.
"""

from dataclasses import dataclass

from config.settings import settings
from portfolio.tracker import PortfolioTracker
from utils.logger import logger


@dataclass
class RiskDecision:
    action: str         # "APPROVE", "REJECT", "REDUCE_SIZE"
    max_position_pct: float
    reasons: list[str]
    risk_score: float   # 0.0 (safe) to 1.0 (dangerous)


class RiskAgent:
    def __init__(self):
        self.config = settings.risk
        self._halt_cooldown = 0  # Rounds remaining in cooldown after consecutive loss halt

    def evaluate(
        self,
        prices: list[float],
        proposed_direction: str,
        portfolio: PortfolioTracker,
        current_price: float,
    ) -> RiskDecision:
        """Evaluate risk for a proposed trade."""
        reasons = []
        risk_score = 0.0
        max_position_pct = self.config.max_position_pct

        # 1. Check drawdown
        drawdown = portfolio.current_drawdown(current_price)
        if drawdown >= self.config.max_drawdown_pct:
            reasons.append(
                f"REJECT: Drawdown {drawdown:.1%} exceeds limit {self.config.max_drawdown_pct:.1%}"
            )
            logger.warning("Risk Agent: REJECT — max drawdown exceeded", drawdown=drawdown)
            return RiskDecision(
                action="REJECT", max_position_pct=0,
                reasons=reasons, risk_score=1.0,
            )
        risk_score += (drawdown / self.config.max_drawdown_pct) * 0.3

        # 2. Check daily loss
        total_value = portfolio.total_value(current_price)
        daily_loss_pct = abs(portfolio.daily_pnl) / total_value if total_value > 0 and portfolio.daily_pnl < 0 else 0
        if daily_loss_pct >= self.config.daily_loss_limit_pct:
            reasons.append(
                f"REJECT: Daily loss {daily_loss_pct:.1%} exceeds limit {self.config.daily_loss_limit_pct:.1%}"
            )
            logger.warning("Risk Agent: REJECT — daily loss limit", daily_loss_pct=daily_loss_pct)
            return RiskDecision(
                action="REJECT", max_position_pct=0,
                reasons=reasons, risk_score=1.0,
            )
        risk_score += (daily_loss_pct / self.config.daily_loss_limit_pct) * 0.2

        # 3. Check consecutive losses (with cooldown recovery)
        if portfolio.consecutive_losses >= self.config.consecutive_loss_halt:
            if self._halt_cooldown <= 0:
                # Start cooldown: skip 10 rounds, then reset and allow trading again
                self._halt_cooldown = 10
            self._halt_cooldown -= 1
            if self._halt_cooldown > 0:
                reasons.append(
                    f"REJECT: {portfolio.consecutive_losses} consecutive losses — cooling down ({self._halt_cooldown} rounds left)"
                )
                logger.warning("Risk Agent: REJECT — consecutive loss cooldown", remaining=self._halt_cooldown)
                return RiskDecision(
                    action="REJECT", max_position_pct=0,
                    reasons=reasons, risk_score=1.0,
                )
            else:
                # Cooldown expired: reset counter, allow trading with reduced size
                portfolio.consecutive_losses = 0
                reasons.append("Consecutive loss cooldown expired — resuming with reduced size")
                max_position_pct *= 0.5
                risk_score += 0.3
        risk_score += (portfolio.consecutive_losses / self.config.consecutive_loss_halt) * 0.2

        # 4. Check volatility
        if len(prices) >= 2:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            mean_r = sum(returns) / len(returns) if returns else 0
            volatility = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
            if volatility > self.config.volatility_threshold:
                reasons.append(
                    f"High volatility {volatility:.4f} exceeds threshold {self.config.volatility_threshold}"
                )
                # Don't reject outright, but reduce position size
                max_position_pct *= 0.5
                risk_score += 0.2
            else:
                reasons.append(f"Volatility OK: {volatility:.4f}")
            risk_score += (min(volatility, self.config.volatility_threshold) / self.config.volatility_threshold) * 0.1

        # 5. Check if we already have a position
        if proposed_direction in ("BUY", "SELL"):
            pair = settings.trading_pair
            if pair in portfolio.positions:
                existing = portfolio.positions[pair]
                if (proposed_direction == "BUY" and existing.side == "long") or \
                   (proposed_direction == "SELL" and existing.side == "short"):
                    reasons.append("Already have a position in the same direction")
                    return RiskDecision(
                        action="REJECT", max_position_pct=0,
                        reasons=reasons, risk_score=risk_score,
                    )

        risk_score = min(1.0, risk_score)

        # Determine action — smooth scaling between 0.5 and 0.9
        if risk_score > 0.9:
            action = "REDUCE_SIZE"
            max_position_pct *= 0.25
            reasons.append(f"Risk score very high ({risk_score:.2f}): 75% size reduction")
        elif risk_score > 0.5:
            action = "REDUCE_SIZE"
            scale = 1.0 - 0.75 * (risk_score - 0.5) / 0.4
            max_position_pct *= scale
            reasons.append(f"Risk score elevated ({risk_score:.2f}): {(1-scale)*100:.0f}% size reduction")
        else:
            action = "APPROVE"
            reasons.append(f"Risk score acceptable ({risk_score:.2f})")

        result = RiskDecision(
            action=action,
            max_position_pct=round(max_position_pct, 4),
            reasons=reasons,
            risk_score=round(risk_score, 4),
        )

        logger.info(
            f"Risk Agent: {action}",
            risk_score=result.risk_score,
            max_position_pct=result.max_position_pct,
        )

        return result
