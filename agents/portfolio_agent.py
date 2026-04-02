from __future__ import annotations
"""
Portfolio Agent: determines position sizing and manages trade construction.
Uses fixed fraction or Kelly criterion for position sizing.
"""

from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from portfolio.tracker import PortfolioTracker
from utils.logger import logger


@dataclass
class TradeIntent:
    pair: str
    direction: str          # "BUY", "SELL", "CLOSE", "HOLD"
    quantity: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_pct: float     # % of portfolio allocated
    reasoning: str


class PortfolioAgent:
    def __init__(self):
        self.config = settings.portfolio
        self.risk_per_trade = self.config.fixed_fraction

    def construct_trade(
        self,
        direction: str,
        confidence: float,
        current_price: float,
        portfolio: PortfolioTracker,
        max_position_pct: float,
        prices: list[float],
    ) -> TradeIntent:
        """Construct a trade intent with proper sizing, stops, and targets."""
        pair = settings.trading_pair

        # If HOLD, return no-op
        if direction == "HOLD":
            return TradeIntent(
                pair=pair, direction="HOLD", quantity=0, entry_price=current_price,
                stop_loss=None, take_profit=None, position_pct=0,
                reasoning="No trade signal",
            )

        # Check if we should close an existing position
        if pair in portfolio.positions:
            pos = portfolio.positions[pair]
            should_close = (
                (direction == "SELL" and pos.side == "long") or
                (direction == "BUY" and pos.side == "short")
            )
            if should_close:
                return TradeIntent(
                    pair=pair, direction="CLOSE", quantity=pos.quantity,
                    entry_price=current_price, stop_loss=None, take_profit=None,
                    position_pct=0,
                    reasoning=f"Closing {pos.side} position — signal reversed to {direction}",
                )

        # Calculate position size
        total_value = portfolio.total_value(current_price)
        position_pct = min(max_position_pct, self._calculate_position_pct(confidence, prices))
        position_value = total_value * position_pct
        quantity = position_value / current_price if current_price > 0 else 0

        # Calculate stop loss and take profit
        atr = self._estimate_atr(prices)
        if direction == "BUY":
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
        else:
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr

        reasoning = (
            f"Position size: {position_pct:.1%} of portfolio "
            f"(${position_value:.2f}), confidence={confidence:.2f}, "
            f"ATR={atr:.2f}"
        )

        trade = TradeIntent(
            pair=pair,
            direction=direction,
            quantity=round(quantity, 8),
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_pct=round(position_pct, 4),
            reasoning=reasoning,
        )

        logger.info(
            f"Portfolio Agent: {direction} {pair}",
            quantity=trade.quantity,
            position_pct=f"{trade.position_pct:.1%}",
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
        )

        return trade

    def _calculate_position_pct(self, confidence: float, prices: list[float]) -> float:
        """Calculate position size as % of portfolio."""
        if self.config.position_sizing_method == "kelly":
            return self._kelly_criterion(confidence)
        else:
            # Fixed fraction scaled by confidence
            return self.risk_per_trade * confidence

    def _kelly_criterion(self, confidence: float) -> float:
        """Simplified Kelly criterion: f = (bp - q) / b
        where b = win/loss ratio, p = win probability, q = 1-p
        """
        # Estimate win probability from confidence
        win_prob = 0.5 + confidence * 0.2  # Scale confidence to 50-70% win prob
        loss_prob = 1 - win_prob
        win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk

        kelly = (win_loss_ratio * win_prob - loss_prob) / win_loss_ratio
        # Half-Kelly for safety
        kelly = max(0, kelly * 0.5)
        # Cap at max position
        return min(kelly, settings.risk.max_position_pct)

    def _estimate_atr(self, prices: list[float], period: int = 14) -> float:
        """Estimate Average True Range from price series."""
        if len(prices) < 2:
            return prices[-1] * 0.02 if prices else 0

        true_ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        recent = true_ranges[-period:]
        return sum(recent) / len(recent) if recent else 0

    def check_stop_loss(
        self,
        portfolio: PortfolioTracker,
        current_price: float,
    ) -> Optional[TradeIntent]:
        """Check if any position hit stop loss or take profit."""
        pair = settings.trading_pair
        if pair not in portfolio.positions:
            return None

        pos = portfolio.positions[pair]
        if pos.stop_loss and pos.side == "long" and current_price <= pos.stop_loss:
            logger.warning(f"Stop loss triggered for {pair} at {current_price}")
            return TradeIntent(
                pair=pair, direction="CLOSE", quantity=pos.quantity,
                entry_price=current_price, stop_loss=None, take_profit=None,
                position_pct=0, reasoning=f"Stop loss triggered at {current_price}",
            )

        if pos.stop_loss and pos.side == "short" and current_price >= pos.stop_loss:
            logger.warning(f"Stop loss triggered for {pair} at {current_price}")
            return TradeIntent(
                pair=pair, direction="CLOSE", quantity=pos.quantity,
                entry_price=current_price, stop_loss=None, take_profit=None,
                position_pct=0, reasoning=f"Stop loss triggered at {current_price}",
            )

        if pos.take_profit and pos.side == "long" and current_price >= pos.take_profit:
            logger.info(f"Take profit triggered for {pair} at {current_price}")
            return TradeIntent(
                pair=pair, direction="CLOSE", quantity=pos.quantity,
                entry_price=current_price, stop_loss=None, take_profit=None,
                position_pct=0, reasoning=f"Take profit triggered at {current_price}",
            )

        if pos.take_profit and pos.side == "short" and current_price <= pos.take_profit:
            logger.info(f"Take profit triggered for {pair} at {current_price}")
            return TradeIntent(
                pair=pair, direction="CLOSE", quantity=pos.quantity,
                entry_price=current_price, stop_loss=None, take_profit=None,
                position_pct=0, reasoning=f"Take profit triggered at {current_price}",
            )

        return None
