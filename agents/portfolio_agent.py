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

        # Position management: HOLD existing positions until SL/TP fires.
        # Auto-flipping on every opposite signal kills R:R because winners get
        # cut short and losers exit at the worst possible point. We let
        # check_stop_loss() handle exits via the SL/TP brackets set at entry.
        # Only force-close on a STRONG opposite signal (confidence > 0.6) to
        # avoid being trapped in a losing position when the trend has clearly
        # reversed.
        STRONG_FLIP_THRESHOLD = 0.6
        if pair in portfolio.positions:
            pos = portfolio.positions[pair]
            opposite = (
                (direction == "SELL" and pos.side == "long") or
                (direction == "BUY" and pos.side == "short")
            )
            if opposite and confidence >= STRONG_FLIP_THRESHOLD:
                return TradeIntent(
                    pair=pair, direction="CLOSE", quantity=pos.quantity,
                    entry_price=current_price, stop_loss=None, take_profit=None,
                    position_pct=0,
                    reasoning=f"Closing {pos.side} — strong opposite signal (conf={confidence:.2f})",
                )
            # Otherwise hold and let SL/TP handle the exit
            return TradeIntent(
                pair=pair, direction="HOLD", quantity=0, entry_price=current_price,
                stop_loss=pos.stop_loss, take_profit=pos.take_profit, position_pct=0,
                reasoning=f"Holding {pos.side} position — waiting for SL/TP",
            )

        # Calculate position size
        # position_pct is the % of equity used as MARGIN.
        # With leverage L, notional exposure = equity * position_pct * L.
        total_value = portfolio.total_value(current_price)
        position_pct = min(max_position_pct, self._calculate_position_pct(confidence, prices))
        leverage = max(1.0, settings.portfolio.leverage)
        notional_value = total_value * position_pct * leverage
        quantity = notional_value / current_price if current_price > 0 else 0

        # Calculate stop loss and take profit
        atr = self._estimate_atr(prices)
        if direction == "BUY":
            stop_loss = current_price - 3 * atr
            take_profit = current_price + 4.5 * atr
        else:
            stop_loss = current_price + 3 * atr
            take_profit = current_price - 4.5 * atr

        reasoning = (
            f"Margin: {position_pct:.1%} of equity × {leverage:.0f}x lev = "
            f"${notional_value:.2f} notional, confidence={confidence:.2f}, "
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
        """Calculate position size as % of portfolio.
        Base = fixed_fraction, scaled up by confidence (not down).
        Minimum 1% to avoid dust trades eaten by fees.
        """
        if self.config.position_sizing_method == "kelly":
            return self._kelly_criterion(confidence)
        else:
            # Base fraction + confidence scaling (e.g., 40% base → 28%-60% range)
            pct = self.risk_per_trade * (0.7 + 0.8 * confidence)
            return max(0.05, pct)  # Floor at 5% to avoid dust trades

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
        """Estimate ATR using stdev of returns as proxy (close-only data)."""
        if len(prices) < 2:
            return prices[-1] * 0.02 if prices else 0

        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        recent = returns[-period:]
        mean_r = sum(recent) / len(recent)
        stdev = (sum((r - mean_r) ** 2 for r in recent) / len(recent)) ** 0.5
        return stdev * prices[-1]

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
