from __future__ import annotations
"""
Execution Agent: routes trade intents to Kraken CLI or on-chain Risk Router.
Supports paper trading (Kraken CLI sandbox), live trading, and internal simulation.
"""

from dataclasses import dataclass
from typing import Optional

from agents.portfolio_agent import TradeIntent
from config.settings import settings
from execution.kraken_executor import KrakenExecutor
from portfolio.tracker import PortfolioTracker
from utils.logger import logger


@dataclass
class ExecutionResult:
    success: bool
    trade_intent: TradeIntent
    execution_price: float
    execution_source: str       # "paper_internal", "kraken_paper", "kraken_live", "onchain"
    tx_hash: Optional[str]      # For on-chain executions
    order_id: Optional[str]     # For Kraken executions
    message: str


class ExecutionAgent:
    def __init__(self, portfolio: PortfolioTracker):
        self.portfolio = portfolio
        self.mode = settings.mode
        self.kraken = KrakenExecutor()

    def execute(self, trade: TradeIntent) -> ExecutionResult:
        """Execute a trade intent through the appropriate channel."""
        if trade.direction == "HOLD":
            return ExecutionResult(
                success=True, trade_intent=trade,
                execution_price=trade.entry_price,
                execution_source="none", tx_hash=None, order_id=None,
                message="HOLD — no action taken",
            )

        if self.mode == "paper" and self.kraken.is_available():
            return self._execute_kraken_paper(trade)
        elif self.mode == "live" and self.kraken.is_available():
            return self._execute_kraken_live(trade)
        else:
            return self._execute_internal(trade)

    def _execute_kraken_paper(self, trade: TradeIntent) -> ExecutionResult:
        """Execute via Kraken CLI paper trading sandbox."""
        pair = trade.pair
        volume = trade.quantity

        if trade.direction == "CLOSE":
            # Determine sell side based on current position
            result = self.kraken.paper_sell(pair, volume)
        elif trade.direction == "BUY":
            result = self.kraken.paper_buy(pair, volume)
        else:  # SELL
            result = self.kraken.paper_sell(pair, volume)

        if result and result.get("action") == "market_order_filled":
            fill_price = result.get("price", trade.entry_price)
            order_id = result.get("order_id", "unknown")

            # Update internal portfolio tracker
            if trade.direction == "CLOSE":
                self.portfolio.close_position(pair, fill_price)
            else:
                side = "long" if trade.direction == "BUY" else "short"
                self.portfolio.open_position(
                    pair, side, volume, fill_price,
                    trade.stop_loss, trade.take_profit,
                    leverage=settings.portfolio.leverage,
                )

            logger.log_trade(trade.direction, pair, volume, fill_price, "kraken_paper")
            return ExecutionResult(
                success=True, trade_intent=trade,
                execution_price=fill_price, execution_source="kraken_paper",
                tx_hash=None, order_id=order_id,
                message=f"Kraken Paper {trade.direction}: {volume} @ ${fill_price:.2f} (order {order_id})",
            )

        # Kraken paper failed — fall back to internal
        logger.warning("Kraken paper order failed, falling back to internal simulation")
        return self._execute_internal(trade)

    def _execute_kraken_live(self, trade: TradeIntent) -> ExecutionResult:
        """Execute via Kraken CLI live trading."""
        pair = trade.pair
        side = "buy" if trade.direction == "BUY" else "sell"
        volume = trade.quantity

        if trade.direction == "CLOSE":
            side = "sell"  # Close long by selling

        result = self.kraken.place_market_order(pair, side, volume)
        if result:
            order_id = result.get("order_id", result.get("txid", "unknown"))
            fill_price = result.get("price", trade.entry_price)

            # Update internal tracker
            if trade.direction == "CLOSE":
                self.portfolio.close_position(pair, fill_price)
            else:
                pos_side = "long" if trade.direction == "BUY" else "short"
                self.portfolio.open_position(
                    pair, pos_side, volume, fill_price,
                    trade.stop_loss, trade.take_profit,
                    leverage=settings.portfolio.leverage,
                )

            logger.log_trade(trade.direction, pair, volume, fill_price, "kraken_live")
            return ExecutionResult(
                success=True, trade_intent=trade,
                execution_price=fill_price, execution_source="kraken_live",
                tx_hash=None, order_id=str(order_id),
                message=f"Kraken Live {trade.direction}: order {order_id}",
            )

        logger.error("Kraken live order failed")
        return ExecutionResult(
            success=False, trade_intent=trade,
            execution_price=trade.entry_price, execution_source="kraken_live",
            tx_hash=None, order_id=None,
            message="Kraken live order failed",
        )

    def _apply_slippage(self, mid_price: float, direction: str, pair: str) -> float:
        """Apply adverse slippage to a market order fill price.
        BUY  → fill above mid (pay more).
        SELL → fill below mid (receive less).
        CLOSE → use opposite side of the open position (long close = sell, short close = buy).
        """
        bps = settings.portfolio.slippage_bps / 10_000.0
        if direction == "BUY":
            return mid_price * (1 + bps)
        if direction == "SELL":
            return mid_price * (1 - bps)
        if direction == "CLOSE":
            pos = self.portfolio.positions.get(pair)
            if pos is None:
                return mid_price
            # Closing a long means selling → fill lower; closing a short means buying → fill higher.
            return mid_price * (1 - bps) if pos.side == "long" else mid_price * (1 + bps)
        return mid_price

    def _execute_internal(self, trade: TradeIntent) -> ExecutionResult:
        """Execute trade using internal portfolio simulation (no external dependency)."""
        price = self._apply_slippage(trade.entry_price, trade.direction, trade.pair)

        if trade.direction == "CLOSE":
            pnl = self.portfolio.close_position(trade.pair, price)
            if pnl is not None:
                logger.log_trade("CLOSE", trade.pair, trade.quantity, price, "internal")
                return ExecutionResult(
                    success=True, trade_intent=trade,
                    execution_price=price, execution_source="paper_internal",
                    tx_hash=None, order_id=None,
                    message=f"Internal CLOSE: PnL=${pnl:.2f}",
                )
            return ExecutionResult(
                success=False, trade_intent=trade,
                execution_price=price, execution_source="paper_internal",
                tx_hash=None, order_id=None,
                message="Failed to close — no position found",
            )

        side = "long" if trade.direction == "BUY" else "short"
        success = self.portfolio.open_position(
            pair=trade.pair, side=side, quantity=trade.quantity, price=price,
            stop_loss=trade.stop_loss, take_profit=trade.take_profit,
            leverage=settings.portfolio.leverage,
        )

        if success:
            logger.log_trade(trade.direction, trade.pair, trade.quantity, price, "internal")
            return ExecutionResult(
                success=True, trade_intent=trade,
                execution_price=price, execution_source="paper_internal",
                tx_hash=None, order_id=None,
                message=f"Internal {trade.direction}: {trade.quantity} @ ${price:.2f}",
            )

        return ExecutionResult(
            success=False, trade_intent=trade,
            execution_price=price, execution_source="paper_internal",
            tx_hash=None, order_id=None,
            message="Insufficient balance or position conflict",
        )
