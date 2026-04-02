from __future__ import annotations
"""
Execution Agent: routes trade intents to Kraken CLI or on-chain Risk Router.
Supports paper trading, live trading via Kraken, and on-chain execution.
"""

from dataclasses import dataclass
from typing import Optional

from agents.portfolio_agent import TradeIntent
from config.settings import settings
from portfolio.tracker import PortfolioTracker
from utils.logger import logger


@dataclass
class ExecutionResult:
    success: bool
    trade_intent: TradeIntent
    execution_price: float
    execution_source: str       # "paper", "kraken_cli", "onchain"
    tx_hash: Optional[str]      # For on-chain executions
    order_id: Optional[str]     # For Kraken executions
    message: str


class ExecutionAgent:
    def __init__(self, portfolio: PortfolioTracker):
        self.portfolio = portfolio
        self.mode = settings.mode

    def execute(self, trade: TradeIntent) -> ExecutionResult:
        """Execute a trade intent through the appropriate channel."""
        if trade.direction == "HOLD":
            return ExecutionResult(
                success=True, trade_intent=trade,
                execution_price=trade.entry_price,
                execution_source="none", tx_hash=None, order_id=None,
                message="HOLD — no action taken",
            )

        if self.mode == "paper":
            return self._execute_paper(trade)
        elif self.mode == "live":
            return self._execute_kraken(trade)
        else:
            return self._execute_paper(trade)

    def _execute_paper(self, trade: TradeIntent) -> ExecutionResult:
        """Execute trade in paper trading mode using portfolio tracker."""
        price = trade.entry_price

        if trade.direction == "CLOSE":
            pnl = self.portfolio.close_position(trade.pair, price)
            if pnl is not None:
                logger.log_trade("CLOSE", trade.pair, trade.quantity, price, "paper")
                return ExecutionResult(
                    success=True, trade_intent=trade,
                    execution_price=price, execution_source="paper",
                    tx_hash=None, order_id=None,
                    message=f"Paper CLOSE: PnL=${pnl:.2f}",
                )
            return ExecutionResult(
                success=False, trade_intent=trade,
                execution_price=price, execution_source="paper",
                tx_hash=None, order_id=None,
                message="Failed to close — no position found",
            )

        side = "long" if trade.direction == "BUY" else "short"
        success = self.portfolio.open_position(
            pair=trade.pair,
            side=side,
            quantity=trade.quantity,
            price=price,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
        )

        if success:
            logger.log_trade(trade.direction, trade.pair, trade.quantity, price, "paper")
            return ExecutionResult(
                success=True, trade_intent=trade,
                execution_price=price, execution_source="paper",
                tx_hash=None, order_id=None,
                message=f"Paper {trade.direction}: {trade.quantity} @ ${price:.2f}",
            )

        return ExecutionResult(
            success=False, trade_intent=trade,
            execution_price=price, execution_source="paper",
            tx_hash=None, order_id=None,
            message="Insufficient balance or position conflict",
        )

    def _execute_kraken(self, trade: TradeIntent) -> ExecutionResult:
        """Execute trade via Kraken CLI."""
        import json
        import subprocess

        cli = settings.kraken.cli_path
        pair = trade.pair

        if trade.direction == "CLOSE":
            # Close via opposite market order
            cmd = [cli, "order", "create", pair, "market",
                   "sell" if trade.direction == "CLOSE" else "buy",
                   str(trade.quantity)]
        elif trade.direction == "BUY":
            cmd = [cli, "order", "create", pair, "market", "buy", str(trade.quantity)]
        else:
            cmd = [cli, "order", "create", pair, "market", "sell", str(trade.quantity)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = json.loads(result.stdout) if result.stdout.strip() else {}
                order_id = output.get("order_id", output.get("txid", "unknown"))
                # Also update paper portfolio for tracking
                if trade.direction == "CLOSE":
                    self.portfolio.close_position(trade.pair, trade.entry_price)
                else:
                    side = "long" if trade.direction == "BUY" else "short"
                    self.portfolio.open_position(
                        trade.pair, side, trade.quantity, trade.entry_price,
                        trade.stop_loss, trade.take_profit,
                    )
                logger.log_trade(trade.direction, pair, trade.quantity, trade.entry_price, "kraken_cli")
                return ExecutionResult(
                    success=True, trade_intent=trade,
                    execution_price=trade.entry_price,
                    execution_source="kraken_cli",
                    tx_hash=None, order_id=str(order_id),
                    message=f"Kraken {trade.direction}: order {order_id}",
                )
            else:
                logger.error(f"Kraken CLI error: {result.stderr}")
                return ExecutionResult(
                    success=False, trade_intent=trade,
                    execution_price=trade.entry_price,
                    execution_source="kraken_cli",
                    tx_hash=None, order_id=None,
                    message=f"Kraken CLI error: {result.stderr[:200]}",
                )
        except FileNotFoundError:
            logger.error("Kraken CLI not installed. Falling back to paper trading.")
            return self._execute_paper(trade)
        except Exception as e:
            logger.error(f"Kraken execution failed: {e}")
            return self._execute_paper(trade)
