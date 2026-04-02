from __future__ import annotations
"""
Portfolio tracker: manages positions, balance, and PnL.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from config.settings import settings
from utils.logger import logger


@dataclass
class Position:
    pair: str
    side: str           # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def notional_value(self) -> float:
        return self.quantity * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return self.quantity * (current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - current_price)


@dataclass
class TradeRecord:
    timestamp: str
    pair: str
    action: str         # "BUY", "SELL", "CLOSE"
    quantity: float
    price: float
    pnl: float = 0.0
    source: str = "paper"


class PortfolioTracker:
    def __init__(self, initial_balance: Optional[float] = None):
        self.balance = initial_balance or settings.portfolio.initial_balance
        self.initial_balance = self.balance
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.peak_balance: float = self.balance
        self.consecutive_losses: int = 0

    def open_position(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        cost = quantity * price
        if cost > self.balance:
            logger.warning(f"Insufficient balance: need {cost:.2f}, have {self.balance:.2f}")
            return False

        if pair in self.positions:
            logger.warning(f"Position already open for {pair}")
            return False

        self.balance -= cost
        self.positions[pair] = Position(
            pair=pair,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.trade_history.append(TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            action="BUY" if side == "long" else "SELL_SHORT",
            quantity=quantity,
            price=price,
        ))

        logger.info(f"Opened {side} position", pair=pair, quantity=quantity, price=price)
        return True

    def close_position(self, pair: str, price: float) -> Optional[float]:
        if pair not in self.positions:
            logger.warning(f"No position to close for {pair}")
            return None

        pos = self.positions[pair]
        pnl = pos.unrealized_pnl(price)
        proceeds = pos.quantity * price
        self.balance += proceeds
        self.daily_pnl += pnl

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update peak balance
        total = self.total_value(price)
        if total > self.peak_balance:
            self.peak_balance = total

        self.trade_history.append(TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            action="CLOSE",
            quantity=pos.quantity,
            price=price,
            pnl=pnl,
        ))

        del self.positions[pair]
        logger.info(f"Closed position", pair=pair, price=price, pnl=round(pnl, 2))
        return pnl

    def total_value(self, current_prices: Union[Dict[str, float], float] = 0) -> float:
        """Total portfolio value = balance + sum of position values."""
        total = self.balance
        for pair, pos in self.positions.items():
            if isinstance(current_prices, dict):
                price = current_prices.get(pair, pos.entry_price)
            else:
                price = current_prices if current_prices > 0 else pos.entry_price
            total += pos.quantity * price
        return total

    def current_drawdown(self, current_price: float = 0) -> float:
        """Current drawdown from peak as a fraction."""
        total = self.total_value(current_price)
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - total) / self.peak_balance

    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trade_history if t.action == "CLOSE")

    def unrealized_pnl(self, current_prices: Union[Dict[str, float], float] = 0) -> float:
        total = 0.0
        for pair, pos in self.positions.items():
            if isinstance(current_prices, dict):
                price = current_prices.get(pair, pos.entry_price)
            else:
                price = current_prices if current_prices > 0 else pos.entry_price
            total += pos.unrealized_pnl(price)
        return total

    def get_state(self, current_price: float = 0) -> dict:
        return {
            "balance": round(self.balance, 2),
            "positions": {
                pair: {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": round(pos.unrealized_pnl(current_price), 2),
                }
                for pair, pos in self.positions.items()
            },
            "total_value": round(self.total_value(current_price), 2),
            "realized_pnl": round(self.realized_pnl(), 2),
            "unrealized_pnl": round(self.unrealized_pnl(current_price), 2),
            "drawdown": round(self.current_drawdown(current_price), 4),
            "consecutive_losses": self.consecutive_losses,
            "num_trades": len([t for t in self.trade_history if t.action == "CLOSE"]),
        }

    def reset_daily_pnl(self) -> None:
        self.daily_pnl = 0.0
