from __future__ import annotations
"""
Structured logging for the AI Trading Agent system.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class TradingLogger:
    def __init__(self, name: str = "trading_agent"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.trade_log: list[dict[str, Any]] = []

    def info(self, msg: str, **kwargs: Any) -> None:
        if kwargs:
            self.logger.info(f"{msg} | {json.dumps(kwargs, default=str)}")
        else:
            self.logger.info(msg)

    def warning(self, msg: str, **kwargs: Any) -> None:
        if kwargs:
            self.logger.warning(f"{msg} | {json.dumps(kwargs, default=str)}")
        else:
            self.logger.warning(msg)

    def error(self, msg: str, **kwargs: Any) -> None:
        if kwargs:
            self.logger.error(f"{msg} | {json.dumps(kwargs, default=str)}")
        else:
            self.logger.error(msg)

    def log_decision(
        self,
        signals: dict,
        risk_decision: str,
        final_action: str,
        portfolio_state: dict,
        explanation: str = "",
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "risk_decision": risk_decision,
            "final_action": final_action,
            "portfolio": portfolio_state,
            "explanation": explanation,
        }
        self.trade_log.append(entry)
        self.info(
            "DECISION",
            action=final_action,
            risk=risk_decision,
            signals=signals,
        )

    def log_trade(
        self,
        action: str,
        pair: str,
        quantity: float,
        price: float,
        source: str = "paper",
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "pair": pair,
            "quantity": quantity,
            "price": price,
            "source": source,
        }
        self.trade_log.append(entry)
        self.info("TRADE EXECUTED", **entry)

    def get_trade_history(self) -> list[dict[str, Any]]:
        return self.trade_log


logger = TradingLogger()
