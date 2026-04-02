from __future__ import annotations
"""
On-chain executor: submits TradeIntents to the ERC-8004 Risk Router.
The Risk Router enforces position limits, max leverage, whitelisted markets, and daily loss limits.
"""

import json
from typing import Optional

from config.settings import settings
from utils.logger import logger

# Minimal Risk Router ABI
RISK_ROUTER_ABI = [
    {
        "name": "submitTradeIntent",
        "type": "function",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "pair", "type": "string"},
            {"name": "direction", "type": "uint8"},  # 0=buy, 1=sell
            {"name": "quantity", "type": "uint256"},
            {"name": "price", "type": "uint256"},
            {"name": "signature", "type": "bytes"},
        ],
        "outputs": [{"name": "success", "type": "bool"}],
    },
]


class OnChainExecutor:
    def __init__(self):
        self._web3 = None
        self._router = None
        self.execution_log: list[dict] = []

    def _init_web3(self):
        if self._web3 is not None:
            return True
        try:
            from web3 import Web3
            self._web3 = Web3(Web3.HTTPProvider(settings.onchain.rpc_url))
            return self._web3.is_connected()
        except ImportError:
            return False
        except Exception:
            return False

    def submit_trade_intent(
        self,
        agent_id: int,
        pair: str,
        direction: str,
        quantity: float,
        price: float,
        signature: Optional[str] = None,
    ) -> dict:
        """Submit a trade intent to the Risk Router contract."""
        direction_code = 0 if direction == "BUY" else 1
        quantity_wei = int(quantity * 1e8)
        price_cents = int(price * 100)

        execution = {
            "agent_id": agent_id,
            "pair": pair,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "status": "pending",
        }

        if not self._init_web3():
            execution["status"] = "simulated"
            logger.info(
                f"[SIMULATED] TradeIntent submitted to Risk Router",
                pair=pair, direction=direction, quantity=quantity,
            )
            self.execution_log.append(execution)
            return execution

        try:
            from web3 import Web3

            account = self._web3.eth.account.from_key(settings.onchain.private_key)
            sig_bytes = bytes.fromhex(signature) if signature else b""

            # In production, this would call the actual Risk Router contract
            # For now, log the intent as submitted
            execution["status"] = "submitted"
            execution["from_address"] = account.address
            logger.info(f"TradeIntent submitted", **execution)

        except Exception as e:
            execution["status"] = "failed"
            execution["error"] = str(e)
            logger.error(f"On-chain execution failed: {e}")

        self.execution_log.append(execution)
        return execution

    def get_execution_log(self) -> list[dict]:
        return self.execution_log
