from __future__ import annotations
"""
ERC-8004 Reputation Registry: submit and query trading feedback on-chain.
"""

from typing import Optional

from config.settings import settings
from utils.logger import logger

REPUTATION_REGISTRY_ABI = [
    {
        "name": "giveFeedback",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "value", "type": "int128"},
            {"name": "valueDecimals", "type": "uint8"},
            {"name": "tag1", "type": "string"},
            {"name": "tag2", "type": "string"},
            {"name": "endpoint", "type": "string"},
            {"name": "feedbackURI", "type": "string"},
            {"name": "feedbackHash", "type": "bytes32"},
        ],
        "outputs": [],
    },
    {
        "name": "getSummary",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "clientAddresses", "type": "address[]"},
            {"name": "tag1", "type": "string"},
            {"name": "tag2", "type": "string"},
        ],
        "outputs": [
            {"name": "count", "type": "uint64"},
            {"name": "summaryValue", "type": "int128"},
            {"name": "summaryValueDecimals", "type": "uint8"},
        ],
    },
]


class ReputationManager:
    def __init__(self):
        self._web3 = None
        self._contract = None
        self.feedback_history: list[dict] = []

    def _init_web3(self):
        if self._web3 is not None:
            return True
        try:
            from web3 import Web3
            self._web3 = Web3(Web3.HTTPProvider(settings.onchain.rpc_url))
            if settings.onchain.reputation_registry_address:
                self._contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(settings.onchain.reputation_registry_address),
                    abi=REPUTATION_REGISTRY_ABI,
                )
            return self._web3.is_connected()
        except ImportError:
            logger.warning("web3 not installed for reputation")
            return False
        except Exception as e:
            logger.error(f"Reputation web3 init failed: {e}")
            return False

    def submit_trading_yield(
        self,
        agent_id: int,
        yield_pct: float,
        period: str = "trade",
        feedback_uri: str = "",
    ) -> bool:
        """Submit trading yield feedback to the Reputation Registry."""
        # Convert yield to int128 with 2 decimals (e.g., -3.2% → -320)
        value = int(yield_pct * 100)

        feedback = {
            "agent_id": agent_id,
            "tag1": "tradingYield",
            "tag2": period,
            "value": yield_pct,
            "value_encoded": value,
        }

        if not self._init_web3() or not self._contract:
            logger.info(f"[SIMULATED] Reputation feedback: tradingYield={yield_pct:.2f}%")
            self.feedback_history.append(feedback)
            return True

        try:
            import hashlib
            from web3 import Web3

            feedback_hash = Web3.to_bytes(
                hexstr=hashlib.sha256(str(feedback).encode()).hexdigest()
            )

            account = self._web3.eth.account.from_key(settings.onchain.private_key)
            tx = self._contract.functions.giveFeedback(
                agent_id,
                value,
                2,  # 2 decimals
                "tradingYield",
                period,
                "",
                feedback_uri,
                feedback_hash,
            ).build_transaction({
                "from": account.address,
                "nonce": self._web3.eth.get_transaction_count(account.address),
                "gas": settings.onchain.gas_limit,
                "chainId": settings.onchain.chain_id,
            })
            signed = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed.raw_transaction)
            self._web3.eth.wait_for_transaction_receipt(tx_hash)

            logger.info(f"Reputation feedback submitted", tx_hash=tx_hash.hex())
            self.feedback_history.append(feedback)
            return True

        except Exception as e:
            logger.error(f"Reputation submission failed: {e}")
            self.feedback_history.append(feedback)
            return False

    def submit_success_rate(
        self,
        agent_id: int,
        win_rate: float,
        period: str = "session",
    ) -> bool:
        """Submit success rate feedback."""
        value = int(win_rate * 10000)  # 4 decimals
        feedback = {
            "agent_id": agent_id,
            "tag1": "successRate",
            "tag2": period,
            "value": win_rate,
        }
        logger.info(f"[SIMULATED] Reputation feedback: successRate={win_rate:.2%}")
        self.feedback_history.append(feedback)
        return True

    def get_feedback_history(self) -> list[dict]:
        return self.feedback_history
