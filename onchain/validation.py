from __future__ import annotations
"""
ERC-8004 Validation Registry: produce and submit validation artifacts.
Uses EIP-712 typed data signatures for trade intents and decision records.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from utils.logger import logger

# EIP-712 domain for our trading agent
EIP712_DOMAIN = {
    "name": "AI Trading Agent",
    "version": "1",
    "chainId": settings.onchain.chain_id,
}

# EIP-712 type definitions for TradeIntent
TRADE_INTENT_TYPE = {
    "TradeIntent": [
        {"name": "pair", "type": "string"},
        {"name": "direction", "type": "string"},
        {"name": "quantity", "type": "uint256"},
        {"name": "price", "type": "uint256"},
        {"name": "timestamp", "type": "uint256"},
        {"name": "agentId", "type": "uint256"},
        {"name": "riskScore", "type": "uint256"},
    ],
}

VALIDATION_REGISTRY_ABI = [
    {
        "name": "validationRequest",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "validatorAddress", "type": "address"},
            {"name": "agentId", "type": "uint256"},
            {"name": "requestURI", "type": "string"},
            {"name": "requestHash", "type": "bytes32"},
        ],
        "outputs": [],
    },
    {
        "name": "validationResponse",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "requestHash", "type": "bytes32"},
            {"name": "response", "type": "uint8"},
            {"name": "responseURI", "type": "string"},
            {"name": "responseHash", "type": "bytes32"},
            {"name": "tag", "type": "string"},
        ],
        "outputs": [],
    },
    {
        "name": "getValidationStatus",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "requestHash", "type": "bytes32"}],
        "outputs": [
            {"name": "status", "type": "uint8"},
            {"name": "response", "type": "uint8"},
        ],
    },
]


class ValidationManager:
    def __init__(self):
        self._web3 = None
        self._contract = None
        self.artifacts: list[dict] = []

    def _init_web3(self):
        if self._web3 is not None:
            return True
        try:
            from web3 import Web3
            self._web3 = Web3(Web3.HTTPProvider(settings.onchain.rpc_url))
            if settings.onchain.validation_registry_address:
                self._contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(settings.onchain.validation_registry_address),
                    abi=VALIDATION_REGISTRY_ABI,
                )
            return self._web3.is_connected()
        except ImportError:
            return False
        except Exception:
            return False

    def create_trade_intent_artifact(
        self,
        pair: str,
        direction: str,
        quantity: float,
        price: float,
        agent_id: int,
        risk_score: float,
        decision_data: dict,
    ) -> dict:
        """Create a signed validation artifact for a trade intent."""
        timestamp = int(datetime.now(timezone.utc).timestamp())

        # Create the structured data
        intent_data = {
            "pair": pair,
            "direction": direction,
            "quantity": str(int(quantity * 1e8)),  # Convert to satoshis/wei
            "price": str(int(price * 100)),         # Convert to cents
            "timestamp": timestamp,
            "agentId": agent_id or 0,
            "riskScore": int(risk_score * 100),
        }

        # Create deterministic hash
        data_json = json.dumps(intent_data, sort_keys=True)
        intent_hash = hashlib.sha256(data_json.encode()).hexdigest()

        # Try to sign with EIP-712 if web3 is available
        signature = self._sign_eip712(intent_data)

        artifact = {
            "type": "TradeIntent",
            "data": intent_data,
            "hash": intent_hash,
            "signature": signature,
            "decision_context": {
                "signal_score": decision_data.get("signal_score", 0),
                "risk_action": decision_data.get("risk_action", ""),
                "portfolio_state": decision_data.get("portfolio_state", {}),
            },
            "eip712_domain": EIP712_DOMAIN,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.artifacts.append(artifact)
        logger.info(f"Validation artifact created", hash=intent_hash[:16])
        return artifact

    def _sign_eip712(self, data: dict) -> Optional[str]:
        """Sign data with EIP-712 typed data signature."""
        if not settings.onchain.private_key:
            return None

        try:
            from eth_account import Account
            from eth_account.messages import encode_typed_data

            # Build EIP-712 message
            full_message = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                    ],
                    **TRADE_INTENT_TYPE,
                },
                "primaryType": "TradeIntent",
                "domain": EIP712_DOMAIN,
                "message": {
                    "pair": data["pair"],
                    "direction": data["direction"],
                    "quantity": int(data["quantity"]),
                    "price": int(data["price"]),
                    "timestamp": data["timestamp"],
                    "agentId": data["agentId"],
                    "riskScore": data["riskScore"],
                },
            }

            signable = encode_typed_data(full_message=full_message)
            signed = Account.sign_message(signable, settings.onchain.private_key)
            return signed.signature.hex()

        except ImportError:
            logger.warning("eth_account not available for EIP-712 signing")
            return None
        except Exception as e:
            logger.warning(f"EIP-712 signing failed: {e}")
            return None

    def submit_validation_request(
        self,
        agent_id: int,
        artifact: dict,
        validator_address: str = "",
    ) -> bool:
        """Submit a validation request to the on-chain Validation Registry."""
        if not self._init_web3() or not self._contract:
            logger.info(
                f"[SIMULATED] Validation request submitted",
                artifact_hash=artifact["hash"][:16],
            )
            return True

        try:
            from web3 import Web3

            request_hash = Web3.to_bytes(hexstr=artifact["hash"])
            account = self._web3.eth.account.from_key(settings.onchain.private_key)

            tx = self._contract.functions.validationRequest(
                Web3.to_checksum_address(validator_address) if validator_address else account.address,
                agent_id,
                "",  # requestURI
                request_hash,
            ).build_transaction({
                "from": account.address,
                "nonce": self._web3.eth.get_transaction_count(account.address),
                "gas": settings.onchain.gas_limit,
                "chainId": settings.onchain.chain_id,
            })
            signed = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed.raw_transaction)
            self._web3.eth.wait_for_transaction_receipt(tx_hash)

            logger.info(f"Validation request on-chain", tx_hash=tx_hash.hex())
            return True

        except Exception as e:
            logger.error(f"Validation request failed: {e}")
            return False

    def get_all_artifacts(self) -> list[dict]:
        return self.artifacts
