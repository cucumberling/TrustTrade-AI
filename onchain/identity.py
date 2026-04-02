from __future__ import annotations
"""
ERC-8004 Identity Registry: Agent identity registration on-chain.
Registers an Agent NFT with a URI pointing to the agent's capabilities and endpoints.
"""

import json
from typing import Optional

from config.settings import settings
from utils.logger import logger

# ERC-8004 Identity Registry ABI (minimal)
IDENTITY_REGISTRY_ABI = [
    {
        "name": "register",
        "type": "function",
        "inputs": [{"name": "agentURI", "type": "string"}],
        "outputs": [{"name": "agentId", "type": "uint256"}],
    },
    {
        "name": "setAgentURI",
        "type": "function",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "newURI", "type": "string"},
        ],
        "outputs": [],
    },
    {
        "name": "getMetadata",
        "type": "function",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "metadataKey", "type": "string"},
        ],
        "outputs": [{"name": "", "type": "bytes"}],
    },
    {
        "name": "ownerOf",
        "type": "function",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
    },
]


def create_agent_registration_json(
    agent_name: str = "AI Trading Agent",
    description: str = "Multi-agent autonomous trading system with trend and momentum strategies",
    mcp_endpoint: str = "",
    wallet_address: str = "",
) -> dict:
    """Create the Agent Registration JSON document (ERC-8004 spec)."""
    return {
        "name": agent_name,
        "description": description,
        "version": "1.0.0",
        "capabilities": [
            "autonomous_trading",
            "risk_management",
            "portfolio_optimization",
            "market_analysis",
        ],
        "strategies": ["trend_following", "momentum", "mean_reversion"],
        "supported_pairs": ["BTC/USD", "ETH/USD"],
        "endpoints": {
            "mcp": mcp_endpoint,
        },
        "wallets": [
            {
                "chain_id": settings.onchain.chain_id,
                "address": wallet_address or settings.onchain.private_key[:10] + "...",
            }
        ],
        "trust_models": ["erc-8004-reputation", "erc-8004-validation"],
        "risk_parameters": {
            "max_drawdown": settings.risk.max_drawdown_pct,
            "max_position_pct": settings.risk.max_position_pct,
            "daily_loss_limit": settings.risk.daily_loss_limit_pct,
        },
    }


class IdentityManager:
    def __init__(self):
        self.agent_id: Optional[int] = None
        self.agent_uri: str = settings.onchain.agent_uri
        self._web3 = None
        self._contract = None

    def _init_web3(self):
        """Lazy init web3 connection."""
        if self._web3 is not None:
            return True
        try:
            from web3 import Web3
            self._web3 = Web3(Web3.HTTPProvider(settings.onchain.rpc_url))
            if not self._web3.is_connected():
                logger.error("Failed to connect to Base Sepolia RPC")
                return False

            if settings.onchain.identity_registry_address:
                self._contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(settings.onchain.identity_registry_address),
                    abi=IDENTITY_REGISTRY_ABI,
                )
            return True
        except ImportError:
            logger.warning("web3 not installed. Run: pip install web3")
            return False
        except Exception as e:
            logger.error(f"Web3 init failed: {e}")
            return False

    def register_agent(self, agent_uri: str) -> Optional[int]:
        """Register agent on the ERC-8004 Identity Registry."""
        if not self._init_web3():
            logger.warning("Web3 unavailable — simulating agent registration")
            self.agent_id = 1  # Mock ID
            self.agent_uri = agent_uri
            logger.info(f"[SIMULATED] Agent registered with ID: {self.agent_id}")
            return self.agent_id

        if not self._contract:
            logger.error("Identity Registry contract address not configured")
            return None

        try:
            from web3 import Web3
            account = self._web3.eth.account.from_key(settings.onchain.private_key)
            tx = self._contract.functions.register(agent_uri).build_transaction({
                "from": account.address,
                "nonce": self._web3.eth.get_transaction_count(account.address),
                "gas": settings.onchain.gas_limit,
                "chainId": settings.onchain.chain_id,
            })
            signed = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)

            # Parse agent ID from logs
            self.agent_id = receipt.get("logs", [{}])[0].get("topics", [None, None])[1]
            self.agent_uri = agent_uri

            logger.info(f"Agent registered on-chain",
                        tx_hash=tx_hash.hex(), agent_id=self.agent_id)
            return self.agent_id

        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return None

    def get_agent_id(self) -> Optional[int]:
        return self.agent_id
