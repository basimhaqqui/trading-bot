from __future__ import annotations

import base64
from datetime import datetime
from typing import Mapping, Protocol

from trading_bot.core.schemas import MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import stable_event_id
from trading_bot.data.http import ReadOnlyJsonRpcTransport
from trading_bot.data.schemas import CollectionBatch


class SolanaAccountTransport(Protocol):
    def call(self, method: str, params: list[object]) -> Mapping[str, object]:
        ...


class SolanaMintAuthorityCollector:
    """Observe public SPL mint authorities without any wallet or transaction access."""

    venue = "solana"
    RPC_URL = "https://api.mainnet-beta.solana.com"
    RPC_DOCUMENTATION_URL = "https://solana.com/docs/rpc/http/getmultipleaccounts"
    TOKEN_DOCUMENTATION_URL = "https://solana.com/docs/tokens/basics"
    SOLANA_CHAIN = "solana"
    MAX_ADDRESSES = 25
    TOKEN_PROGRAM_IDS = frozenset(
        {
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "TokenzQdBNbLqP5VEhdk7u6RRsJbMpbB5R2r7mS1rG",
        }
    )
    _BASE58_ALPHABET = frozenset("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

    def __init__(self, transport: SolanaAccountTransport | None = None) -> None:
        self.transport = transport or ReadOnlyJsonRpcTransport(
            self.RPC_URL,
            "api.mainnet-beta.solana.com",
            frozenset({"getMultipleAccounts"}),
        )

    def collect_mint_authorities(
        self,
        token_addresses: tuple[str, ...],
        *,
        collected_at: datetime | None = None,
    ) -> CollectionBatch:
        """Record finalized mint/freeze authority state for bounded discovered mints.

        This observes exactly one public account-read method.  It does not inspect
        holder concentration, transfer extensions, quotes, swaps, simulations, or
        transactions, so it cannot make a token sandbox-eligible by itself.
        """
        if not token_addresses or len(token_addresses) > self.MAX_ADDRESSES:
            raise ValueError(f"token_addresses must contain 1 to {self.MAX_ADDRESSES} addresses")
        if len(set(token_addresses)) != len(token_addresses):
            raise ValueError("token_addresses must be unique")
        if any(not _is_solana_pubkey(address) for address in token_addresses):
            raise ValueError("token_addresses must be base58 Solana public keys")
        received_at = require_aware(collected_at, "collected_at") if collected_at else utc_now()
        response = self.transport.call(
            "getMultipleAccounts",
            [list(token_addresses), {"commitment": "finalized", "encoding": "base64"}],
        )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("Solana getMultipleAccounts response is missing result")
        context = result.get("context")
        values = result.get("value")
        if not isinstance(context, Mapping) or not isinstance(context.get("slot"), int):
            raise ValueError("Solana getMultipleAccounts response is missing finalized slot")
        if not isinstance(values, list) or len(values) != len(token_addresses):
            raise ValueError("Solana getMultipleAccounts response did not preserve address order")
        events = tuple(
            self._authority_event(address, raw_account, context["slot"], received_at)
            for address, raw_account in zip(token_addresses, values, strict=True)
        )
        return CollectionBatch(
            self.venue,
            events=events,
            metadata={
                "requested_addresses": len(token_addresses),
                "finalized_slot": context["slot"],
                "wallet_or_transaction_authority": False,
            },
        )

    def _authority_event(
        self, token_address: str, raw_account: object, slot: int, received_at: datetime
    ) -> MarketEvent:
        account = _parse_mint_account(raw_account)
        observed = account is not None
        mint_authority_active = account[0] if account is not None else None
        freeze_authority_active = account[1] if account is not None else None
        account_owner = account[2] if account is not None else None
        reasons = [
            "holder_concentration_unobserved",
            "transfer_behavior_unobserved",
            "round_trip_simulation_unobserved",
        ]
        if not observed:
            reasons.insert(0, "onchain_authorities_unobserved")
        if mint_authority_active is True:
            reasons.append("mint_authority_active")
        if freeze_authority_active is True:
            reasons.append("freeze_authority_active")
        payload = {
            "chain_id": self.SOLANA_CHAIN,
            "token_address": token_address,
            "finalized_slot": slot,
            "account_owner": account_owner,
            "mint_authority_active": mint_authority_active,
            "freeze_authority_active": freeze_authority_active,
            "onchain_authorities_observed": observed,
            "holder_concentration_observed": False,
            "transfer_behavior_observed": False,
            "round_trip_simulation_observed": False,
            "source_method": "getMultipleAccounts",
            "source_documentation_url": self.RPC_DOCUMENTATION_URL,
            "token_documentation_url": self.TOKEN_DOCUMENTATION_URL,
            "observation_received_at": received_at.isoformat(),
            "safety_status": "blocked_unverified",
            "safety_reasons": tuple(reasons),
            "wallet_or_transaction_authority": False,
            "forecast_created": False,
            "shadow_intent_created": False,
        }
        return MarketEvent(
            stable_event_id(
                "solana:mint-authority-observation",
                {"token_address": token_address, "slot": slot, "payload": payload},
            ),
            MarketEventType.ONCHAIN_STATE,
            self.venue,
            f"dexscreener:memecoin:{self.SOLANA_CHAIN}:{token_address}",
            received_at,
            received_at,
            "solana-rpc-get-multiple-accounts-finalized-v1",
            payload,
            ingested_at=received_at,
        )


def _is_solana_pubkey(value: object) -> bool:
    return (
        isinstance(value, str)
        and 32 <= len(value) <= 44
        and all(character in SolanaMintAuthorityCollector._BASE58_ALPHABET for character in value)
    )


def _parse_mint_account(raw_account: object) -> tuple[bool, bool, str] | None:
    if not isinstance(raw_account, Mapping):
        return None
    owner = raw_account.get("owner")
    data = raw_account.get("data")
    if owner not in SolanaMintAuthorityCollector.TOKEN_PROGRAM_IDS:
        return None
    if not isinstance(data, list) or len(data) != 2 or data[1] != "base64" or not isinstance(data[0], str):
        return None
    try:
        decoded = base64.b64decode(data[0], validate=True)
    except (ValueError, TypeError):
        return None
    if len(decoded) < 82:
        return None
    mint_tag = int.from_bytes(decoded[0:4], "little")
    freeze_tag = int.from_bytes(decoded[46:50], "little")
    if mint_tag not in {0, 1} or freeze_tag not in {0, 1}:
        return None
    return mint_tag == 1, freeze_tag == 1, owner
