from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Mapping, Protocol
from urllib.parse import urlsplit

from trading_bot.core.schemas import MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import is_valid_solana_public_key, stable_event_id
from trading_bot.data.http import ReadOnlyJsonRpcTransport
from trading_bot.data.schemas import CollectionBatch


class SolanaAccountTransport(Protocol):
    def call(self, method: str, params: list[object]) -> Mapping[str, object]:
        ...


class SolanaMintAuthorityCollector:
    """Observe public SPL mint controls without any wallet or transaction access."""

    venue = "solana"
    RPC_URL = "https://api.mainnet-beta.solana.com"
    READ_ONLY_RPC_URL_ENV = "SOLANA_READ_ONLY_RPC_URL"
    RPC_DOCUMENTATION_URL = "https://solana.com/docs/rpc/http/getmultipleaccounts"
    LARGEST_ACCOUNTS_DOCUMENTATION_URL = "https://solana.com/docs/rpc/http/gettokenlargestaccounts"
    TOKEN_SUPPLY_DOCUMENTATION_URL = "https://solana.com/docs/rpc/http/gettokensupply"
    SIGNATURES_DOCUMENTATION_URL = "https://solana.com/docs/rpc/http/getsignaturesforaddress"
    TOKEN_DOCUMENTATION_URL = "https://solana.com/docs/tokens/basics"
    SOLANA_CHAIN = "solana"
    MAX_ADDRESSES = 25
    MAX_HOLDER_CONCENTRATION_ADDRESSES = 25
    MAX_HOLDER_ACTIVITY_ADDRESSES = 10
    HOLDER_ACTIVITY_ACCOUNT_SAMPLE = 2
    HOLDER_ACTIVITY_SIGNATURE_LIMIT = 10
    # Public RPC documentation publishes a per-method rate limit. Pace every
    # individual finalized read well below that limit. This is operational
    # backpressure only: it neither changes selected addresses nor treats a
    # partial observation as safe.
    HOLDER_REQUEST_SPACING_SECONDS = 0.5
    TOKEN_PROGRAM_IDS = frozenset(
        {
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "TokenzQdBNbLqP5VEhdk7u6RRsJbMpbB5R2r7mS1rG",
        }
    )
    TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdk7u6RRsJbMpbB5R2r7mS1rG"
    BASE_MINT_BYTES = 82
    MAX_TLV_EXTENSIONS = 64
    # The official Token-2022 ExtensionType enum is repr(u16), so these
    # discriminants are part of the public account format rather than a venue
    # heuristic.  Only controls that can materially affect a transfer are
    # surfaced as booleans; every other extension remains explicitly recorded.
    TRANSFER_FEE_CONFIG = 1
    DEFAULT_ACCOUNT_STATE = 6
    NON_TRANSFERABLE = 9
    PERMANENT_DELEGATE = 12
    TRANSFER_HOOK = 14
    PAUSABLE = 26
    TRANSFER_CONTROL_EXTENSION_TYPES = frozenset(
        {
            TRANSFER_FEE_CONFIG,
            DEFAULT_ACCOUNT_STATE,
            NON_TRANSFERABLE,
            PERMANENT_DELEGATE,
            TRANSFER_HOOK,
            PAUSABLE,
        }
    )
    def __init__(
        self,
        transport: SolanaAccountTransport | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.transport = transport or self._read_only_transport(environment)

    @classmethod
    def is_valid_mint_address(cls, value: object) -> bool:
        """Accept only 32-byte base58 public keys before a bounded RPC read."""
        return is_valid_solana_public_key(value)

    @classmethod
    def _read_only_transport(
        cls, environment: Mapping[str, str] | None = None
    ) -> ReadOnlyJsonRpcTransport:
        values = os.environ if environment is None else environment
        configured_endpoint = values.get(cls.READ_ONLY_RPC_URL_ENV, "").strip()
        endpoint = configured_endpoint or cls.RPC_URL
        parts = urlsplit(endpoint)
        if (
            parts.scheme != "https"
            or parts.hostname is None
            or parts.port not in (None, 443)
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError(
                f"{cls.READ_ONLY_RPC_URL_ENV} must be an HTTPS endpoint without userinfo"
            )
        return ReadOnlyJsonRpcTransport(
            endpoint,
            parts.hostname,
            frozenset({
                "getMultipleAccounts",
                "getSignaturesForAddress",
                "getTokenLargestAccounts",
                "getTokenSupply",
            }),
            allow_endpoint_path=bool(configured_endpoint),
        )

    def collect_mint_authorities(
        self,
        token_addresses: tuple[str, ...],
        *,
        collected_at: datetime | None = None,
    ) -> CollectionBatch:
        """Record finalized mint/freeze authority state for bounded discovered mints.

        This observes exactly one public account-read method. It structurally
        enumerates Token-2022 mint extensions but does not inspect holders,
        quotes, swaps, simulations, or transactions, so it cannot make a token
        sandbox-eligible by itself.
        """
        if not token_addresses or len(token_addresses) > self.MAX_ADDRESSES:
            raise ValueError(f"token_addresses must contain 1 to {self.MAX_ADDRESSES} addresses")
        if len(set(token_addresses)) != len(token_addresses):
            raise ValueError("token_addresses must be unique")
        if any(not self.is_valid_mint_address(address) for address in token_addresses):
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

    def collect_holder_concentrations(
        self,
        token_addresses: tuple[str, ...],
        *,
        collected_at: datetime | None = None,
    ) -> CollectionBatch:
        """Record bounded finalized holder concentration without identifying holders.

        The two documented RPC reads must report the same finalized slot before
        concentration is marked observed. This avoids presenting a supply and
        holder list from different ledger states as a single snapshot.
        """
        if (
            not token_addresses
            or len(token_addresses) > self.MAX_HOLDER_CONCENTRATION_ADDRESSES
        ):
            raise ValueError(
                "token_addresses must contain 1 to "
                f"{self.MAX_HOLDER_CONCENTRATION_ADDRESSES} addresses"
            )
        if len(set(token_addresses)) != len(token_addresses):
            raise ValueError("token_addresses must be unique")
        if any(not self.is_valid_mint_address(address) for address in token_addresses):
            raise ValueError("token_addresses must be base58 Solana public keys")
        received_at = require_aware(collected_at, "collected_at") if collected_at else utc_now()
        events = []
        for index, address in enumerate(token_addresses):
            events.append(self._holder_concentration_event(address, received_at))
            if index + 1 < len(token_addresses):
                sleep(self.HOLDER_REQUEST_SPACING_SECONDS)
        return CollectionBatch(
            self.venue,
            events=tuple(events),
            metadata={
                "requested_addresses": len(token_addresses),
                "wallet_or_transaction_authority": False,
            },
        )

    def collect_holder_activity(
        self,
        token_addresses: tuple[str, ...],
        *,
        collected_at: datetime | None = None,
    ) -> CollectionBatch:
        """Record bounded finalized account activity without retaining identities.

        This only checks whether a small public sample of the largest token
        accounts has recent successful finalized transaction references.  It
        cannot establish future transferability, does not inspect or submit a
        transaction, and never satisfies the transfer-behavior or round-trip
        safety gates.
        """
        if not token_addresses or len(token_addresses) > self.MAX_HOLDER_ACTIVITY_ADDRESSES:
            raise ValueError(
                "token_addresses must contain 1 to "
                f"{self.MAX_HOLDER_ACTIVITY_ADDRESSES} addresses"
            )
        if len(set(token_addresses)) != len(token_addresses):
            raise ValueError("token_addresses must be unique")
        if any(not self.is_valid_mint_address(address) for address in token_addresses):
            raise ValueError("token_addresses must be base58 Solana public keys")
        received_at = require_aware(collected_at, "collected_at") if collected_at else utc_now()
        events = []
        for index, address in enumerate(token_addresses):
            events.append(self._holder_activity_event(address, received_at))
            if index + 1 < len(token_addresses):
                sleep(self.HOLDER_REQUEST_SPACING_SECONDS)
        return CollectionBatch(
            self.venue,
            events=tuple(events),
            metadata={
                "requested_addresses": len(token_addresses),
                "sampled_token_accounts_per_mint": self.HOLDER_ACTIVITY_ACCOUNT_SAMPLE,
                "wallet_or_transaction_authority": False,
            },
        )

    def _holder_activity_event(self, token_address: str, received_at: datetime) -> MarketEvent:
        largest = _parse_largest_accounts(
            self.transport.call(
                "getTokenLargestAccounts", [token_address, {"commitment": "finalized"}]
            ),
            require_addresses=True,
        )
        sampled_accounts = (
            largest.addresses[: self.HOLDER_ACTIVITY_ACCOUNT_SAMPLE]
            if largest is not None
            else ()
        )
        successful_references = 0
        observed = largest is not None
        for account_address in sampled_accounts:
            response = self.transport.call(
                "getSignaturesForAddress",
                [
                    account_address,
                    {
                        "commitment": "finalized",
                        "limit": self.HOLDER_ACTIVITY_SIGNATURE_LIMIT,
                    },
                ],
            )
            signatures = _parse_finalized_signature_count(response)
            if signatures is None:
                observed = False
                continue
            successful_references += signatures
        reasons = [
            "onchain_authorities_unobserved",
            "holder_concentration_unobserved",
            "transfer_behavior_unobserved",
            "round_trip_simulation_unobserved",
        ]
        if not observed:
            reasons.append("holder_activity_unobserved")
        elif successful_references == 0:
            reasons.append("no_recent_finalized_holder_activity")
        payload = {
            "chain_id": self.SOLANA_CHAIN,
            "token_address": token_address,
            "holder_activity_observed": observed,
            "sampled_token_account_count": len(sampled_accounts),
            "sampled_successful_finalized_reference_count": successful_references,
            "sampled_signature_limit_per_account": self.HOLDER_ACTIVITY_SIGNATURE_LIMIT,
            "source_methods": ("getTokenLargestAccounts", "getSignaturesForAddress"),
            "largest_accounts_documentation_url": self.LARGEST_ACCOUNTS_DOCUMENTATION_URL,
            "signatures_documentation_url": self.SIGNATURES_DOCUMENTATION_URL,
            "observation_received_at": received_at.isoformat(),
            "safety_status": "blocked_unverified",
            "safety_reasons": tuple(reasons),
            "onchain_authorities_observed": False,
            "holder_concentration_observed": False,
            "transfer_behavior_observed": False,
            "round_trip_simulation_observed": False,
            "wallet_or_transaction_authority": False,
            "forecast_created": False,
            "shadow_intent_created": False,
        }
        return MarketEvent(
            stable_event_id(
                "solana:holder-activity-observation",
                {"token_address": token_address, "payload": payload},
            ),
            MarketEventType.ONCHAIN_STATE,
            self.venue,
            f"dexscreener:memecoin:{self.SOLANA_CHAIN}:{token_address}",
            received_at,
            received_at,
            "solana-rpc-finalized-holder-activity-v1",
            payload,
            ingested_at=received_at,
        )

    def _holder_concentration_event(
        self, token_address: str, received_at: datetime
    ) -> MarketEvent:
        largest_response = self.transport.call(
            "getTokenLargestAccounts", [token_address, {"commitment": "finalized"}]
        )
        sleep(self.HOLDER_REQUEST_SPACING_SECONDS)
        supply_response = self.transport.call(
            "getTokenSupply", [token_address, {"commitment": "finalized"}]
        )
        largest = _parse_largest_accounts(largest_response)
        supply = _parse_token_supply(supply_response)
        observed = (
            largest is not None
            and supply is not None
            and largest.slot == supply.slot
            and supply.amount > 0
            and largest.top_amount <= supply.amount
            and sum(largest.amounts) <= supply.amount
        )
        top_share_bps = (
            largest.top_amount * 10_000 // supply.amount if observed and largest is not None and supply is not None else None
        )
        top_twenty_share_bps = (
            sum(largest.amounts) * 10_000 // supply.amount
            if observed and largest is not None and supply is not None
            else None
        )
        reasons = [
            "onchain_authorities_unobserved",
            "transfer_behavior_unobserved",
            "round_trip_simulation_unobserved",
        ]
        if not observed:
            reasons.insert(1, "holder_concentration_unobserved")
        payload = {
            "chain_id": self.SOLANA_CHAIN,
            "token_address": token_address,
            "finalized_slot": largest.slot if observed and largest is not None else None,
            "largest_accounts_finalized_slot": largest.slot if largest is not None else None,
            "supply_finalized_slot": supply.slot if supply is not None else None,
            "holder_concentration_observed": observed,
            "reported_largest_accounts": len(largest.amounts) if largest is not None else 0,
            "top_holder_share_bps": top_share_bps,
            "top_twenty_holder_share_bps": top_twenty_share_bps,
            "source_methods": ("getTokenLargestAccounts", "getTokenSupply"),
            "largest_accounts_documentation_url": self.LARGEST_ACCOUNTS_DOCUMENTATION_URL,
            "token_supply_documentation_url": self.TOKEN_SUPPLY_DOCUMENTATION_URL,
            "observation_received_at": received_at.isoformat(),
            "safety_status": "blocked_unverified",
            "safety_reasons": tuple(reasons),
            "wallet_or_transaction_authority": False,
            "forecast_created": False,
            "shadow_intent_created": False,
        }
        return MarketEvent(
            stable_event_id(
                "solana:holder-concentration-observation",
                {"token_address": token_address, "payload": payload},
            ),
            MarketEventType.ONCHAIN_STATE,
            self.venue,
            f"dexscreener:memecoin:{self.SOLANA_CHAIN}:{token_address}",
            received_at,
            received_at,
            "solana-rpc-token-holder-concentration-finalized-v1",
            payload,
            ingested_at=received_at,
        )

    def _authority_event(
        self, token_address: str, raw_account: object, slot: int, received_at: datetime
    ) -> MarketEvent:
        account = _parse_mint_account(raw_account)
        observed = account is not None
        mint_authority_active = account.mint_authority_active if account is not None else None
        freeze_authority_active = account.freeze_authority_active if account is not None else None
        account_owner = account.account_owner if account is not None else None
        reasons = [
            "holder_concentration_unobserved",
            "round_trip_simulation_unobserved",
        ]
        if not observed:
            reasons.insert(0, "onchain_authorities_unobserved")
            reasons.insert(1, "transfer_behavior_unobserved")
        elif not account.transfer_behavior_observed:
            reasons.insert(1, "transfer_behavior_unobserved")
        if mint_authority_active is True:
            reasons.append("mint_authority_active")
        if freeze_authority_active is True:
            reasons.append("freeze_authority_active")
        if account is not None and account.permanent_delegate_active:
            reasons.append("permanent_delegate_active")
        if account is not None and account.transfer_hook_active:
            reasons.append("transfer_hook_active")
        if account is not None and account.pausable_active:
            reasons.append("pausable_transfer_control_active")
        if account is not None and account.non_transferable_active:
            reasons.append("non_transferable_extension_active")
        if account is not None and account.default_account_frozen:
            reasons.append("default_token_accounts_frozen")
        if account is not None and account.unknown_extension_types:
            reasons.append("unknown_token_2022_extension_present")
        payload = {
            "chain_id": self.SOLANA_CHAIN,
            "token_address": token_address,
            "finalized_slot": slot,
            "account_owner": account_owner,
            "mint_authority_active": mint_authority_active,
            "freeze_authority_active": freeze_authority_active,
            "onchain_authorities_observed": observed,
            "token_2022_extensions_observed": account.token_2022_extensions_observed
            if account is not None
            else False,
            "token_2022_extension_types": account.extension_types if account is not None else (),
            "unknown_token_2022_extension_types": account.unknown_extension_types
            if account is not None
            else (),
            "permanent_delegate_active": account.permanent_delegate_active
            if account is not None
            else None,
            "transfer_hook_active": account.transfer_hook_active if account is not None else None,
            "pausable_active": account.pausable_active if account is not None else None,
            "transfers_currently_paused": account.transfers_currently_paused
            if account is not None
            else None,
            "non_transferable_active": account.non_transferable_active
            if account is not None
            else None,
            "transfer_fee_config_active": account.transfer_fee_config_active
            if account is not None
            else None,
            "default_account_frozen": account.default_account_frozen
            if account is not None
            else None,
            "holder_concentration_observed": False,
            "transfer_behavior_observed": account.transfer_behavior_observed
            if account is not None
            else False,
            "round_trip_simulation_observed": False,
            "source_method": "getMultipleAccounts",
            "source_documentation_url": self.RPC_DOCUMENTATION_URL,
            "token_documentation_url": self.TOKEN_DOCUMENTATION_URL,
            "token_extensions_documentation_url": (
                "https://solana.com/docs/tokens/extensions"
            ),
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
            "solana-rpc-get-multiple-accounts-finalized-v2",
            payload,
            ingested_at=received_at,
        )


@dataclass(frozen=True)
class _MintAccountObservation:
    mint_authority_active: bool
    freeze_authority_active: bool
    account_owner: str
    transfer_behavior_observed: bool
    token_2022_extensions_observed: bool
    extension_types: tuple[int, ...]
    unknown_extension_types: tuple[int, ...]
    permanent_delegate_active: bool
    transfer_hook_active: bool
    pausable_active: bool
    transfers_currently_paused: bool
    non_transferable_active: bool
    transfer_fee_config_active: bool
    default_account_frozen: bool


@dataclass(frozen=True)
class _LargestAccountsObservation:
    slot: int
    amounts: tuple[int, ...]
    addresses: tuple[str, ...] = ()

    @property
    def top_amount(self) -> int:
        return max(self.amounts, default=0)


@dataclass(frozen=True)
class _TokenSupplyObservation:
    slot: int
    amount: int


def _parse_largest_accounts(
    response: Mapping[str, object], *, require_addresses: bool = False
) -> _LargestAccountsObservation | None:
    result = response.get("result")
    if not isinstance(result, Mapping):
        return None
    context = result.get("context")
    values = result.get("value")
    if not isinstance(context, Mapping) or not isinstance(context.get("slot"), int):
        return None
    if not isinstance(values, list) or not values:
        return None
    amounts: list[int] = []
    addresses: list[str] = []
    for value in values:
        if not isinstance(value, Mapping):
            return None
        amount = _parse_nonnegative_amount(value.get("amount"))
        if amount is None:
            return None
        amounts.append(amount)
        address = value.get("address")
        if require_addresses and not SolanaMintAuthorityCollector.is_valid_mint_address(address):
            return None
        if isinstance(address, str):
            addresses.append(address)
    return _LargestAccountsObservation(context["slot"], tuple(amounts), tuple(addresses))


def _parse_finalized_signature_count(response: Mapping[str, object]) -> int | None:
    result = response.get("result")
    if not isinstance(result, list):
        return None
    count = 0
    for item in result:
        if not isinstance(item, Mapping):
            return None
        if item.get("confirmationStatus") != "finalized":
            return None
        if item.get("err") is None:
            count += 1
    return count


def _parse_token_supply(response: Mapping[str, object]) -> _TokenSupplyObservation | None:
    result = response.get("result")
    if not isinstance(result, Mapping):
        return None
    context = result.get("context")
    value = result.get("value")
    if not isinstance(context, Mapping) or not isinstance(context.get("slot"), int):
        return None
    if not isinstance(value, Mapping):
        return None
    amount = _parse_nonnegative_amount(value.get("amount"))
    if amount is None:
        return None
    return _TokenSupplyObservation(context["slot"], amount)


def _parse_nonnegative_amount(value: object) -> int | None:
    if not isinstance(value, str) or not value.isdecimal():
        return None
    return int(value)


def _parse_mint_account(raw_account: object) -> _MintAccountObservation | None:
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
    if len(decoded) < SolanaMintAuthorityCollector.BASE_MINT_BYTES:
        return None
    mint_tag = int.from_bytes(decoded[0:4], "little")
    freeze_tag = int.from_bytes(decoded[46:50], "little")
    if mint_tag not in {0, 1} or freeze_tag not in {0, 1}:
        return None
    if owner != SolanaMintAuthorityCollector.TOKEN_2022_PROGRAM_ID:
        return _MintAccountObservation(
            mint_tag == 1,
            freeze_tag == 1,
            owner,
            True,
            False,
            (),
            (),
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )
    extension_data = decoded[SolanaMintAuthorityCollector.BASE_MINT_BYTES :]
    extension_types = _parse_tlv_extension_types(extension_data)
    if extension_types is None:
        return None
    extension_set = frozenset(extension_types)
    permanent_delegate_data = _extension_data(
        extension_data, SolanaMintAuthorityCollector.PERMANENT_DELEGATE
    )
    transfer_hook_data = _extension_data(
        extension_data, SolanaMintAuthorityCollector.TRANSFER_HOOK
    )
    pausable_data = _extension_data(extension_data, SolanaMintAuthorityCollector.PAUSABLE)
    non_transferable_data = _extension_data(
        extension_data, SolanaMintAuthorityCollector.NON_TRANSFERABLE
    )
    default_account_state_data = _extension_data(
        extension_data, SolanaMintAuthorityCollector.DEFAULT_ACCOUNT_STATE
    )
    if (
        permanent_delegate_data is not None and len(permanent_delegate_data) != 32
    ) or (transfer_hook_data is not None and len(transfer_hook_data) != 64) or (
        pausable_data is not None
        and (len(pausable_data) != 33 or pausable_data[32] not in {0, 1})
    ) or (non_transferable_data is not None and non_transferable_data):
        return None
    if default_account_state_data is not None and (
        len(default_account_state_data) != 1 or default_account_state_data[0] not in {1, 2}
    ):
        return None
    return _MintAccountObservation(
        mint_tag == 1,
        freeze_tag == 1,
        owner,
        True,
        True,
        extension_types,
        tuple(
            extension_type
            for extension_type in extension_types
            if extension_type not in SolanaMintAuthorityCollector.TRANSFER_CONTROL_EXTENSION_TYPES
        ),
        _optional_pubkey_is_set(permanent_delegate_data),
        _transfer_hook_is_set(transfer_hook_data),
        _optional_pubkey_is_set(pausable_data),
        bool(pausable_data is not None and len(pausable_data) >= 33 and pausable_data[32]),
        SolanaMintAuthorityCollector.NON_TRANSFERABLE in extension_set,
        SolanaMintAuthorityCollector.TRANSFER_FEE_CONFIG in extension_set,
        bool(default_account_state_data is not None and default_account_state_data[0] == 2),
    )


def _parse_tlv_extension_types(data: bytes) -> tuple[int, ...] | None:
    """Decode bounded Token-2022 TLV headers after a base mint account."""
    extensions: list[int] = []
    offset = 0
    while offset < len(data):
        if data[offset:] == bytes(len(data) - offset):
            break
        if (
            len(data) - offset < 4
            or len(extensions) >= SolanaMintAuthorityCollector.MAX_TLV_EXTENSIONS
        ):
            return None
        extension_type = int.from_bytes(data[offset : offset + 2], "little")
        length = int.from_bytes(data[offset + 2 : offset + 4], "little")
        offset += 4
        if extension_type == 0:
            return None
        if length > len(data) - offset:
            return None
        extensions.append(extension_type)
        offset += length
    return tuple(extensions)


def _extension_data(data: bytes, target_type: int) -> bytes | None:
    offset = 0
    while offset < len(data):
        if data[offset:] == bytes(len(data) - offset) or len(data) - offset < 4:
            return None
        extension_type = int.from_bytes(data[offset : offset + 2], "little")
        length = int.from_bytes(data[offset + 2 : offset + 4], "little")
        offset += 4
        if length > len(data) - offset:
            return None
        value = data[offset : offset + length]
        if extension_type == target_type:
            return value
        offset += length
    return None


def _optional_pubkey_is_set(data: bytes | None) -> bool:
    return data is not None and len(data) >= 32 and any(data[:32])


def _transfer_hook_is_set(data: bytes | None) -> bool:
    # TransferHook stores optional authority then optional hook program id.
    return data is not None and len(data) >= 64 and any(data[32:64])
