from __future__ import annotations

from datetime import datetime
import math

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import require_object, require_string, stable_event_id
from trading_bot.data.http import ReadOnlyHttpTransport, ReadOnlyTransport
from trading_bot.data.quality import inspect_events
from trading_bot.data.schemas import CollectionBatch


class DexscreenerCollector:
    venue = "dexscreener"
    PROFILE_ENDPOINT = "/token-profiles/latest/v1"
    TOKEN_PAIRS_ENDPOINT = "/tokens/v1/solana/"
    PROFILE_DOCUMENTATION_URL = "https://docs.dexscreener.com/api/reference"
    SOLANA_CHAIN = "solana"
    SAFETY_REASONS = (
        "onchain_authorities_unobserved",
        "holder_concentration_unobserved",
        "transfer_behavior_unobserved",
        "round_trip_simulation_unobserved",
    )

    def __init__(self, transport: ReadOnlyTransport | None = None) -> None:
        self.transport = transport or ReadOnlyHttpTransport(
            "https://api.dexscreener.com", "api.dexscreener.com"
        )

    def collect_token_profiles(
        self,
        *,
        collected_at: datetime | None = None,
        limit: int = 25,
        include_pool_observations: bool = False,
    ) -> CollectionBatch:
        """Collect public discoveries and, optionally, untrusted pool snapshots.

        Pool data is only a point-in-time market observation. It cannot satisfy any
        authority, holder, transfer, or round-trip gate and is deliberately not a
        forecast, safety snapshot, or shadow intent.
        """
        if not 1 <= limit <= 100:
            raise ValueError("token profile limit must be between 1 and 100")
        if type(include_pool_observations) is not bool:
            raise ValueError("include_pool_observations must be boolean")
        override = require_aware(collected_at, "collected_at") if collected_at else None
        raw_profiles = self.transport.get_json_array(self.PROFILE_ENDPOINT)
        received_at = override or utc_now()
        instruments: list[Instrument] = []
        events: list[MarketEvent] = []
        token_addresses: list[str] = []
        for raw in raw_profiles:
            profile = require_object(raw, "token profile")
            chain_id = require_string(profile.get("chainId"), "token profile.chainId").lower()
            if chain_id != self.SOLANA_CHAIN:
                continue
            token_address = require_string(
                profile.get("tokenAddress"), "token profile.tokenAddress"
            )
            instrument = Instrument(
                f"dexscreener:memecoin:{chain_id}:{token_address}",
                self.venue,
                token_address,
                AssetClass.MEMECOIN,
                "USD",
                metadata={
                    "chain_id": chain_id,
                    "discovery_source_url": self.PROFILE_DOCUMENTATION_URL,
                },
            )
            raw_profile = dict(profile)
            event = MarketEvent(
                stable_event_id(
                    "dexscreener:token-profile",
                    {
                        "token_address": token_address,
                        "profile": raw_profile,
                        "received_at": received_at,
                    },
                ),
                MarketEventType.ONCHAIN_STATE,
                self.venue,
                instrument.instrument_id,
                received_at,
                received_at,
                "dexscreener-public-token-profile-v1",
                {
                    "chain_id": chain_id,
                    "token_address": token_address,
                    "raw_profile": raw_profile,
                    "source_endpoint": self.PROFILE_ENDPOINT,
                    "source_documentation_url": self.PROFILE_DOCUMENTATION_URL,
                    "safety_status": "blocked_unverified",
                    "safety_reasons": self.SAFETY_REASONS,
                    "onchain_authorities_observed": False,
                    "holder_concentration_observed": False,
                    "transfer_behavior_observed": False,
                    "round_trip_simulation_observed": False,
                    "wallet_or_transaction_authority": False,
                },
                ingested_at=received_at,
            )
            instruments.append(instrument)
            events.append(event)
            if token_address not in token_addresses:
                token_addresses.append(token_address)
            if len(events) >= limit:
                break
        if include_pool_observations and token_addresses:
            events.extend(self._collect_pool_observations(token_addresses, received_at))
        return CollectionBatch(
            self.venue,
            tuple(instruments),
            tuple(events),
            inspect_events(events),
            metadata={
                "public_endpoint": True,
                "source_documentation_url": self.PROFILE_DOCUMENTATION_URL,
                "solana_profiles_seen": len(token_addresses),
                "pool_observations_seen": len(events) - len(token_addresses),
                "pool_observations_enabled": include_pool_observations,
                "safety_status": "blocked_unverified",
                "wallet_or_transaction_authority": False,
            },
        )

    def _collect_pool_observations(
        self, token_addresses: list[str], received_at: datetime
    ) -> list[MarketEvent]:
        # The documented batch endpoint accepts at most 30 comma-separated token
        # addresses. The profile job is capped at 25, preserving a single bounded
        # public read with no wallet, RPC, or transaction capability.
        raw_pairs = self.transport.get_json_array(
            f"{self.TOKEN_PAIRS_ENDPOINT}{','.join(token_addresses)}"
        )
        candidates: dict[str, tuple[float, str, dict[str, object]]] = {}
        requested = set(token_addresses)
        for item in raw_pairs:
            pair = require_object(item, "token pair")
            if str(pair.get("chainId", "")).lower() != self.SOLANA_CHAIN:
                continue
            pair_address = pair.get("pairAddress")
            base_token = pair.get("baseToken")
            quote_token = pair.get("quoteToken")
            if not isinstance(pair_address, str) or not pair_address:
                continue
            base_address = base_token.get("address") if isinstance(base_token, dict) else None
            quote_address = quote_token.get("address") if isinstance(quote_token, dict) else None
            matched = next(
                (
                    address
                    for address in (base_address, quote_address)
                    if isinstance(address, str) and address in requested
                ),
                None,
            )
            if matched is None:
                continue
            liquidity = pair.get("liquidity")
            liquidity_usd = liquidity.get("usd") if isinstance(liquidity, dict) else None
            try:
                liquidity_value = float(liquidity_usd)
            except (TypeError, ValueError):
                liquidity_value = -1.0
            if not math.isfinite(liquidity_value) or liquidity_value < 0:
                liquidity_value = -1.0
            raw_pair = dict(pair)
            candidate = (liquidity_value, pair_address, raw_pair)
            current = candidates.get(matched)
            if current is None or candidate[:2] > current[:2]:
                candidates[matched] = candidate

        observations: list[MarketEvent] = []
        for token_address in token_addresses:
            selected = candidates.get(token_address)
            if selected is None:
                continue
            _, pair_address, raw_pair = selected
            instrument_id = f"dexscreener:memecoin:{self.SOLANA_CHAIN}:{token_address}"
            observations.append(
                MarketEvent(
                    stable_event_id(
                        "dexscreener:pool-observation",
                        {
                            "token_address": token_address,
                            "pair_address": pair_address,
                            "pair": raw_pair,
                            "received_at": received_at,
                        },
                    ),
                    MarketEventType.ONCHAIN_STATE,
                    self.venue,
                    instrument_id,
                    received_at,
                    received_at,
                    "dexscreener-public-token-pairs-v1",
                    {
                        "chain_id": self.SOLANA_CHAIN,
                        "token_address": token_address,
                        "pair_address": pair_address,
                        "raw_pair": raw_pair,
                        "source_endpoint": f"{self.TOKEN_PAIRS_ENDPOINT}{{addresses}}",
                        "source_documentation_url": self.PROFILE_DOCUMENTATION_URL,
                        "observation_received_at": received_at.isoformat(),
                        "safety_status": "blocked_unverified",
                        "safety_reasons": self.SAFETY_REASONS,
                        "onchain_authorities_observed": False,
                        "holder_concentration_observed": False,
                        "transfer_behavior_observed": False,
                        "round_trip_simulation_observed": False,
                        "wallet_or_transaction_authority": False,
                        "forecast_created": False,
                        "shadow_intent_created": False,
                    },
                    ingested_at=received_at,
                )
            )
        return observations
