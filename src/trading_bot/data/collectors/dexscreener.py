from __future__ import annotations

from datetime import datetime

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import require_object, require_string, stable_event_id
from trading_bot.data.http import ReadOnlyHttpTransport, ReadOnlyTransport
from trading_bot.data.quality import inspect_events
from trading_bot.data.schemas import CollectionBatch


class DexscreenerCollector:
    venue = "dexscreener"
    PROFILE_ENDPOINT = "/token-profiles/latest/v1"
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
        self, *, collected_at: datetime | None = None, limit: int = 25
    ) -> CollectionBatch:
        if not 1 <= limit <= 100:
            raise ValueError("token profile limit must be between 1 and 100")
        override = require_aware(collected_at, "collected_at") if collected_at else None
        raw_profiles = self.transport.get_json_array(self.PROFILE_ENDPOINT)
        received_at = override or utc_now()
        instruments: list[Instrument] = []
        events: list[MarketEvent] = []
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
            if len(events) >= limit:
                break
        return CollectionBatch(
            self.venue,
            tuple(instruments),
            tuple(events),
            inspect_events(events),
            metadata={
                "public_endpoint": True,
                "source_documentation_url": self.PROFILE_DOCUMENTATION_URL,
                "solana_profiles_seen": len(events),
                "safety_status": "blocked_unverified",
                "wallet_or_transaction_authority": False,
            },
        )
