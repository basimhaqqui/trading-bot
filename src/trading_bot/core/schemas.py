from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping

from trading_bot.core.serialization import require_aware, sha256_digest, utc_now


class AssetClass(StrEnum):
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    PERPETUAL = "perpetual"
    CRYPTO = "crypto"
    MEMECOIN = "memecoin"
    PREDICTION = "prediction"


class MarketEventType(StrEnum):
    TRADE = "trade"
    QUOTE = "quote"
    BOOK_SNAPSHOT = "book_snapshot"
    BOOK_DELTA = "book_delta"
    BAR = "bar"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    CORPORATE_ACTION = "corporate_action"
    FUNDAMENTAL = "fundamental"
    CONTRACT_RULE = "contract_rule"
    ONCHAIN_STATE = "onchain_state"
    NEWS = "news"
    SETTLEMENT = "settlement"


class ForecastKind(StrEnum):
    BINARY_PROBABILITY = "binary_probability"
    RETURN_DISTRIBUTION = "return_distribution"
    VOLATILITY = "volatility"
    FUNDING_RATE = "funding_rate"
    SAFETY_SCORE = "safety_score"
    REGIME = "regime"


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    venue: str
    symbol: str
    asset_class: AssetClass
    quote_currency: str
    multiplier: float = 1.0
    active_from: datetime | None = None
    active_until: datetime | None = None
    expiry: datetime | None = None
    settlement: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.venue or not self.symbol:
            raise ValueError("instrument_id, venue, and symbol are required")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        for name in ("active_from", "active_until", "expiry"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_aware(value, name))
        if self.active_from and self.active_until and self.active_until <= self.active_from:
            raise ValueError("active_until must be after active_from")


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    event_type: MarketEventType
    venue: str
    instrument_id: str
    event_time: datetime
    available_at: datetime
    source: str
    payload: Mapping[str, Any]
    sequence: int | None = None
    ingested_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("event_time", "available_at", "ingested_at"):
            object.__setattr__(self, name, require_aware(getattr(self, name), name))
        if not all((self.event_id, self.venue, self.instrument_id, self.source)):
            raise ValueError("event_id, venue, instrument_id, and source are required")
        if self.available_at < self.event_time:
            raise ValueError("available_at cannot precede event_time")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence cannot be negative")

    @property
    def digest(self) -> str:
        return sha256_digest(self)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    family: str
    market: AssetClass
    mechanism: str
    target: str
    horizon: str
    information_set: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    proposed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposed_at", require_aware(self.proposed_at, "proposed_at"))
        if not all((self.hypothesis_id, self.family, self.mechanism, self.target, self.horizon)):
            raise ValueError("hypothesis identity, mechanism, target, and horizon are required")
        if not self.information_set or not self.invalidation_conditions:
            raise ValueError("hypothesis must define information and invalidation conditions")


@dataclass(frozen=True)
class Forecast:
    forecast_id: str
    specialist_id: str
    model_version: str
    instrument_id: str
    kind: ForecastKind
    generated_at: datetime
    valid_until: datetime
    values: Mapping[str, float | str | bool]
    confidence: float
    uncertainty: Mapping[str, float]
    evidence_event_ids: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", require_aware(self.generated_at, "generated_at"))
        object.__setattr__(self, "valid_until", require_aware(self.valid_until, "valid_until"))
        if self.valid_until <= self.generated_at:
            raise ValueError("valid_until must be after generated_at")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.values:
            raise ValueError("forecast values are required")
        if self.kind is ForecastKind.BINARY_PROBABILITY:
            probability = self.values.get("probability")
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError("binary forecast requires probability between 0 and 1")
        if not self.evidence_event_ids:
            raise ValueError("forecast must cite at least one market event")
