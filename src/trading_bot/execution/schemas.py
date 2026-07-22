from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from trading_bot.core.schemas import AssetClass
from trading_bot.core.serialization import require_aware, utc_now


class ExecutionEnvironment(StrEnum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    strategy_id: str
    model_version: str
    instrument_id: str
    venue: str
    asset_class: AssetClass
    side: OrderSide
    notional: float
    environment: ExecutionEnvironment
    allowed_order_types: tuple[OrderType, ...]
    expires_at: datetime
    max_price: float | None = None
    min_price: float | None = None
    reduce_only: bool = False
    created_at: datetime = field(default_factory=utc_now)
    quantity: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    forecast_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", require_aware(self.created_at, "created_at"))
        object.__setattr__(self, "expires_at", require_aware(self.expires_at, "expires_at"))
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if not all((self.intent_id, self.strategy_id, self.model_version, self.instrument_id, self.venue)):
            raise ValueError("intent identity and routing fields are required")
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if not self.allowed_order_types:
            raise ValueError("at least one order type must be allowed")
        if self.max_price is not None and self.max_price <= 0:
            raise ValueError("max_price must be positive")
        if self.min_price is not None and self.min_price < 0:
            raise ValueError("min_price cannot be negative")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.forecast_id is not None and not self.forecast_id:
            raise ValueError("forecast_id cannot be blank")


@dataclass(frozen=True)
class Position:
    instrument_id: str
    venue: str
    asset_class: AssetClass
    signed_notional: float
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    liquidation_distance_pct: float | None = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_at: datetime
    equity: float
    available_cash: float
    positions: tuple[Position, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_at", require_aware(self.snapshot_at, "snapshot_at"))
        if self.equity <= 0:
            raise ValueError("equity must be positive")
        if self.available_cash < 0:
            raise ValueError("available_cash cannot be negative")


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    intent_id: str
    reasons: tuple[str, ...]
    evaluated_at: datetime
    projected_gross_notional: float
    projected_venue_notional: float
    projected_asset_notional: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluated_at", require_aware(self.evaluated_at, "evaluated_at"))
        if not self.intent_id:
            raise ValueError("intent_id is required")
        if min(
            self.projected_gross_notional,
            self.projected_venue_notional,
            self.projected_asset_notional,
        ) < 0:
            raise ValueError("projected notionals cannot be negative")


@dataclass(frozen=True)
class ApprovedOrderIntent:
    intent: OrderIntent
    decision: RiskDecision
    signature: str
    signed_at: datetime
    key_id: str
    approval_expires_at: datetime
    risk_metadata: Mapping[str, float | str | bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "signed_at", require_aware(self.signed_at, "signed_at"))
        object.__setattr__(
            self,
            "approval_expires_at",
            require_aware(self.approval_expires_at, "approval_expires_at"),
        )
        if not self.decision.approved:
            raise ValueError("cannot wrap a rejected risk decision")
        if not self.signature or not self.key_id:
            raise ValueError("signature and key_id are required")
        if self.approval_expires_at <= self.signed_at:
            raise ValueError("approval must expire after it is signed")
