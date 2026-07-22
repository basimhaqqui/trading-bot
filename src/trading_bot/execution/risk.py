from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping

from trading_bot.core.schemas import AssetClass, Instrument
from trading_bot.core.serialization import canonical_json, require_aware, utc_now
from trading_bot.execution.schemas import (
    ApprovedOrderIntent,
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    PortfolioSnapshot,
    RiskDecision,
)


@dataclass(frozen=True)
class RiskLimits:
    max_gross_notional: float
    max_instrument_notional: float
    max_venue_notional: float
    asset_class_caps: Mapping[AssetClass, float] = field(default_factory=dict)
    min_liquidation_distance_pct: float = 0.15
    allow_live: bool = False

    def __post_init__(self) -> None:
        if min(
            self.max_gross_notional,
            self.max_instrument_notional,
            self.max_venue_notional,
        ) <= 0:
            raise ValueError("notional limits must be positive")
        if not 0 < self.min_liquidation_distance_pct < 1:
            raise ValueError("liquidation distance must be between 0 and 1")


class ApprovalSigner:
    def __init__(self, key: bytes, *, key_id: str = "risk-v1") -> None:
        if len(key) < 16:
            raise ValueError("risk signing key must be at least 16 bytes")
        self._key = key
        self.key_id = key_id

    def sign_payload(self, payload: object) -> str:
        return hmac.new(
            self._key,
            canonical_json(payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, approval: ApprovedOrderIntent) -> bool:
        unsigned = {
            "intent": approval.intent,
            "decision": approval.decision,
            "signed_at": approval.signed_at,
            "key_id": approval.key_id,
            "approval_expires_at": approval.approval_expires_at,
            "risk_metadata": approval.risk_metadata,
        }
        expected = self.sign_payload(unsigned)
        return hmac.compare_digest(expected, approval.signature)


class RiskGovernor:
    def __init__(self, limits: RiskLimits, signer: ApprovalSigner) -> None:
        self.limits = limits
        self.signer = signer
        self.kill_switch_active = False

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        instrument: Instrument,
        portfolio: PortfolioSnapshot,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = require_aware(now or utc_now(), "now")
        reasons: list[str] = []
        if self.kill_switch_active:
            reasons.append("system kill switch is active")
        if now >= intent.expires_at:
            reasons.append("order intent has expired")
        if intent.environment is ExecutionEnvironment.LIVE and not self.limits.allow_live:
            reasons.append("live execution is disabled")
        if intent.instrument_id != instrument.instrument_id:
            reasons.append("intent and instrument identifiers do not match")
        if intent.venue != instrument.venue or intent.asset_class is not instrument.asset_class:
            reasons.append("intent venue or asset class does not match instrument master")

        positions = {position.instrument_id: position for position in portfolio.positions}
        current = positions.get(intent.instrument_id)
        current_signed = current.signed_notional if current else 0.0
        change = intent.notional if intent.side is OrderSide.BUY else -intent.notional
        projected_instrument = current_signed + change
        if intent.reduce_only and abs(projected_instrument) >= abs(current_signed):
            reasons.append("reduce-only intent does not reduce absolute exposure")

        gross_other = sum(
            abs(position.signed_notional)
            for position in portfolio.positions
            if position.instrument_id != intent.instrument_id
        )
        projected_gross = gross_other + abs(projected_instrument)

        projected_venue = sum(
            abs(position.signed_notional)
            for position in portfolio.positions
            if position.venue == intent.venue and position.instrument_id != intent.instrument_id
        ) + abs(projected_instrument)
        projected_asset = sum(
            abs(position.signed_notional)
            for position in portfolio.positions
            if position.asset_class is intent.asset_class
            and position.instrument_id != intent.instrument_id
        ) + abs(projected_instrument)

        if projected_gross > self.limits.max_gross_notional:
            reasons.append("projected gross notional exceeds system limit")
        if abs(projected_instrument) > self.limits.max_instrument_notional:
            reasons.append("projected instrument notional exceeds limit")
        if projected_venue > self.limits.max_venue_notional:
            reasons.append("projected venue notional exceeds limit")
        asset_cap = self.limits.asset_class_caps.get(intent.asset_class)
        if asset_cap is not None and projected_asset > asset_cap:
            reasons.append(f"projected {intent.asset_class.value} notional exceeds asset cap")
        if intent.notional > portfolio.available_cash and not intent.reduce_only:
            reasons.append("order notional exceeds available cash")

        for position in portfolio.positions:
            distance = position.liquidation_distance_pct
            if distance is not None and distance < self.limits.min_liquidation_distance_pct:
                reasons.append(
                    f"{position.instrument_id} is too close to liquidation for new risk"
                )
                break

        return RiskDecision(
            approved=not reasons,
            intent_id=intent.intent_id,
            reasons=tuple(reasons),
            evaluated_at=now,
            projected_gross_notional=projected_gross,
            projected_venue_notional=projected_venue,
            projected_asset_notional=projected_asset,
        )

    def approve(
        self,
        intent: OrderIntent,
        *,
        instrument: Instrument,
        portfolio: PortfolioSnapshot,
        now: datetime | None = None,
        approval_ttl: timedelta = timedelta(seconds=30),
    ) -> ApprovedOrderIntent:
        now = require_aware(now or utc_now(), "now")
        decision = self.evaluate(intent, instrument=instrument, portfolio=portfolio, now=now)
        if not decision.approved:
            raise PermissionError("; ".join(decision.reasons))
        expires_at = min(intent.expires_at, now + approval_ttl)
        metadata: dict[str, float | str | bool] = {
            "gross_notional": decision.projected_gross_notional,
            "venue_notional": decision.projected_venue_notional,
            "asset_notional": decision.projected_asset_notional,
            "live_enabled": self.limits.allow_live,
        }
        unsigned = {
            "intent": intent,
            "decision": decision,
            "signed_at": now,
            "key_id": self.signer.key_id,
            "approval_expires_at": expires_at,
            "risk_metadata": metadata,
        }
        return ApprovedOrderIntent(
            intent=intent,
            decision=decision,
            signature=self.signer.sign_payload(unsigned),
            signed_at=now,
            key_id=self.signer.key_id,
            approval_expires_at=expires_at,
            risk_metadata=metadata,
        )
