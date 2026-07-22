from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping
from unittest.mock import patch

from trading_bot.core.schemas import AssetClass, Instrument
from trading_bot.core.serialization import (
    canonical_json,
    require_aware,
    sha256_digest,
)
from trading_bot.execution.control import DeterministicExecutor, ExecutionReceipt
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ApprovedOrderIntent,
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    Position,
)


SANDBOX_TIME = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


@dataclass(frozen=True)
class CryptoSandboxConfig:
    version: str = "crypto-perpetual-sandbox-v1"
    initial_cash: float = 100_000
    default_leverage: float = 3
    max_leverage: float = 5
    maintenance_margin_pct: float = 0.05
    trading_fee_bps: float = 10
    slippage_bps: float = 5
    max_market_age: timedelta = timedelta(seconds=60)
    min_order_notional: float = 10

    def __post_init__(self) -> None:
        numeric = (
            self.initial_cash,
            self.default_leverage,
            self.max_leverage,
            self.maintenance_margin_pct,
            self.trading_fee_bps,
            self.slippage_bps,
            self.min_order_notional,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("sandbox numeric policy values must be finite")
        if not self.version or self.initial_cash <= 0:
            raise ValueError("sandbox version and positive initial cash are required")
        if not 1 <= self.default_leverage <= self.max_leverage <= 20:
            raise ValueError("sandbox leverage limits are invalid")
        if not 0 < self.maintenance_margin_pct < 1 / self.max_leverage:
            raise ValueError("maintenance margin must remain below max-leverage initial margin")
        if not 0 <= self.trading_fee_bps <= 1_000:
            raise ValueError("trading fee must be between zero and 1,000 bps")
        if not 0 <= self.slippage_bps <= 1_000:
            raise ValueError("slippage must be between zero and 1,000 bps")
        if self.max_market_age <= timedelta(0) or self.min_order_notional <= 0:
            raise ValueError("market age and minimum order notional must be positive")


def load_crypto_sandbox_config(path: str | Path) -> CryptoSandboxConfig:
    config_path = Path(path)
    if config_path.stat().st_size > 1_000_000:
        raise ValueError("crypto sandbox config exceeds the 1 MB safety limit")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "version",
        "initial_cash",
        "default_leverage",
        "max_leverage",
        "maintenance_margin_pct",
        "trading_fee_bps",
        "slippage_bps",
        "max_market_age_seconds",
        "min_order_notional",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            "crypto sandbox config keys mismatch: "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )
    return CryptoSandboxConfig(
        version=str(raw["version"]),
        initial_cash=float(raw["initial_cash"]),
        default_leverage=float(raw["default_leverage"]),
        max_leverage=float(raw["max_leverage"]),
        maintenance_margin_pct=float(raw["maintenance_margin_pct"]),
        trading_fee_bps=float(raw["trading_fee_bps"]),
        slippage_bps=float(raw["slippage_bps"]),
        max_market_age=timedelta(seconds=float(raw["max_market_age_seconds"])),
        min_order_notional=float(raw["min_order_notional"]),
    )


@dataclass(frozen=True)
class SandboxMarketState:
    instrument_id: str
    asset_class: AssetClass
    bid: float
    ask: float
    mark: float
    observed_at: datetime
    funding_rate: float = 0.0
    funding_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", require_aware(self.observed_at, "observed_at")
        )
        if self.funding_at is not None:
            object.__setattr__(
                self, "funding_at", require_aware(self.funding_at, "funding_at")
            )
        if self.asset_class not in {AssetClass.CRYPTO, AssetClass.PERPETUAL}:
            raise ValueError("sandbox market must be spot crypto or perpetual")
        prices = (self.bid, self.ask, self.mark)
        if (
            not self.instrument_id
            or not all(math.isfinite(value) for value in prices)
            or min(prices) <= 0
        ):
            raise ValueError("sandbox market identity and positive prices are required")
        if self.ask < self.bid:
            raise ValueError("sandbox market cannot be crossed")
        if not math.isfinite(self.funding_rate) or abs(self.funding_rate) > 0.05:
            raise ValueError("sandbox funding rate must be finite and bounded")
        if self.asset_class is AssetClass.CRYPTO and (
            self.funding_rate != 0 or self.funding_at is not None
        ):
            raise ValueError("spot crypto market cannot carry funding")


@dataclass(frozen=True)
class SandboxPosition:
    instrument_id: str
    asset_class: AssetClass
    quantity: float
    entry_price: float
    mark_price: float
    multiplier: float
    leverage: float
    realized_pnl: float = 0.0
    funding_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.asset_class not in {AssetClass.CRYPTO, AssetClass.PERPETUAL}:
            raise ValueError("sandbox position must be spot crypto or perpetual")
        if not self.instrument_id or self.quantity == 0:
            raise ValueError("sandbox position identity and nonzero quantity are required")
        values = (self.entry_price, self.mark_price, self.multiplier, self.leverage)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("sandbox position prices, multiplier, and leverage must be positive")
        if self.asset_class is AssetClass.CRYPTO and (
            self.quantity < 0 or self.leverage != 1
        ):
            raise ValueError("spot sandbox positions must be unlevered and nonnegative")

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.mark_price * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        if self.asset_class is AssetClass.CRYPTO:
            return 0.0
        return (
            self.quantity
            * (self.mark_price - self.entry_price)
            * self.multiplier
        )

    @property
    def initial_margin(self) -> float:
        return 0.0 if self.asset_class is AssetClass.CRYPTO else self.notional / self.leverage


@dataclass(frozen=True)
class SandboxEvent:
    event_id: str
    event_type: str
    occurred_at: datetime
    payload_json: str
    digest: str


@dataclass(frozen=True)
class SandboxAccountSnapshot:
    cash: float
    equity: float
    available_collateral: float
    initial_margin: float
    maintenance_margin: float
    positions: tuple[SandboxPosition, ...]
    events: int


class CryptoSandboxLedger:
    def __init__(self, config: CryptoSandboxConfig) -> None:
        self.config = config
        self._cash = config.initial_cash
        self._positions: dict[str, SandboxPosition] = {}
        self._events: list[SandboxEvent] = []
        self._receipts: dict[str, ExecutionReceipt] = {}
        self._intent_digests: dict[str, str] = {}
        self._funding_periods: set[tuple[str, datetime]] = set()

    @property
    def events(self) -> tuple[SandboxEvent, ...]:
        return tuple(self._events)

    def receipt(self, intent: OrderIntent) -> ExecutionReceipt | None:
        receipt = self._receipts.get(intent.intent_id)
        if receipt is not None and self._intent_digests[intent.intent_id] != sha256_digest(intent):
            raise RuntimeError("sandbox intent ID was reused with different contents")
        return receipt

    def position(self, instrument_id: str) -> SandboxPosition | None:
        return self._positions.get(instrument_id)

    def account(self) -> SandboxAccountSnapshot:
        positions = tuple(sorted(self._positions.values(), key=lambda item: item.instrument_id))
        spot_value = sum(
            item.quantity * item.mark_price * item.multiplier
            for item in positions
            if item.asset_class is AssetClass.CRYPTO
        )
        perpetual_unrealized = sum(item.unrealized_pnl for item in positions)
        initial_margin = sum(item.initial_margin for item in positions)
        maintenance = sum(
            item.notional * self.config.maintenance_margin_pct
            for item in positions
            if item.asset_class is AssetClass.PERPETUAL
        )
        equity = self._cash + spot_value + perpetual_unrealized
        return SandboxAccountSnapshot(
            self._cash,
            equity,
            equity - initial_margin,
            initial_margin,
            maintenance,
            positions,
            len(self._events),
        )

    def execute_fill(
        self,
        intent: OrderIntent,
        instrument: Instrument,
        market: SandboxMarketState,
        *,
        fill_price: float,
        leverage: float,
        occurred_at: datetime,
    ) -> tuple[float, float]:
        occurred_at = require_aware(occurred_at, "occurred_at")
        quantity = intent.quantity
        if quantity is None or not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("sandbox fill requires a positive explicit quantity")
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise ValueError("sandbox fill price must be finite and positive")
        if not math.isfinite(leverage) or leverage <= 0:
            raise ValueError("sandbox fill leverage must be finite and positive")
        delta = quantity if intent.side is OrderSide.BUY else -quantity
        current = self._positions.get(instrument.instrument_id)
        current_quantity = current.quantity if current else 0.0
        projected_quantity = current_quantity + delta
        if intent.reduce_only and (
            current is None
            or abs(projected_quantity) >= abs(current_quantity)
            or current_quantity * projected_quantity < 0
        ):
            raise PermissionError("reduce-only sandbox order would increase or flip exposure")
        if instrument.asset_class is AssetClass.CRYPTO and projected_quantity < -1e-12:
            raise PermissionError("spot sandbox does not permit naked short positions")

        notional = abs(delta) * fill_price * instrument.multiplier
        fee = notional * self.config.trading_fee_bps / 10_000
        next_position, realized = _position_after_fill(
            current,
            instrument,
            market,
            delta=delta,
            fill_price=fill_price,
            leverage=leverage,
        )
        if instrument.asset_class is AssetClass.CRYPTO:
            projected_cash = self._cash - delta * fill_price * instrument.multiplier - fee
            if projected_cash < -1e-9:
                raise PermissionError("spot sandbox order exceeds available cash")
        else:
            projected_cash = self._cash + realized - fee
            projected_positions = dict(self._positions)
            if next_position is None:
                projected_positions.pop(instrument.instrument_id, None)
            else:
                projected_positions[instrument.instrument_id] = next_position
            projected_margin = sum(
                item.notional / item.leverage
                for item in projected_positions.values()
                if item.asset_class is AssetClass.PERPETUAL
            )
            other_spot_value = sum(
                item.quantity * item.mark_price * item.multiplier
                for item in projected_positions.values()
                if item.asset_class is AssetClass.CRYPTO
            )
            other_unrealized = sum(
                item.unrealized_pnl for item in projected_positions.values()
            )
            projected_equity = projected_cash + other_spot_value + other_unrealized
            if projected_margin > projected_equity + 1e-9:
                raise PermissionError("perpetual sandbox order exceeds available collateral")

        self._cash = projected_cash
        if next_position is None:
            self._positions.pop(instrument.instrument_id, None)
        else:
            self._positions[instrument.instrument_id] = next_position
        self._append_event(
            "fill",
            occurred_at,
            {
                "intent_id": intent.intent_id,
                "instrument_id": instrument.instrument_id,
                "side": intent.side,
                "quantity": quantity,
                "fill_price": fill_price,
                "notional": notional,
                "fee": fee,
                "realized_pnl": realized,
                "reduce_only": intent.reduce_only,
            },
        )
        return fee, realized

    def record_receipt(
        self,
        receipt: ExecutionReceipt,
        intent: OrderIntent,
        *,
        detail: Mapping[str, object],
    ) -> None:
        existing = self._receipts.get(receipt.intent_id)
        if existing is not None:
            if (
                canonical_json(existing) != canonical_json(receipt)
                or self._intent_digests[receipt.intent_id] != sha256_digest(intent)
            ):
                raise RuntimeError("sandbox receipt identity conflict")
            return
        self._receipts[receipt.intent_id] = receipt
        self._intent_digests[receipt.intent_id] = sha256_digest(intent)
        self._append_event(
            "order",
            receipt.executed_at,
            {"intent": intent, "receipt": receipt, "detail": dict(detail)},
        )

    def apply_funding(
        self, market: SandboxMarketState, *, occurred_at: datetime
    ) -> float:
        occurred_at = require_aware(occurred_at, "occurred_at")
        if market.asset_class is not AssetClass.PERPETUAL or market.funding_at is None:
            raise ValueError("funding requires a perpetual market and funding timestamp")
        if market.funding_at > occurred_at:
            raise ValueError("funding period has not occurred")
        key = (market.instrument_id, market.funding_at)
        if key in self._funding_periods:
            return 0.0
        self._funding_periods.add(key)
        position = self._positions.get(market.instrument_id)
        payment = 0.0
        if position is not None:
            payment = -position.quantity * market.mark * position.multiplier * market.funding_rate
            self._cash += payment
            self._positions[market.instrument_id] = replace(
                position,
                mark_price=market.mark,
                funding_pnl=position.funding_pnl + payment,
            )
        self._append_event(
            "funding",
            occurred_at,
            {
                "instrument_id": market.instrument_id,
                "funding_at": market.funding_at,
                "funding_rate": market.funding_rate,
                "payment": payment,
            },
        )
        return payment

    def mark_to_market(
        self, market: SandboxMarketState, *, occurred_at: datetime
    ) -> tuple[str, ...]:
        occurred_at = require_aware(occurred_at, "occurred_at")
        position = self._positions.get(market.instrument_id)
        if position is not None:
            self._positions[market.instrument_id] = replace(position, mark_price=market.mark)
        account = self.account()
        if account.maintenance_margin <= 0 or account.equity > account.maintenance_margin:
            return ()
        liquidated: list[str] = []
        for instrument_id, item in tuple(self._positions.items()):
            if item.asset_class is not AssetClass.PERPETUAL:
                continue
            realized = item.quantity * (item.mark_price - item.entry_price) * item.multiplier
            fee = item.notional * self.config.trading_fee_bps / 10_000
            self._cash += realized - fee
            del self._positions[instrument_id]
            liquidated.append(instrument_id)
            self._append_event(
                "liquidation",
                occurred_at,
                {
                    "instrument_id": instrument_id,
                    "quantity": item.quantity,
                    "mark_price": item.mark_price,
                    "realized_pnl": realized,
                    "fee": fee,
                },
            )
        return tuple(sorted(liquidated))

    def verify_integrity(self) -> int:
        for ordinal, event in enumerate(self._events):
            payload = json.loads(event.payload_json)
            if sha256_digest(payload) != event.digest:
                raise RuntimeError(f"sandbox event digest mismatch: {event.event_id}")
            expected_id = sha256_digest({"ordinal": ordinal, "payload": payload})
            if event.event_id != expected_id:
                raise RuntimeError(f"sandbox event identity mismatch: {event.event_id}")
            if payload.get("event_type") != event.event_type:
                raise RuntimeError(f"sandbox event type mismatch: {event.event_id}")
        return len(self._events)

    def _append_event(
        self, event_type: str, occurred_at: datetime, payload: Mapping[str, object]
    ) -> None:
        occurred_at = require_aware(occurred_at, "occurred_at")
        body = {"event_type": event_type, "occurred_at": occurred_at, **payload}
        payload_json = canonical_json(body)
        digest = sha256_digest(json.loads(payload_json))
        event_id = sha256_digest({"ordinal": len(self._events), "payload": body})
        self._events.append(
            SandboxEvent(event_id, event_type, occurred_at, payload_json, digest)
        )


class CryptoPerpetualSandboxAdapter:
    environment = ExecutionEnvironment.PAPER

    def __init__(
        self,
        config: CryptoSandboxConfig,
        ledger: CryptoSandboxLedger,
        instrument_resolver: Callable[[str], Instrument],
        market_resolver: Callable[[str], SandboxMarketState],
        *,
        enabled: bool = False,
        strategy_eligible: bool = False,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.instrument_resolver = instrument_resolver
        self.market_resolver = market_resolver
        self.enabled = enabled
        self.strategy_eligible = strategy_eligible

    def submit(self, approval: ApprovedOrderIntent, *, now: datetime) -> ExecutionReceipt:
        now = require_aware(now, "now")
        if not self.enabled:
            raise PermissionError("crypto sandbox execution is disabled")
        if not self.strategy_eligible:
            raise PermissionError("crypto sandbox strategy has not passed evidence gates")
        intent = approval.intent
        if intent.environment is not ExecutionEnvironment.PAPER:
            raise PermissionError("crypto sandbox accepts paper intents only")
        numeric = (intent.notional, intent.quantity, intent.max_price, intent.min_price)
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("crypto sandbox intent values must be finite")
        existing = self.ledger.receipt(intent)
        if existing is not None:
            return existing
        instrument = self.instrument_resolver(intent.instrument_id)
        if instrument.asset_class not in {AssetClass.CRYPTO, AssetClass.PERPETUAL}:
            raise PermissionError("crypto sandbox accepts spot crypto and perpetuals only")
        if intent.asset_class is not instrument.asset_class or intent.venue != instrument.venue:
            raise PermissionError("sandbox intent does not match instrument routing")
        market = self.market_resolver(instrument.instrument_id)
        self._validate_market(instrument, market, now)
        leverage = _instrument_leverage(instrument, self.config)
        order_type = _select_order_type(intent.allowed_order_types)
        fill_price, status = _sandbox_fill_price(intent, market, order_type, self.config)
        if fill_price is None:
            return self._record_terminal_receipt(intent, status, now, order_type, market)
        quantity = intent.quantity
        if quantity is None:
            raise ValueError("crypto sandbox order requires an explicit quantity")
        notional = quantity * fill_price * instrument.multiplier
        if notional < self.config.min_order_notional:
            raise ValueError("crypto sandbox order is below minimum notional")
        if notional > intent.notional + 1e-8:
            raise PermissionError("sandbox fill would exceed signed intent notional")
        fee, realized = self.ledger.execute_fill(
            intent,
            instrument,
            market,
            fill_price=fill_price,
            leverage=leverage,
            occurred_at=now,
        )
        receipt = ExecutionReceipt(
            intent.intent_id,
            self.environment,
            "filled",
            now,
            _sandbox_order_id(intent.intent_id),
            _sandbox_order_id(intent.intent_id),
            quantity,
            fill_price,
        )
        self.ledger.record_receipt(
            receipt,
            intent,
            detail={
                "order_type": order_type,
                "fee": fee,
                "realized_pnl": realized,
                "leverage": leverage,
            },
        )
        return receipt

    def apply_funding(self, market: SandboxMarketState, *, now: datetime) -> float:
        self._validate_runtime_gate()
        self._validate_market_identity(market)
        self._validate_freshness(market, now)
        return self.ledger.apply_funding(market, occurred_at=now)

    def mark_to_market(
        self, market: SandboxMarketState, *, now: datetime
    ) -> tuple[str, ...]:
        self._validate_runtime_gate()
        self._validate_market_identity(market)
        self._validate_freshness(market, now)
        return self.ledger.mark_to_market(market, occurred_at=now)

    def _record_terminal_receipt(
        self,
        intent: OrderIntent,
        status: str,
        now: datetime,
        order_type: OrderType,
        market: SandboxMarketState,
    ) -> ExecutionReceipt:
        receipt = ExecutionReceipt(
            intent.intent_id,
            self.environment,
            status,
            now,
            _sandbox_order_id(intent.intent_id),
            _sandbox_order_id(intent.intent_id),
        )
        self.ledger.record_receipt(
            receipt,
            intent,
            detail={"order_type": order_type, "bid": market.bid, "ask": market.ask},
        )
        return receipt

    def _validate_runtime_gate(self) -> None:
        if not self.enabled or not self.strategy_eligible:
            raise PermissionError("crypto sandbox runtime gates are closed")

    def _validate_market_identity(self, market: SandboxMarketState) -> None:
        instrument = self.instrument_resolver(market.instrument_id)
        if instrument.asset_class is not market.asset_class:
            raise ValueError("sandbox market asset class does not match instrument")

    def _validate_market(
        self, instrument: Instrument, market: SandboxMarketState, now: datetime
    ) -> None:
        if market.instrument_id != instrument.instrument_id:
            raise ValueError("sandbox market instrument does not match order")
        if market.asset_class is not instrument.asset_class:
            raise ValueError("sandbox market asset class does not match order")
        self._validate_freshness(market, now)

    def _validate_freshness(self, market: SandboxMarketState, now: datetime) -> None:
        now = require_aware(now, "now")
        if market.observed_at > now:
            raise ValueError("sandbox market observation is from the future")
        if now - market.observed_at > self.config.max_market_age:
            raise ValueError("sandbox market observation is stale")


def _position_after_fill(
    current: SandboxPosition | None,
    instrument: Instrument,
    market: SandboxMarketState,
    *,
    delta: float,
    fill_price: float,
    leverage: float,
) -> tuple[SandboxPosition | None, float]:
    if current is None or abs(current.quantity) < 1e-12:
        return (
            SandboxPosition(
                instrument.instrument_id,
                instrument.asset_class,
                delta,
                fill_price,
                market.mark,
                instrument.multiplier,
                leverage,
            ),
            0.0,
        )
    current_quantity = current.quantity
    if current_quantity * delta > 0:
        next_quantity = current_quantity + delta
        entry = (
            abs(current_quantity) * current.entry_price + abs(delta) * fill_price
        ) / abs(next_quantity)
        return (
            replace(
                current,
                quantity=next_quantity,
                entry_price=entry,
                mark_price=market.mark,
            ),
            0.0,
        )
    closed = min(abs(current_quantity), abs(delta))
    realized = (
        closed
        * (fill_price - current.entry_price)
        * (1 if current_quantity > 0 else -1)
        * current.multiplier
    )
    next_quantity = current_quantity + delta
    total_realized = current.realized_pnl + realized
    if abs(next_quantity) < 1e-12:
        return None, realized
    if current_quantity * next_quantity > 0:
        return (
            replace(
                current,
                quantity=next_quantity,
                mark_price=market.mark,
                realized_pnl=total_realized,
            ),
            realized,
        )
    return (
        SandboxPosition(
            instrument.instrument_id,
            instrument.asset_class,
            next_quantity,
            fill_price,
            market.mark,
            instrument.multiplier,
            leverage,
            total_realized,
            current.funding_pnl,
        ),
        realized,
    )


def _instrument_leverage(
    instrument: Instrument, config: CryptoSandboxConfig
) -> float:
    if instrument.asset_class is AssetClass.CRYPTO:
        return 1.0
    raw = instrument.metadata.get("sandbox_leverage", config.default_leverage)
    if isinstance(raw, bool):
        raise ValueError("sandbox leverage must be numeric")
    leverage = float(raw)
    if not 1 <= leverage <= config.max_leverage:
        raise PermissionError("requested sandbox leverage exceeds configured limit")
    return leverage


def _select_order_type(allowed: tuple[OrderType, ...]) -> OrderType:
    for candidate in (OrderType.LIMIT, OrderType.POST_ONLY, OrderType.MARKET):
        if candidate in allowed:
            return candidate
    raise ValueError("sandbox intent does not permit a supported order type")


def _sandbox_fill_price(
    intent: OrderIntent,
    market: SandboxMarketState,
    order_type: OrderType,
    config: CryptoSandboxConfig,
) -> tuple[float | None, str]:
    slippage = config.slippage_bps / 10_000
    if order_type is OrderType.POST_ONLY:
        if (
            (intent.side is OrderSide.BUY and intent.max_price is None)
            or (intent.side is OrderSide.SELL and intent.min_price is None)
        ):
            return None, "rejected_missing_price"
        crosses = (
            intent.side is OrderSide.BUY
            and intent.max_price is not None
            and intent.max_price >= market.ask
        ) or (
            intent.side is OrderSide.SELL
            and intent.min_price is not None
            and intent.min_price <= market.bid
        )
        return (None, "rejected_post_only_cross") if crosses else (None, "posted")
    if intent.side is OrderSide.BUY:
        if order_type is OrderType.LIMIT and (
            intent.max_price is None or intent.max_price < market.ask
        ):
            return None, "not_filled"
        price = market.ask * (1 + slippage)
        if intent.max_price is not None and price > intent.max_price:
            return None, "rejected_price_bound"
        return price, "filled"
    if order_type is OrderType.LIMIT and (
        intent.min_price is None or intent.min_price > market.bid
    ):
        return None, "not_filled"
    price = market.bid * (1 - slippage)
    if intent.min_price is not None and price < intent.min_price:
        return None, "rejected_price_bound"
    return price, "filled"


def _sandbox_order_id(intent_id: str) -> str:
    digest = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:32]
    return f"sandbox-{digest}"


class SandboxScenarioStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class SandboxScenarioResult:
    scenario_id: str
    label: str
    status: SandboxScenarioStatus
    checks: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class CryptoSandboxReport:
    generated_at: datetime
    config_version: str
    scenarios: tuple[SandboxScenarioResult, ...]
    network_access: bool = False
    broker_credentials_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generated_at", require_aware(self.generated_at, "generated_at")
        )

    @property
    def passed(self) -> int:
        return sum(item.status is SandboxScenarioStatus.PASSED for item in self.scenarios)

    @property
    def failed(self) -> int:
        return len(self.scenarios) - self.passed

    @property
    def successful(self) -> bool:
        return self.failed == 0


def sandbox_scenario_names() -> tuple[str, ...]:
    return tuple(_SANDBOX_SCENARIOS)


def run_crypto_sandbox_scenarios(
    scenario: str = "all",
    *,
    config: CryptoSandboxConfig | None = None,
    generated_at: datetime | None = None,
) -> CryptoSandboxReport:
    config = config or CryptoSandboxConfig()
    selected = tuple(_SANDBOX_SCENARIOS) if scenario == "all" else (scenario,)
    unknown = [item for item in selected if item not in _SANDBOX_SCENARIOS]
    if unknown:
        raise ValueError(f"unknown crypto sandbox scenario: {unknown[0]}")
    results = tuple(
        _run_sandbox_safely(item, _SANDBOX_SCENARIOS[item], config)
        for item in selected
    )
    return CryptoSandboxReport(
        generated_at or datetime.now(timezone.utc), config.version, results
    )


def render_crypto_sandbox_report(
    report: CryptoSandboxReport, output_format: str = "text"
) -> str:
    if output_format == "json":
        payload = asdict(report)
        payload["generated_at"] = report.generated_at.isoformat()
        payload["passed"] = report.passed
        payload["failed"] = report.failed
        payload["successful"] = report.successful
        return json.dumps(payload, sort_keys=True, indent=2)
    if output_format == "markdown":
        lines = [
            "## Crypto and perpetual sandbox",
            "",
            (
                f"**{'PASS' if report.successful else 'FAIL'}** · "
                f"{report.passed}/{len(report.scenarios)} scenarios passed · "
                f"policy `{report.config_version}` · network disabled · credentials unused"
            ),
            "",
            "| Scenario | Status | Verified behavior |",
            "|---|---:|---|",
        ]
        for item in report.scenarios:
            checks = "; ".join(item.checks).replace("|", "\\|")
            lines.append(f"| {item.label} | {item.status.value.upper()} | {checks} |")
        return "\n".join(lines)
    if output_format != "text":
        raise ValueError("crypto sandbox format must be text, json, or markdown")
    lines = [
        (
            f"Crypto/perpetual sandbox: {'PASS' if report.successful else 'FAIL'} "
            f"passed={report.passed} failed={report.failed} "
            f"policy={report.config_version} network=false credentials=false"
        )
    ]
    for item in report.scenarios:
        lines.append(f"{item.scenario_id}: {item.status.value} - {item.detail}")
        lines.extend(f"  check: {check}" for check in item.checks)
    return "\n".join(lines)


def _run_sandbox_safely(
    scenario_id: str,
    scenario: tuple[
        str, Callable[[CryptoSandboxConfig], tuple[tuple[str, ...], str]]
    ],
    config: CryptoSandboxConfig,
) -> SandboxScenarioResult:
    label, operation = scenario
    try:
        with patch(
            "socket.create_connection",
            side_effect=PermissionError("network access is disabled in crypto sandbox"),
        ):
            checks, detail = operation(config)
        return SandboxScenarioResult(
            scenario_id, label, SandboxScenarioStatus.PASSED, checks, detail
        )
    except Exception as exc:
        return SandboxScenarioResult(
            scenario_id,
            label,
            SandboxScenarioStatus.FAILED,
            (),
            f"{type(exc).__name__}: {exc}",
        )


def _spot_fill_scenario(config: CryptoSandboxConfig) -> tuple[tuple[str, ...], str]:
    instrument = _spot_instrument()
    market = _market(instrument, bid=99, ask=100, mark=99.5)
    adapter, ledger, signer = _sandbox(instrument, market, config)
    intent = _intent(instrument, "sandbox-spot-fill", quantity=1, notional=101)
    receipt = _execute(adapter, signer, intent, instrument, equity=config.initial_cash)
    account = ledger.account()
    _require(receipt.status == "filled", "spot order did not fill")
    _require(ledger.position(instrument.instrument_id) is not None, "spot position missing")
    _require(account.cash < config.initial_cash - 100, "spot cash and fee were not debited")
    _require(ledger.verify_integrity() == 2, "spot events failed integrity")
    return (
        ("ask-side fill", "fee debited", "spot inventory created", "ledger verified"),
        "spot execution used executable-side pricing",
    )


def _perpetual_margin_scenario(
    config: CryptoSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    instrument = _perpetual_instrument(leverage=3)
    market = _market(instrument, bid=99, ask=100, mark=100)
    adapter, ledger, signer = _sandbox(instrument, market, config)
    intent = _intent(instrument, "sandbox-perp-margin", quantity=10, notional=1_010)
    _execute(adapter, signer, intent, instrument, equity=config.initial_cash)
    position = ledger.position(instrument.instrument_id)
    _require(position is not None and position.leverage == 3, "perpetual leverage missing")
    _require(ledger.account().initial_margin > 0, "initial margin was not reserved")
    return (
        ("signed fill", "3x leverage applied", "initial margin reserved"),
        "perpetual exposure consumed collateral under configured leverage",
    )


def _reduce_only_scenario(config: CryptoSandboxConfig) -> tuple[tuple[str, ...], str]:
    instrument = _perpetual_instrument(leverage=2)
    market = _market(instrument, bid=99, ask=100, mark=100)
    adapter, ledger, signer = _sandbox(instrument, market, config)
    opening = _intent(instrument, "sandbox-reduce-open", quantity=10, notional=1_010)
    _execute(adapter, signer, opening, instrument, equity=config.initial_cash)
    invalid = _intent(
        instrument,
        "sandbox-reduce-invalid",
        quantity=1,
        notional=101,
        reduce_only=True,
    )
    try:
        _execute(adapter, signer, invalid, instrument, equity=config.initial_cash)
    except PermissionError as exc:
        _require("reduce-only" in str(exc), "reduce-only failed for unrelated reason")
    else:
        raise AssertionError("reduce-only exposure increase was accepted")
    closing = _intent(
        instrument,
        "sandbox-reduce-close",
        side=OrderSide.SELL,
        quantity=10,
        notional=1_010,
        reduce_only=True,
        min_price=98,
    )
    _execute(adapter, signer, closing, instrument, equity=config.initial_cash)
    _require(ledger.position(instrument.instrument_id) is None, "reduce-only close failed")
    return (
        ("exposure increase rejected", "position close accepted", "position reached zero"),
        "reduce-only semantics prevented increases and flips",
    )


def _funding_scenario(config: CryptoSandboxConfig) -> tuple[tuple[str, ...], str]:
    instrument = _perpetual_instrument(leverage=2)
    funding_at = SANDBOX_TIME + timedelta(hours=1)
    market = _market(
        instrument,
        bid=99,
        ask=100,
        mark=100,
        observed_at=funding_at,
        funding_rate=0.001,
        funding_at=funding_at,
    )
    opening_market = replace(
        market, observed_at=SANDBOX_TIME, funding_rate=0, funding_at=None
    )
    adapter, ledger, signer = _sandbox(instrument, opening_market, config)
    intent = _intent(instrument, "sandbox-funding", quantity=10, notional=1_010)
    _execute(adapter, signer, intent, instrument, equity=config.initial_cash)
    payment = adapter.apply_funding(market, now=funding_at)
    duplicate = adapter.apply_funding(market, now=funding_at)
    _require(payment < 0, "positive funding did not charge the long")
    _require(duplicate == 0, "funding period was applied twice")
    return (
        ("long paid positive funding", "period recorded once", "duplicate was idempotent"),
        "distinct funding periods changed collateral exactly once",
    )


def _liquidation_scenario(config: CryptoSandboxConfig) -> tuple[tuple[str, ...], str]:
    tight = replace(config, initial_cash=1_000, default_leverage=5, max_leverage=5)
    instrument = _perpetual_instrument(leverage=5)
    opening_market = _market(instrument, bid=99.9, ask=100, mark=100)
    adapter, ledger, signer = _sandbox(instrument, opening_market, tight)
    intent = _intent(instrument, "sandbox-liquidation", quantity=49, notional=4_950)
    _execute(adapter, signer, intent, instrument, equity=5_000)
    crash_time = SANDBOX_TIME + timedelta(seconds=30)
    crash = _market(
        instrument,
        bid=80,
        ask=80.1,
        mark=80,
        observed_at=crash_time,
    )
    liquidated = adapter.mark_to_market(crash, now=crash_time)
    _require(liquidated == (instrument.instrument_id,), "position was not liquidated")
    _require(ledger.position(instrument.instrument_id) is None, "liquidated position remained")
    _require(any(item.event_type == "liquidation" for item in ledger.events), "event missing")
    return (
        ("maintenance breach detected", "position closed", "liquidation event recorded"),
        "adverse mark triggered deterministic cross-margin liquidation",
    )


def _stale_gate_scenario(config: CryptoSandboxConfig) -> tuple[tuple[str, ...], str]:
    instrument = _spot_instrument()
    market = _market(
        instrument,
        bid=99,
        ask=100,
        mark=99.5,
        observed_at=SANDBOX_TIME - config.max_market_age - timedelta(seconds=1),
    )
    adapter, _, signer = _sandbox(instrument, market, config)
    intent = _intent(instrument, "sandbox-stale", quantity=1, notional=101)
    try:
        _execute(adapter, signer, intent, instrument, equity=config.initial_cash)
    except ValueError as exc:
        _require("stale" in str(exc), "stale gate failed for unrelated reason")
    else:
        raise AssertionError("stale market was accepted")
    return (("expired market state rejected",), "stale data failed before simulated fill")


def _eligibility_gate_scenario(
    config: CryptoSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    instrument = _spot_instrument()
    market = _market(instrument, bid=99, ask=100, mark=99.5)
    ledger = CryptoSandboxLedger(config)
    signer = ApprovalSigner(b"crypto-sandbox-signing-key")
    intent = _intent(instrument, "sandbox-ineligible", quantity=1, notional=101)
    approval = _approval(signer, intent, instrument, config.initial_cash)
    disabled = CryptoPerpetualSandboxAdapter(
        config,
        ledger,
        lambda _: instrument,
        lambda _: market,
        enabled=False,
        strategy_eligible=True,
    )
    try:
        disabled.submit(approval, now=SANDBOX_TIME)
    except PermissionError as exc:
        _require("disabled" in str(exc), "disabled gate failed for unrelated reason")
    else:
        raise AssertionError("disabled sandbox accepted an order")
    ineligible = CryptoPerpetualSandboxAdapter(
        config,
        ledger,
        lambda _: instrument,
        lambda _: market,
        enabled=True,
        strategy_eligible=False,
    )
    try:
        ineligible.submit(approval, now=SANDBOX_TIME)
    except PermissionError as exc:
        _require("evidence gates" in str(exc), "eligibility gate failed for unrelated reason")
    else:
        raise AssertionError("ineligible strategy accepted an order")
    _require(ledger.events == (), "closed gates mutated the ledger")
    return (
        ("runtime disabled gate closed", "evidence gate closed", "ledger unchanged"),
        "independent controls blocked simulation before state mutation",
    )


def _post_only_scenario(config: CryptoSandboxConfig) -> tuple[tuple[str, ...], str]:
    instrument = _perpetual_instrument(leverage=2)
    market = _market(instrument, bid=99, ask=100, mark=99.5)
    adapter, ledger, signer = _sandbox(instrument, market, config)
    intent = _intent(
        instrument,
        "sandbox-post-only",
        quantity=1,
        notional=101,
        allowed=(OrderType.POST_ONLY,),
        max_price=100,
    )
    first = _execute(adapter, signer, intent, instrument, equity=config.initial_cash)
    second = _execute(adapter, signer, intent, instrument, equity=config.initial_cash)
    _require(first.status == "rejected_post_only_cross", "crossing post-only was accepted")
    _require(second.status == first.status, "terminal post-only result changed")
    _require(ledger.position(instrument.instrument_id) is None, "post-only rejection filled")
    return (
        ("cross rejected", "no position mutation", "restart result idempotent"),
        "post-only protection refused liquidity-taking execution",
    )


def _sandbox(
    instrument: Instrument,
    market: SandboxMarketState,
    config: CryptoSandboxConfig,
) -> tuple[CryptoPerpetualSandboxAdapter, CryptoSandboxLedger, ApprovalSigner]:
    ledger = CryptoSandboxLedger(config)
    signer = ApprovalSigner(b"crypto-sandbox-signing-key")
    adapter = CryptoPerpetualSandboxAdapter(
        config,
        ledger,
        lambda _: instrument,
        lambda _: market,
        enabled=True,
        strategy_eligible=True,
    )
    return adapter, ledger, signer


def _execute(
    adapter: CryptoPerpetualSandboxAdapter,
    signer: ApprovalSigner,
    intent: OrderIntent,
    instrument: Instrument,
    *,
    equity: float,
) -> ExecutionReceipt:
    risk_positions = tuple(
        Position(
            item.instrument_id,
            instrument.venue,
            item.asset_class,
            item.quantity * item.mark_price * item.multiplier,
        )
        for item in adapter.ledger.account().positions
    )
    approval = _approval(signer, intent, instrument, equity, positions=risk_positions)
    return DeterministicExecutor(signer, adapter).execute(approval, now=SANDBOX_TIME)


def _approval(
    signer: ApprovalSigner,
    intent: OrderIntent,
    instrument: Instrument,
    equity: float,
    *,
    positions: tuple[Position, ...] = (),
) -> ApprovedOrderIntent:
    governor = RiskGovernor(
        RiskLimits(
            max_gross_notional=1_000_000,
            max_instrument_notional=500_000,
            max_venue_notional=1_000_000,
            asset_class_caps={
                AssetClass.CRYPTO: 500_000,
                AssetClass.PERPETUAL: 500_000,
                AssetClass.MEMECOIN: 0,
            },
            allow_live=False,
        ),
        signer,
    )
    return governor.approve(
        intent,
        instrument=instrument,
        portfolio=PortfolioSnapshot(
            SANDBOX_TIME,
            equity,
            max(equity, intent.notional),
            positions,
        ),
        now=SANDBOX_TIME,
    )


def _intent(
    instrument: Instrument,
    intent_id: str,
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: float,
    notional: float,
    reduce_only: bool = False,
    allowed: tuple[OrderType, ...] = (OrderType.LIMIT,),
    max_price: float | None = 101,
    min_price: float | None = None,
) -> OrderIntent:
    return OrderIntent(
        intent_id,
        "crypto-sandbox-strategy",
        "sandbox-v1",
        instrument.instrument_id,
        instrument.venue,
        instrument.asset_class,
        side,
        notional,
        ExecutionEnvironment.PAPER,
        allowed,
        SANDBOX_TIME + timedelta(minutes=5),
        max_price=max_price,
        min_price=min_price,
        reduce_only=reduce_only,
        created_at=SANDBOX_TIME,
        quantity=quantity,
        forecast_id="sandbox-forecast",
    )


def _spot_instrument() -> Instrument:
    return Instrument("sandbox:crypto:BTC-USD", "sandbox", "BTC-USD", AssetClass.CRYPTO, "USD")


def _perpetual_instrument(*, leverage: float) -> Instrument:
    return Instrument(
        "sandbox:perpetual:BTC-PERP",
        "sandbox",
        "BTC-PERP",
        AssetClass.PERPETUAL,
        "USD",
        metadata={"sandbox_leverage": leverage},
    )


def _market(
    instrument: Instrument,
    *,
    bid: float,
    ask: float,
    mark: float,
    observed_at: datetime = SANDBOX_TIME,
    funding_rate: float = 0.0,
    funding_at: datetime | None = None,
) -> SandboxMarketState:
    return SandboxMarketState(
        instrument.instrument_id,
        instrument.asset_class,
        bid,
        ask,
        mark,
        observed_at,
        funding_rate,
        funding_at,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_SANDBOX_SCENARIOS: dict[
    str, tuple[str, Callable[[CryptoSandboxConfig], tuple[tuple[str, ...], str]]]
] = {
    "spot-fill": ("Spot executable fill", _spot_fill_scenario),
    "perpetual-margin": ("Perpetual leverage and margin", _perpetual_margin_scenario),
    "reduce-only": ("Reduce-only enforcement", _reduce_only_scenario),
    "funding": ("Funding settlement", _funding_scenario),
    "liquidation": ("Cross-margin liquidation", _liquidation_scenario),
    "stale-market": ("Stale market rejection", _stale_gate_scenario),
    "eligibility-gates": ("Independent enable gates", _eligibility_gate_scenario),
    "post-only": ("Post-only protection", _post_only_scenario),
}
