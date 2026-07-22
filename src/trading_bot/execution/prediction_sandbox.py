from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping
from unittest.mock import patch
from urllib.parse import urlsplit

from trading_bot.core.serialization import canonical_json, require_aware, sha256_digest


SANDBOX_TIME = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _decimal(value: Decimal | float | int | str, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _money(value: Decimal) -> float:
    return float(value)


@dataclass(frozen=True)
class PredictionSandboxConfig:
    version: str = "prediction-settlement-sandbox-v1"
    initial_cash: Decimal = Decimal("100000")
    taker_fee_coefficient: Decimal = Decimal("0.07")
    maker_fee_coefficient: Decimal = Decimal("0.0175")
    fee_increment: Decimal = Decimal("0.0001")
    settlement_increment: Decimal = Decimal("0.01")
    max_market_age: timedelta = timedelta(seconds=60)
    max_order_contracts: int = 1_000
    max_market_cost: Decimal = Decimal("5000")
    max_total_cost: Decimal = Decimal("10000")
    fee_schedule_effective_date: date = date(2026, 7, 7)
    fee_source_url: str = "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
    settlement_source_url: str = (
        "https://docs.kalshi.com/getting_started/market_settlement"
    )

    def __post_init__(self) -> None:
        for field_name in (
            "initial_cash",
            "taker_fee_coefficient",
            "maker_fee_coefficient",
            "fee_increment",
            "settlement_increment",
            "max_market_cost",
            "max_total_cost",
        ):
            object.__setattr__(self, field_name, _decimal(getattr(self, field_name), field_name))
        if not self.version or self.initial_cash <= 0:
            raise ValueError("prediction sandbox version and positive cash are required")
        if not 0 < self.taker_fee_coefficient <= 1:
            raise ValueError("taker fee coefficient must be between zero and one")
        if not 0 <= self.maker_fee_coefficient <= self.taker_fee_coefficient:
            raise ValueError("maker fee coefficient must not exceed taker coefficient")
        if not 0 < self.fee_increment <= Decimal("0.01"):
            raise ValueError("fee increment must be positive and no greater than one cent")
        if self.settlement_increment != Decimal("0.01"):
            raise ValueError("settlement payout increment must be one cent")
        if self.max_market_age <= timedelta(0):
            raise ValueError("maximum market age must be positive")
        if self.max_order_contracts < 1:
            raise ValueError("maximum order contracts must be positive")
        if not 0 < self.max_market_cost <= self.max_total_cost <= self.initial_cash:
            raise ValueError("prediction sandbox cost limits are invalid")
        for source in (self.fee_source_url, self.settlement_source_url):
            parsed = urlsplit(source)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("prediction rule sources must be absolute HTTPS URLs")


def load_prediction_sandbox_config(path: str | Path) -> PredictionSandboxConfig:
    config_path = Path(path)
    if config_path.stat().st_size > 1_000_000:
        raise ValueError("prediction sandbox config exceeds the 1 MB safety limit")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "version",
        "initial_cash",
        "taker_fee_coefficient",
        "maker_fee_coefficient",
        "fee_increment",
        "settlement_increment",
        "max_market_age_seconds",
        "max_order_contracts",
        "max_market_cost",
        "max_total_cost",
        "fee_schedule_effective_date",
        "fee_source_url",
        "settlement_source_url",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            "prediction sandbox config keys mismatch: "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )
    try:
        effective = date.fromisoformat(str(raw["fee_schedule_effective_date"]))
        max_age = timedelta(seconds=float(raw["max_market_age_seconds"]))
        max_contracts = int(raw["max_order_contracts"])
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction sandbox date, market age, or limits are invalid") from exc
    if type(raw["max_order_contracts"]) is not int:
        raise ValueError("maximum order contracts must be a whole number")
    return PredictionSandboxConfig(
        version=str(raw["version"]),
        initial_cash=_decimal(raw["initial_cash"], "initial_cash"),
        taker_fee_coefficient=_decimal(
            raw["taker_fee_coefficient"], "taker_fee_coefficient"
        ),
        maker_fee_coefficient=_decimal(
            raw["maker_fee_coefficient"], "maker_fee_coefficient"
        ),
        fee_increment=_decimal(raw["fee_increment"], "fee_increment"),
        settlement_increment=_decimal(
            raw["settlement_increment"], "settlement_increment"
        ),
        max_market_age=max_age,
        max_order_contracts=max_contracts,
        max_market_cost=_decimal(raw["max_market_cost"], "max_market_cost"),
        max_total_cost=_decimal(raw["max_total_cost"], "max_total_cost"),
        fee_schedule_effective_date=effective,
        fee_source_url=str(raw["fee_source_url"]),
        settlement_source_url=str(raw["settlement_source_url"]),
    )


class PredictionOutcome(StrEnum):
    YES = "yes"
    NO = "no"


class PredictionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"


class PredictionLiquidity(StrEnum):
    TAKER = "taker"
    MAKER = "maker"


class PredictionMarketStatus(StrEnum):
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    CLOSED = "closed"
    DETERMINED = "determined"
    DISPUTED = "disputed"
    AMENDED = "amended"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class PredictionMarketState:
    market_id: str
    yes_bid: Decimal
    yes_ask: Decimal
    observed_at: datetime
    status: PredictionMarketStatus = PredictionMarketStatus.ACTIVE
    taker_fee_multiplier: Decimal = Decimal("1")
    maker_fee_multiplier: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", require_aware(self.observed_at, "observed_at"))
        object.__setattr__(self, "yes_bid", _decimal(self.yes_bid, "yes_bid"))
        object.__setattr__(self, "yes_ask", _decimal(self.yes_ask, "yes_ask"))
        for field_name in ("taker_fee_multiplier", "maker_fee_multiplier"):
            object.__setattr__(
                self, field_name, _decimal(getattr(self, field_name), field_name)
            )
        if not self.market_id:
            raise ValueError("prediction market identity is required")
        if not isinstance(self.status, PredictionMarketStatus):
            raise ValueError("prediction market status is invalid")
        if not _ZERO <= self.yes_bid <= self.yes_ask <= _ONE:
            raise ValueError("prediction prices must satisfy 0 <= bid <= ask <= 1")
        if any(
            value not in {_ZERO, _ONE}
            for value in (self.taker_fee_multiplier, self.maker_fee_multiplier)
        ):
            raise ValueError("sandbox fee multipliers must be zero or one")

    def executable_price(
        self, outcome: PredictionOutcome, action: PredictionAction
    ) -> Decimal:
        prices = {
            (PredictionOutcome.YES, PredictionAction.BUY): self.yes_ask,
            (PredictionOutcome.YES, PredictionAction.SELL): self.yes_bid,
            (PredictionOutcome.NO, PredictionAction.BUY): _ONE - self.yes_bid,
            (PredictionOutcome.NO, PredictionAction.SELL): _ONE - self.yes_ask,
        }
        return prices[(outcome, action)]


@dataclass(frozen=True)
class PredictionOrder:
    order_id: str
    strategy_id: str
    market_id: str
    outcome: PredictionOutcome
    action: PredictionAction
    quantity: int
    max_cost: Decimal
    created_at: datetime
    expires_at: datetime
    liquidity: PredictionLiquidity = PredictionLiquidity.TAKER
    limit_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", require_aware(self.created_at, "created_at"))
        object.__setattr__(self, "expires_at", require_aware(self.expires_at, "expires_at"))
        object.__setattr__(self, "max_cost", _decimal(self.max_cost, "max_cost"))
        if self.limit_price is not None:
            object.__setattr__(
                self, "limit_price", _decimal(self.limit_price, "limit_price")
            )
        if not self.order_id or not self.strategy_id or not self.market_id:
            raise ValueError("prediction order identity is required")
        if not isinstance(self.outcome, PredictionOutcome) or not isinstance(
            self.action, PredictionAction
        ):
            raise ValueError("prediction order outcome or action is invalid")
        if not isinstance(self.liquidity, PredictionLiquidity):
            raise ValueError("prediction order liquidity is invalid")
        if (
            not isinstance(self.quantity, int)
            or isinstance(self.quantity, bool)
            or self.quantity < 1
        ):
            raise ValueError("prediction order quantity must be a positive whole number")
        if self.max_cost <= 0:
            raise ValueError("prediction order maximum cost must be positive")
        if self.expires_at <= self.created_at:
            raise ValueError("prediction order must expire after creation")
        if self.limit_price is not None and not _ZERO <= self.limit_price <= _ONE:
            raise ValueError("prediction limit price must be between zero and one")


@dataclass(frozen=True)
class PredictionApproval:
    order: PredictionOrder
    signed_at: datetime
    expires_at: datetime
    key_id: str
    signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "signed_at", require_aware(self.signed_at, "signed_at"))
        object.__setattr__(self, "expires_at", require_aware(self.expires_at, "expires_at"))
        if not self.key_id or not self.signature or self.expires_at <= self.signed_at:
            raise ValueError("prediction approval identity and valid expiry are required")


class PredictionApprovalSigner:
    def __init__(self, secret: bytes, key_id: str = "prediction-sandbox-v1") -> None:
        if len(secret) < 16 or not key_id:
            raise ValueError("prediction sandbox signing key and key ID are required")
        self._secret = secret
        self.key_id = key_id

    def approve(
        self,
        order: PredictionOrder,
        *,
        now: datetime,
        ttl: timedelta = timedelta(minutes=2),
    ) -> PredictionApproval:
        now = require_aware(now, "now")
        expires_at = min(order.expires_at, now + ttl)
        payload = self._payload(order, now, expires_at)
        signature = hmac.new(
            self._secret, canonical_json(payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return PredictionApproval(order, now, expires_at, self.key_id, signature)

    def verify(self, approval: PredictionApproval) -> bool:
        if approval.key_id != self.key_id:
            return False
        payload = self._payload(
            approval.order, approval.signed_at, approval.expires_at
        )
        expected = hmac.new(
            self._secret, canonical_json(payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, approval.signature)

    def _payload(
        self, order: PredictionOrder, signed_at: datetime, expires_at: datetime
    ) -> Mapping[str, object]:
        return {
            "order": order,
            "signed_at": signed_at,
            "expires_at": expires_at,
            "key_id": self.key_id,
        }


@dataclass(frozen=True)
class PredictionPosition:
    market_id: str
    outcome: PredictionOutcome
    quantity: int
    cost_basis: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost_basis", _decimal(self.cost_basis, "cost_basis"))
        if not self.market_id or self.quantity < 1 or self.cost_basis < 0:
            raise ValueError("prediction position must have identity and positive quantity")

    @property
    def average_price(self) -> Decimal:
        return self.cost_basis / self.quantity


@dataclass(frozen=True)
class PredictionTradeReceipt:
    order_id: str
    status: str
    executed_at: datetime
    price: Decimal | None = None
    quantity: int = 0
    fee: Decimal = _ZERO
    cash_change: Decimal = _ZERO

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "executed_at", require_aware(self.executed_at, "executed_at")
        )
        object.__setattr__(self, "fee", _decimal(self.fee, "fee"))
        object.__setattr__(self, "cash_change", _decimal(self.cash_change, "cash_change"))
        if self.price is not None:
            object.__setattr__(self, "price", _decimal(self.price, "price"))
        if not self.order_id or not self.status or self.quantity < 0 or self.fee < 0:
            raise ValueError("prediction receipt is invalid")


@dataclass(frozen=True)
class PredictionSettlement:
    settlement_id: str
    market_id: str
    yes_payout: Decimal
    finalized_at: datetime
    source_event_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "finalized_at", require_aware(self.finalized_at, "finalized_at")
        )
        object.__setattr__(self, "yes_payout", _decimal(self.yes_payout, "yes_payout"))
        if not self.settlement_id or not self.market_id or not self.source_event_id:
            raise ValueError("prediction settlement identity and source are required")
        if not _ZERO <= self.yes_payout <= _ONE:
            raise ValueError("prediction settlement payout must be between zero and one")


@dataclass(frozen=True)
class PredictionSettlementReceipt:
    settlement_id: str
    market_id: str
    settled_at: datetime
    yes_payout: Decimal
    gross_payout: Decimal
    cash_payout: Decimal
    rounding_adjustment: Decimal
    settlement_fee: Decimal = _ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "settled_at", require_aware(self.settled_at, "settled_at"))
        for field_name in (
            "yes_payout",
            "gross_payout",
            "cash_payout",
            "rounding_adjustment",
            "settlement_fee",
        ):
            object.__setattr__(
                self, field_name, _decimal(getattr(self, field_name), field_name)
            )
        if not self.settlement_id or not self.market_id:
            raise ValueError("prediction settlement receipt identity is required")
        if not _ZERO <= self.yes_payout <= _ONE:
            raise ValueError("prediction settlement receipt payout is invalid")
        if min(self.gross_payout, self.cash_payout, self.settlement_fee) < 0:
            raise ValueError("prediction settlement receipt amounts cannot be negative")


@dataclass(frozen=True)
class PredictionLedgerEvent:
    event_id: str
    event_type: str
    occurred_at: datetime
    payload_json: str
    digest: str


@dataclass(frozen=True)
class PredictionAccountSnapshot:
    cash: Decimal
    open_cost: Decimal
    positions: tuple[PredictionPosition, ...]
    realized_pnl: Decimal
    fees_paid: Decimal
    events: int


class PredictionSandboxLedger:
    def __init__(self, config: PredictionSandboxConfig) -> None:
        self.config = config
        self._cash = config.initial_cash
        self._positions: dict[tuple[str, PredictionOutcome], PredictionPosition] = {}
        self._realized_pnl = _ZERO
        self._fees_paid = _ZERO
        self._orders: dict[str, tuple[str, PredictionTradeReceipt]] = {}
        self._settlements: dict[str, tuple[str, PredictionSettlementReceipt]] = {}
        self._events: list[PredictionLedgerEvent] = []

    @property
    def events(self) -> tuple[PredictionLedgerEvent, ...]:
        return tuple(self._events)

    def position(
        self, market_id: str, outcome: PredictionOutcome
    ) -> PredictionPosition | None:
        return self._positions.get((market_id, outcome))

    def account(self) -> PredictionAccountSnapshot:
        positions = tuple(
            sorted(self._positions.values(), key=lambda item: (item.market_id, item.outcome))
        )
        return PredictionAccountSnapshot(
            self._cash,
            sum((item.cost_basis for item in positions), _ZERO),
            positions,
            self._realized_pnl,
            self._fees_paid,
            len(self._events),
        )

    def order_receipt(self, order: PredictionOrder) -> PredictionTradeReceipt | None:
        existing = self._orders.get(order.order_id)
        if existing is None:
            return None
        if existing[0] != sha256_digest(order):
            raise RuntimeError("prediction order ID was reused with different contents")
        return existing[1]

    def record_terminal_order(
        self, order: PredictionOrder, receipt: PredictionTradeReceipt
    ) -> PredictionTradeReceipt:
        existing = self.order_receipt(order)
        if existing is not None:
            if canonical_json(existing) != canonical_json(receipt):
                raise RuntimeError("prediction order receipt identity conflict")
            return existing
        self._orders[order.order_id] = (sha256_digest(order), receipt)
        self._append_event(
            "order", receipt.executed_at, {"order": order, "receipt": receipt}
        )
        return receipt

    def execute_trade(
        self,
        order: PredictionOrder,
        *,
        price: Decimal,
        fee: Decimal,
        occurred_at: datetime,
    ) -> PredictionTradeReceipt:
        existing = self.order_receipt(order)
        if existing is not None:
            return existing
        price = _decimal(price, "trade price")
        fee = _decimal(fee, "trade fee")
        occurred_at = require_aware(occurred_at, "occurred_at")
        key = (order.market_id, order.outcome)
        current = self._positions.get(key)
        notional = price * order.quantity
        if order.action is PredictionAction.BUY:
            cash_change = -(notional + fee)
            next_quantity = (current.quantity if current else 0) + order.quantity
            next_cost = (current.cost_basis if current else _ZERO) + notional
            next_position = PredictionPosition(
                order.market_id, order.outcome, next_quantity, next_cost
            )
            if self._cash + cash_change < 0:
                raise PermissionError("prediction sandbox order exceeds available cash")
        else:
            if current is None or current.quantity < order.quantity:
                raise PermissionError("prediction sandbox does not permit naked short sales")
            average = current.average_price
            released_cost = average * order.quantity
            cash_change = notional - fee
            self._realized_pnl += notional - released_cost - fee
            next_quantity = current.quantity - order.quantity
            next_position = (
                PredictionPosition(
                    order.market_id,
                    order.outcome,
                    next_quantity,
                    current.cost_basis - released_cost,
                )
                if next_quantity
                else None
            )
        self._cash += cash_change
        self._fees_paid += fee
        if next_position is None:
            self._positions.pop(key, None)
        else:
            self._positions[key] = next_position
        receipt = PredictionTradeReceipt(
            order.order_id,
            "filled",
            occurred_at,
            price,
            order.quantity,
            fee,
            cash_change,
        )
        self._orders[order.order_id] = (sha256_digest(order), receipt)
        self._append_event(
            "fill", occurred_at, {"order": order, "receipt": receipt}
        )
        return receipt

    def settle(
        self,
        settlement: PredictionSettlement,
        market: PredictionMarketState,
        *,
        occurred_at: datetime,
    ) -> PredictionSettlementReceipt:
        occurred_at = require_aware(occurred_at, "occurred_at")
        digest = sha256_digest(settlement)
        existing = self._settlements.get(settlement.market_id)
        if existing is not None:
            if existing[0] != digest:
                raise RuntimeError("prediction market received conflicting settlement")
            return existing[1]
        if market.market_id != settlement.market_id:
            raise ValueError("prediction settlement market identity mismatch")
        if market.status is not PredictionMarketStatus.FINALIZED:
            raise PermissionError("prediction settlement requires finalized market state")
        if settlement.finalized_at > occurred_at:
            raise ValueError("prediction settlement is from the future")
        if market.observed_at < settlement.finalized_at:
            raise ValueError("finalized market state predates the settlement")
        yes = self._positions.get((settlement.market_id, PredictionOutcome.YES))
        no = self._positions.get((settlement.market_id, PredictionOutcome.NO))
        gross = (
            (settlement.yes_payout * yes.quantity if yes else _ZERO)
            + ((_ONE - settlement.yes_payout) * no.quantity if no else _ZERO)
        )
        cash_payout = gross.quantize(
            self.config.settlement_increment, rounding=ROUND_HALF_UP
        )
        adjustment = cash_payout - gross
        released_cost = (yes.cost_basis if yes else _ZERO) + (
            no.cost_basis if no else _ZERO
        )
        self._cash += cash_payout
        self._realized_pnl += cash_payout - released_cost
        self._positions.pop((settlement.market_id, PredictionOutcome.YES), None)
        self._positions.pop((settlement.market_id, PredictionOutcome.NO), None)
        receipt = PredictionSettlementReceipt(
            settlement.settlement_id,
            settlement.market_id,
            occurred_at,
            settlement.yes_payout,
            gross,
            cash_payout,
            adjustment,
        )
        self._settlements[settlement.market_id] = (digest, receipt)
        self._append_event(
            "settlement",
            occurred_at,
            {"settlement": settlement, "receipt": receipt},
        )
        return receipt

    def verify_integrity(self) -> int:
        for ordinal, event in enumerate(self._events):
            payload = json.loads(event.payload_json)
            if event.digest != sha256_digest(payload):
                raise RuntimeError(f"prediction event digest mismatch: {event.event_id}")
            expected = sha256_digest({"ordinal": ordinal, "payload": payload})
            if event.event_id != expected:
                raise RuntimeError(f"prediction event identity mismatch: {event.event_id}")
            if payload.get("event_type") != event.event_type:
                raise RuntimeError(f"prediction event type mismatch: {event.event_id}")
        return len(self._events)

    def _append_event(
        self, event_type: str, occurred_at: datetime, payload: Mapping[str, object]
    ) -> None:
        body = {"event_type": event_type, "occurred_at": occurred_at, **payload}
        payload_json = canonical_json(body)
        normalized = json.loads(payload_json)
        event_id = sha256_digest({"ordinal": len(self._events), "payload": normalized})
        self._events.append(
            PredictionLedgerEvent(
                event_id,
                event_type,
                occurred_at,
                payload_json,
                sha256_digest(normalized),
            )
        )


class PredictionSettlementSandbox:
    def __init__(
        self,
        config: PredictionSandboxConfig,
        ledger: PredictionSandboxLedger,
        signer: PredictionApprovalSigner,
        market_resolver: Callable[[str], PredictionMarketState],
        *,
        enabled: bool = False,
        strategy_eligible: bool = False,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.signer = signer
        self.market_resolver = market_resolver
        self.enabled = enabled
        self.strategy_eligible = strategy_eligible

    def submit(
        self, approval: PredictionApproval, *, now: datetime
    ) -> PredictionTradeReceipt:
        now = require_aware(now, "now")
        self._validate_gates()
        if not self.signer.verify(approval):
            raise PermissionError("prediction sandbox approval signature is invalid")
        order = approval.order
        existing = self.ledger.order_receipt(order)
        if existing is not None:
            return existing
        if now >= approval.expires_at or now >= order.expires_at:
            raise PermissionError("prediction sandbox approval has expired")
        if order.quantity > self.config.max_order_contracts:
            raise PermissionError("prediction order exceeds the contract limit")
        market = self.market_resolver(order.market_id)
        if market.market_id != order.market_id:
            raise ValueError("prediction market resolver returned the wrong market")
        self._validate_market(market, now)
        price = market.executable_price(order.outcome, order.action)
        if order.limit_price is not None:
            outside = (
                order.action is PredictionAction.BUY and price > order.limit_price
            ) or (
                order.action is PredictionAction.SELL and price < order.limit_price
            )
            if outside:
                return self.ledger.record_terminal_order(
                    order,
                    PredictionTradeReceipt(order.order_id, "not_filled", now),
                )
        fee = prediction_trade_fee(
            price,
            order.quantity,
            order.liquidity,
            (
                market.taker_fee_multiplier
                if order.liquidity is PredictionLiquidity.TAKER
                else market.maker_fee_multiplier
            ),
            self.config,
        )
        maximum_debit = price * order.quantity + fee
        if order.action is PredictionAction.BUY and maximum_debit > order.max_cost:
            raise PermissionError("prediction fill exceeds signed maximum cost")
        self._validate_cost_limits(order, price)
        return self.ledger.execute_trade(order, price=price, fee=fee, occurred_at=now)

    def settle(
        self, settlement: PredictionSettlement, *, now: datetime
    ) -> PredictionSettlementReceipt:
        now = require_aware(now, "now")
        if not self.enabled:
            raise PermissionError("prediction settlement sandbox is disabled")
        market = self.market_resolver(settlement.market_id)
        if market.observed_at > now:
            raise ValueError("prediction market observation is from the future")
        return self.ledger.settle(settlement, market, occurred_at=now)

    def _validate_gates(self) -> None:
        if not self.enabled:
            raise PermissionError("prediction settlement sandbox is disabled")
        if not self.strategy_eligible:
            raise PermissionError("prediction strategy has not passed evidence gates")

    def _validate_market(self, market: PredictionMarketState, now: datetime) -> None:
        if market.status is not PredictionMarketStatus.ACTIVE:
            raise PermissionError("prediction market is not active for trading")
        if market.observed_at > now:
            raise ValueError("prediction market observation is from the future")
        if now - market.observed_at > self.config.max_market_age:
            raise ValueError("prediction market observation is stale")

    def _validate_cost_limits(self, order: PredictionOrder, price: Decimal) -> None:
        if order.action is PredictionAction.SELL:
            return
        account = self.ledger.account()
        order_cost = price * order.quantity
        market_cost = sum(
            (item.cost_basis for item in account.positions if item.market_id == order.market_id),
            _ZERO,
        )
        if market_cost + order_cost > self.config.max_market_cost:
            raise PermissionError("prediction order exceeds per-market cost limit")
        if account.open_cost + order_cost > self.config.max_total_cost:
            raise PermissionError("prediction order exceeds total open-cost limit")


def prediction_trade_fee(
    price: Decimal | float | str,
    quantity: int,
    liquidity: PredictionLiquidity,
    multiplier: Decimal | float | str,
    config: PredictionSandboxConfig,
) -> Decimal:
    price_value = _decimal(price, "fee price")
    multiplier_value = _decimal(multiplier, "fee multiplier")
    if not _ZERO <= price_value <= _ONE:
        raise ValueError("prediction fee price must be between zero and one")
    if not isinstance(liquidity, PredictionLiquidity):
        raise ValueError("prediction fee liquidity is invalid")
    if (
        not isinstance(quantity, int)
        or isinstance(quantity, bool)
        or quantity < 1
    ):
        raise ValueError("prediction fee quantity must be a positive whole number")
    if multiplier_value not in {_ZERO, _ONE}:
        raise ValueError("prediction fee multiplier must be zero or one")
    coefficient = (
        config.taker_fee_coefficient
        if liquidity is PredictionLiquidity.TAKER
        else config.maker_fee_coefficient
    )
    raw = multiplier_value * coefficient * quantity * price_value * (_ONE - price_value)
    if raw == 0:
        return _ZERO
    units = (raw / config.fee_increment).to_integral_value(rounding=ROUND_CEILING)
    return units * config.fee_increment


class PredictionScenarioStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class PredictionScenarioResult:
    scenario_id: str
    label: str
    status: PredictionScenarioStatus
    checks: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class PredictionSandboxReport:
    generated_at: datetime
    config_version: str
    rule_effective_date: date
    scenarios: tuple[PredictionScenarioResult, ...]
    network_access: bool = False
    venue_credentials_used: bool = False
    real_orders_placed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generated_at", require_aware(self.generated_at, "generated_at")
        )

    @property
    def passed(self) -> int:
        return sum(
            item.status is PredictionScenarioStatus.PASSED for item in self.scenarios
        )

    @property
    def failed(self) -> int:
        return len(self.scenarios) - self.passed

    @property
    def successful(self) -> bool:
        return self.failed == 0


def prediction_scenario_names() -> tuple[str, ...]:
    return tuple(_PREDICTION_SCENARIOS)


def run_prediction_sandbox_scenarios(
    scenario: str = "all",
    *,
    config: PredictionSandboxConfig | None = None,
    generated_at: datetime | None = None,
) -> PredictionSandboxReport:
    config = config or PredictionSandboxConfig()
    selected = tuple(_PREDICTION_SCENARIOS) if scenario == "all" else (scenario,)
    unknown = [item for item in selected if item not in _PREDICTION_SCENARIOS]
    if unknown:
        raise ValueError(f"unknown prediction sandbox scenario: {unknown[0]}")
    results = tuple(
        _run_scenario_safely(item, _PREDICTION_SCENARIOS[item], config)
        for item in selected
    )
    return PredictionSandboxReport(
        generated_at or datetime.now(timezone.utc),
        config.version,
        config.fee_schedule_effective_date,
        results,
    )


def render_prediction_sandbox_report(
    report: PredictionSandboxReport, output_format: str = "text"
) -> str:
    if output_format == "json":
        payload = asdict(report)
        payload["generated_at"] = report.generated_at.isoformat()
        payload["rule_effective_date"] = report.rule_effective_date.isoformat()
        payload["passed"] = report.passed
        payload["failed"] = report.failed
        payload["successful"] = report.successful
        return json.dumps(payload, sort_keys=True, indent=2, default=str)
    if output_format == "markdown":
        lines = [
            "## Prediction settlement sandbox",
            "",
            (
                f"**{'PASS' if report.successful else 'FAIL'}** · "
                f"{report.passed}/{len(report.scenarios)} scenarios passed · "
                f"policy `{report.config_version}` · rules {report.rule_effective_date} · "
                "network disabled · credentials unused · real orders 0"
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
        raise ValueError("prediction sandbox format must be text, json, or markdown")
    lines = [
        (
            f"Prediction settlement sandbox: {'PASS' if report.successful else 'FAIL'} "
            f"passed={report.passed} failed={report.failed} "
            f"policy={report.config_version} rules={report.rule_effective_date} "
            "network=false credentials=false real_orders=0"
        )
    ]
    for item in report.scenarios:
        lines.append(f"{item.scenario_id}: {item.status.value} - {item.detail}")
        lines.extend(f"  check: {check}" for check in item.checks)
    return "\n".join(lines)


def _run_scenario_safely(
    scenario_id: str,
    scenario: tuple[
        str, Callable[[PredictionSandboxConfig], tuple[tuple[str, ...], str]]
    ],
    config: PredictionSandboxConfig,
) -> PredictionScenarioResult:
    label, operation = scenario
    try:
        with patch(
            "socket.create_connection",
            side_effect=PermissionError("network access is disabled in prediction sandbox"),
        ):
            checks, detail = operation(config)
        return PredictionScenarioResult(
            scenario_id, label, PredictionScenarioStatus.PASSED, checks, detail
        )
    except Exception as exc:
        return PredictionScenarioResult(
            scenario_id,
            label,
            PredictionScenarioStatus.FAILED,
            (),
            f"{type(exc).__name__}: {exc}",
        )


def _yes_settlement_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, signer, market = _sandbox(config)
    order = _order("prediction-yes", PredictionOutcome.YES, quantity=10, max_cost="6")
    receipt = sandbox.submit(signer.approve(order, now=SANDBOX_TIME), now=SANDBOX_TIME)
    finalized = replace(market, status=PredictionMarketStatus.FINALIZED)
    sandbox.market_resolver = lambda _: finalized
    sandbox.strategy_eligible = False
    settlement = PredictionSettlement(
        "settle-yes", market.market_id, _ONE, SANDBOX_TIME, "public-settlement-yes"
    )
    payout = sandbox.settle(settlement, now=SANDBOX_TIME)
    _require(receipt.price == Decimal("0.55"), "yes order did not use executable ask")
    _require(payout.cash_payout == Decimal("10.00"), "yes winner did not pay one dollar")
    _require(ledger.position(market.market_id, PredictionOutcome.YES) is None, "position remained")
    return (
        (
            "ask-side Yes fill",
            "$1 winning payout",
            "position closed",
            "settlement survived eligibility closure",
        ),
        "finalized Yes outcome paid winning contracts and closed inventory",
    )


def _no_settlement_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, signer, market = _sandbox(config)
    order = _order("prediction-no", PredictionOutcome.NO, quantity=5, max_cost="2.5")
    receipt = sandbox.submit(signer.approve(order, now=SANDBOX_TIME), now=SANDBOX_TIME)
    finalized = replace(market, status=PredictionMarketStatus.FINALIZED)
    sandbox.market_resolver = lambda _: finalized
    payout = sandbox.settle(
        PredictionSettlement(
            "settle-no", market.market_id, _ZERO, SANDBOX_TIME, "public-settlement-no"
        ),
        now=SANDBOX_TIME,
    )
    _require(receipt.price == Decimal("0.47"), "no order did not use complement ask")
    _require(payout.cash_payout == Decimal("5.00"), "no winner payout was incorrect")
    _require(ledger.verify_integrity() == 2, "prediction ledger integrity failed")
    return (
        ("complement No price", "$1 No payout", "ledger verified"),
        "finalized No outcome paid the complementary contract value",
    )


def _scalar_rounding_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    sandbox, _, signer, market = _sandbox(config)
    for outcome, identifier, quantity in (
        (PredictionOutcome.YES, "scalar-yes", 3),
        (PredictionOutcome.NO, "scalar-no", 2),
    ):
        order = _order(identifier, outcome, quantity=quantity, max_cost="3")
        sandbox.submit(signer.approve(order, now=SANDBOX_TIME), now=SANDBOX_TIME)
    finalized = replace(market, status=PredictionMarketStatus.FINALIZED)
    sandbox.market_resolver = lambda _: finalized
    receipt = sandbox.settle(
        PredictionSettlement(
            "settle-scalar",
            market.market_id,
            Decimal("0.333"),
            SANDBOX_TIME,
            "public-settlement-scalar",
        ),
        now=SANDBOX_TIME,
    )
    _require(receipt.gross_payout == Decimal("2.333"), "scalar payout was incorrect")
    _require(receipt.cash_payout == Decimal("2.33"), "scalar payout was not cent-rounded")
    _require(
        receipt.rounding_adjustment == Decimal("-0.003"),
        "settlement rounding adjustment was not retained",
    )
    _require(receipt.settlement_fee == 0, "settlement fee was charged")
    return (
        ("scalar payout", "cent rounding", "zero settlement fee"),
        "rule-defined scalar values paid complementary Yes and No holdings",
    )


def _lifecycle_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, signer, market = _sandbox(config)
    order = _order("lifecycle-order", PredictionOutcome.YES, quantity=1, max_cost="1")
    sandbox.submit(signer.approve(order, now=SANDBOX_TIME), now=SANDBOX_TIME)
    settlement = PredictionSettlement(
        "settle-lifecycle", market.market_id, _ONE, SANDBOX_TIME, "public-lifecycle"
    )
    for status in (
        PredictionMarketStatus.DETERMINED,
        PredictionMarketStatus.DISPUTED,
        PredictionMarketStatus.AMENDED,
    ):
        sandbox.market_resolver = lambda _, status=status: replace(market, status=status)
        try:
            sandbox.settle(settlement, now=SANDBOX_TIME)
        except PermissionError as exc:
            _require("finalized" in str(exc), "lifecycle failed for unrelated reason")
        else:
            raise AssertionError(f"{status.value} market settled before finalization")
    _require(
        ledger.position(market.market_id, PredictionOutcome.YES) is not None,
        "position changed",
    )
    return (
        ("determination blocked", "dispute blocked", "amendment blocked"),
        "settlement remained pending until the market reached finalized state",
    )


def _idempotency_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    sandbox, _, signer, market = _sandbox(config)
    order = _order("idempotent-order", PredictionOutcome.YES, quantity=2, max_cost="2")
    approval = signer.approve(order, now=SANDBOX_TIME)
    first_fill = sandbox.submit(approval, now=SANDBOX_TIME)
    second_fill = sandbox.submit(
        approval, now=SANDBOX_TIME + timedelta(minutes=3)
    )
    finalized = replace(market, status=PredictionMarketStatus.FINALIZED)
    sandbox.market_resolver = lambda _: finalized
    settlement = PredictionSettlement(
        "settle-idempotent", market.market_id, _ONE, SANDBOX_TIME, "public-idempotent"
    )
    first = sandbox.settle(settlement, now=SANDBOX_TIME)
    second = sandbox.settle(settlement, now=SANDBOX_TIME)
    _require(first_fill == second_fill, "order retry changed fill")
    _require(first == second, "settlement retry changed payout")
    try:
        sandbox.settle(replace(settlement, yes_payout=_ZERO), now=SANDBOX_TIME)
    except RuntimeError as exc:
        _require("conflicting" in str(exc), "conflict failed for unrelated reason")
    else:
        raise AssertionError("conflicting settlement was accepted")
    return (
        ("order retry idempotent", "settlement retry idempotent", "conflict rejected"),
        "restart-safe identities prevented duplicate or contradictory accounting",
    )


def _fees_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    taker = prediction_trade_fee(
        Decimal("0.50"), 100, PredictionLiquidity.TAKER, _ONE, config
    )
    maker = prediction_trade_fee(
        Decimal("0.50"), 100, PredictionLiquidity.MAKER, _ONE, config
    )
    exempt = prediction_trade_fee(
        Decimal("0.50"), 100, PredictionLiquidity.TAKER, _ZERO, config
    )
    _require(taker == Decimal("1.7500"), "general taker formula was incorrect")
    _require(maker == Decimal("0.4375"), "general maker formula was incorrect")
    _require(exempt == 0, "zero-multiplier series was charged")
    return (
        ("taker formula", "maker formula", "series multiplier"),
        "versioned fee coefficients and centicent rounding matched policy",
    )


def _market_safety_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    stale_market = PredictionMarketState(
        "sandbox:prediction:TEST",
        Decimal("0.53"),
        Decimal("0.55"),
        SANDBOX_TIME - config.max_market_age - timedelta(seconds=1),
    )
    ledger = PredictionSandboxLedger(config)
    signer = PredictionApprovalSigner(b"prediction-sandbox-signing-key")
    sandbox = PredictionSettlementSandbox(
        config,
        ledger,
        signer,
        lambda _: stale_market,
        enabled=True,
        strategy_eligible=True,
    )
    order = _order("stale-order", PredictionOutcome.YES, quantity=1, max_cost="1")
    try:
        sandbox.submit(signer.approve(order, now=SANDBOX_TIME), now=SANDBOX_TIME)
    except ValueError as exc:
        _require("stale" in str(exc), "stale gate failed for unrelated reason")
    else:
        raise AssertionError("stale prediction book was accepted")
    closed = replace(stale_market, observed_at=SANDBOX_TIME, status=PredictionMarketStatus.CLOSED)
    sandbox.market_resolver = lambda _: closed
    try:
        sandbox.submit(signer.approve(order, now=SANDBOX_TIME), now=SANDBOX_TIME)
    except PermissionError as exc:
        _require("not active" in str(exc), "closed gate failed for unrelated reason")
    else:
        raise AssertionError("closed prediction market accepted a trade")
    return (
        ("stale book rejected", "closed market rejected"),
        "market freshness and lifecycle gates stopped invalid simulated fills",
    )


def _risk_gates_scenario(
    config: PredictionSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    _, _, signer, market = _sandbox(config)
    order = _order("gated-order", PredictionOutcome.YES, quantity=1, max_cost="1")
    approval = signer.approve(order, now=SANDBOX_TIME)
    ledger = PredictionSandboxLedger(config)
    disabled = PredictionSettlementSandbox(
        config, ledger, signer, lambda _: market, enabled=False, strategy_eligible=True
    )
    ineligible = PredictionSettlementSandbox(
        config, ledger, signer, lambda _: market, enabled=True, strategy_eligible=False
    )
    for sandbox, message in ((disabled, "disabled"), (ineligible, "evidence")):
        try:
            sandbox.submit(approval, now=SANDBOX_TIME)
        except PermissionError as exc:
            _require(message in str(exc), "runtime gate failed for unrelated reason")
        else:
            raise AssertionError("closed prediction sandbox gate accepted an order")
    enabled = PredictionSettlementSandbox(
        config, ledger, signer, lambda _: market, enabled=True, strategy_eligible=True
    )
    sell = _order(
        "naked-sale",
        PredictionOutcome.YES,
        action=PredictionAction.SELL,
        quantity=1,
        max_cost="1",
    )
    try:
        enabled.submit(signer.approve(sell, now=SANDBOX_TIME), now=SANDBOX_TIME)
    except PermissionError as exc:
        _require("naked short" in str(exc), "short gate failed for unrelated reason")
    else:
        raise AssertionError("naked prediction sale was accepted")
    _require(ledger.events == (), "closed gates or rejected sale mutated ledger")
    return (
        ("runtime gate", "evidence gate", "no naked short", "ledger unchanged"),
        "independent controls and fully collateralized inventory failed closed",
    )


def _sandbox(
    config: PredictionSandboxConfig,
) -> tuple[
    PredictionSettlementSandbox,
    PredictionSandboxLedger,
    PredictionApprovalSigner,
    PredictionMarketState,
]:
    market = PredictionMarketState(
        "sandbox:prediction:TEST",
        Decimal("0.53"),
        Decimal("0.55"),
        SANDBOX_TIME,
    )
    ledger = PredictionSandboxLedger(config)
    signer = PredictionApprovalSigner(b"prediction-sandbox-signing-key")
    sandbox = PredictionSettlementSandbox(
        config,
        ledger,
        signer,
        lambda _: market,
        enabled=True,
        strategy_eligible=True,
    )
    return sandbox, ledger, signer, market


def _order(
    order_id: str,
    outcome: PredictionOutcome,
    *,
    action: PredictionAction = PredictionAction.BUY,
    quantity: int,
    max_cost: Decimal | str,
) -> PredictionOrder:
    return PredictionOrder(
        order_id,
        "prediction-sandbox-strategy",
        "sandbox:prediction:TEST",
        outcome,
        action,
        quantity,
        _decimal(max_cost, "max_cost"),
        SANDBOX_TIME,
        SANDBOX_TIME + timedelta(minutes=5),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_PREDICTION_SCENARIOS: dict[
    str,
    tuple[str, Callable[[PredictionSandboxConfig], tuple[tuple[str, ...], str]]],
] = {
    "yes-settlement": ("Yes contract settlement", _yes_settlement_scenario),
    "no-settlement": ("No contract settlement", _no_settlement_scenario),
    "scalar-rounding": ("Scalar payout and rounding", _scalar_rounding_scenario),
    "lifecycle": ("Determination and dispute lifecycle", _lifecycle_scenario),
    "idempotency": ("Order and settlement idempotency", _idempotency_scenario),
    "fees": ("Maker and taker fees", _fees_scenario),
    "market-safety": ("Market freshness and closure", _market_safety_scenario),
    "risk-gates": ("Independent risk gates", _risk_gates_scenario),
}
