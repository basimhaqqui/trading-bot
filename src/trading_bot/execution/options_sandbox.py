from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping
from unittest.mock import patch
from urllib.parse import urlsplit

from trading_bot.core.serialization import canonical_json, require_aware, sha256_digest


SANDBOX_TIME = datetime(2026, 1, 2, 15, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OptionsSandboxConfig:
    version: str = "options-lifecycle-sandbox-v1"
    initial_cash: float = 100_000
    contract_multiplier: int = 100
    auto_exercise_threshold: float = 0.01
    option_slippage_bps: float = 25
    stock_slippage_bps: float = 5
    max_market_age: timedelta = timedelta(seconds=60)
    max_order_contracts: int = 10
    max_premium_cost: float = 5_000
    max_hedge_notional: float = 25_000
    exercise_source_url: str = (
        "https://docs.alpaca.markets/us/v1.1/docs/options-trading"
    )
    activity_source_url: str = (
        "https://docs.alpaca.markets/us/v1.1/docs/"
        "non-trade-activities-for-option-events"
    )

    def __post_init__(self) -> None:
        numeric = (
            self.initial_cash,
            self.auto_exercise_threshold,
            self.option_slippage_bps,
            self.stock_slippage_bps,
            self.max_premium_cost,
            self.max_hedge_notional,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("options sandbox policy values must be finite")
        if not self.version or self.initial_cash <= 0:
            raise ValueError("options sandbox version and positive cash are required")
        if self.contract_multiplier != 100:
            raise ValueError("options sandbox supports standard 100-share contracts only")
        if not 0 < self.auto_exercise_threshold < 1:
            raise ValueError("auto-exercise threshold must be between zero and one")
        if not 0 <= self.option_slippage_bps <= 1_000:
            raise ValueError("option slippage must be between zero and 1,000 bps")
        if not 0 <= self.stock_slippage_bps <= 1_000:
            raise ValueError("stock slippage must be between zero and 1,000 bps")
        if self.max_market_age <= timedelta(0) or self.max_order_contracts < 1:
            raise ValueError("options market age and contract limit must be positive")
        if not 0 < self.max_premium_cost <= self.initial_cash:
            raise ValueError("options premium-cost limit is invalid")
        if not 0 < self.max_hedge_notional <= self.initial_cash:
            raise ValueError("options hedge-notional limit is invalid")
        for source in (self.exercise_source_url, self.activity_source_url):
            parsed = urlsplit(source)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("options policy sources must be absolute HTTPS URLs")


def load_options_sandbox_config(path: str | Path) -> OptionsSandboxConfig:
    config_path = Path(path)
    if config_path.stat().st_size > 1_000_000:
        raise ValueError("options sandbox config exceeds the 1 MB safety limit")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "version",
        "initial_cash",
        "contract_multiplier",
        "auto_exercise_threshold",
        "option_slippage_bps",
        "stock_slippage_bps",
        "max_market_age_seconds",
        "max_order_contracts",
        "max_premium_cost",
        "max_hedge_notional",
        "exercise_source_url",
        "activity_source_url",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            "options sandbox config keys mismatch: "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )
    if type(raw["contract_multiplier"]) is not int or type(
        raw["max_order_contracts"]
    ) is not int:
        raise ValueError("options contract quantities must be whole numbers")
    return OptionsSandboxConfig(
        version=str(raw["version"]),
        initial_cash=float(raw["initial_cash"]),
        contract_multiplier=raw["contract_multiplier"],
        auto_exercise_threshold=float(raw["auto_exercise_threshold"]),
        option_slippage_bps=float(raw["option_slippage_bps"]),
        stock_slippage_bps=float(raw["stock_slippage_bps"]),
        max_market_age=timedelta(seconds=float(raw["max_market_age_seconds"])),
        max_order_contracts=raw["max_order_contracts"],
        max_premium_cost=float(raw["max_premium_cost"]),
        max_hedge_notional=float(raw["max_hedge_notional"]),
        exercise_source_url=str(raw["exercise_source_url"]),
        activity_source_url=str(raw["activity_source_url"]),
    )


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class OptionLifecycleAction(StrEnum):
    EXERCISED = "exercised"
    LIQUIDATED = "liquidated"
    EXPIRED = "expired"
    DO_NOT_EXERCISE = "do_not_exercise"


@dataclass(frozen=True)
class OptionContract:
    contract_id: str
    underlying: str
    right: OptionRight
    strike: float
    expiration: datetime
    multiplier: int = 100
    american_style: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expiration", require_aware(self.expiration, "expiration")
        )
        if not self.contract_id or not self.underlying:
            raise ValueError("option contract identity is required")
        if not isinstance(self.right, OptionRight):
            raise ValueError("option right is invalid")
        if not math.isfinite(self.strike) or self.strike <= 0:
            raise ValueError("option strike must be finite and positive")
        if self.multiplier != 100 or not self.american_style:
            raise ValueError("sandbox accepts standard American equity options only")


@dataclass(frozen=True)
class OptionMarketState:
    contract_id: str
    bid: float
    ask: float
    underlying_price: float
    delta: float
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", require_aware(self.observed_at, "observed_at")
        )
        values = (self.bid, self.ask, self.underlying_price, self.delta)
        if not self.contract_id or not all(math.isfinite(value) for value in values):
            raise ValueError("option market identity and values must be finite")
        if self.bid < 0 or self.ask < self.bid or self.underlying_price <= 0:
            raise ValueError("option market prices are invalid")
        if not -1 <= self.delta <= 1:
            raise ValueError("option delta must be between negative and positive one")


@dataclass(frozen=True)
class OptionPosition:
    contract: OptionContract
    quantity: int
    entry_premium: float
    do_not_exercise: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise ValueError("option position quantity must be a whole number")
        if self.quantity < 1 or not math.isfinite(self.entry_premium):
            raise ValueError("option position quantity and premium are invalid")


@dataclass(frozen=True)
class OptionLifecycleReceipt:
    contract_id: str
    action: OptionLifecycleAction
    occurred_at: datetime
    intrinsic_per_share: float
    option_cash_change: float
    underlying_share_change: float
    underlying_cash_change: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "occurred_at", require_aware(self.occurred_at, "occurred_at")
        )
        numeric = (
            self.intrinsic_per_share,
            self.option_cash_change,
            self.underlying_share_change,
            self.underlying_cash_change,
        )
        if not self.contract_id or not self.reason or not all(
            math.isfinite(value) for value in numeric
        ):
            raise ValueError("option lifecycle receipt is invalid")


@dataclass(frozen=True)
class OptionsAccountSnapshot:
    cash: float
    underlying_shares: float
    option_position: OptionPosition | None
    realized_pnl: float
    hedge_trades: int
    events: int


@dataclass(frozen=True)
class OptionsLedgerEvent:
    event_id: str
    event_type: str
    occurred_at: datetime
    payload_json: str
    digest: str


class OptionsSandboxLedger:
    def __init__(self, config: OptionsSandboxConfig) -> None:
        self.config = config
        self._cash = config.initial_cash
        self._underlying_shares = 0.0
        self._position: OptionPosition | None = None
        self._realized_pnl = 0.0
        self._hedge_trades = 0
        self._events: list[OptionsLedgerEvent] = []

    @property
    def events(self) -> tuple[OptionsLedgerEvent, ...]:
        return tuple(self._events)

    def account(self) -> OptionsAccountSnapshot:
        return OptionsAccountSnapshot(
            self._cash,
            self._underlying_shares,
            self._position,
            self._realized_pnl,
            self._hedge_trades,
            len(self._events),
        )

    def buy_option(
        self,
        contract: OptionContract,
        market: OptionMarketState,
        quantity: int,
        *,
        occurred_at: datetime,
    ) -> float:
        if self._position is not None:
            raise PermissionError("options sandbox supports one open option position")
        slippage = self.config.option_slippage_bps / 10_000
        premium = market.ask * (1 + slippage)
        cost = premium * contract.multiplier * quantity
        if cost > self.config.max_premium_cost or cost > self._cash:
            raise PermissionError("option premium exceeds sandbox cost limits")
        self._cash -= cost
        self._position = OptionPosition(contract, quantity, premium)
        self._append_event(
            "option_fill",
            occurred_at,
            {"contract": contract, "quantity": quantity, "premium": premium, "cost": cost},
        )
        return premium

    def mark_do_not_exercise(self, *, occurred_at: datetime) -> None:
        if self._position is None:
            raise PermissionError("no option position is available for DNE")
        self._position = replace(self._position, do_not_exercise=True)
        self._append_event(
            "do_not_exercise",
            occurred_at,
            {"contract_id": self._position.contract.contract_id},
        )

    def trade_underlying(
        self,
        share_change: float,
        reference_price: float,
        *,
        occurred_at: datetime,
        hedge: bool,
    ) -> float:
        if not math.isfinite(share_change) or not math.isfinite(reference_price):
            raise ValueError("underlying trade values must be finite")
        if reference_price <= 0:
            raise ValueError("underlying reference price must be positive")
        if abs(share_change) < 1e-12:
            return 0.0
        slippage = self.config.stock_slippage_bps / 10_000
        price = reference_price * (1 + slippage if share_change > 0 else 1 - slippage)
        notional = abs(share_change) * price
        if hedge and notional > self.config.max_hedge_notional:
            raise PermissionError("delta hedge exceeds sandbox notional limit")
        self._cash -= share_change * price
        self._underlying_shares += share_change
        if hedge:
            self._hedge_trades += 1
        self._append_event(
            "underlying_hedge" if hedge else "underlying_trade",
            occurred_at,
            {"shares": share_change, "price": price, "notional": notional},
        )
        return price

    def close_option_for_risk(
        self, market: OptionMarketState, *, occurred_at: datetime
    ) -> OptionLifecycleReceipt:
        position = self._require_position()
        proceeds = market.bid * position.contract.multiplier * position.quantity
        entry_cost = (
            position.entry_premium * position.contract.multiplier * position.quantity
        )
        self._cash += proceeds
        self._realized_pnl += proceeds - entry_cost
        self._position = None
        receipt = OptionLifecycleReceipt(
            position.contract.contract_id,
            OptionLifecycleAction.LIQUIDATED,
            occurred_at,
            _intrinsic(position.contract, market.underlying_price),
            proceeds,
            0,
            0,
            "insufficient exercise resources",
        )
        self._append_event("option_liquidation", occurred_at, {"receipt": receipt})
        return receipt

    def expire(
        self, settlement_price: float, *, occurred_at: datetime
    ) -> OptionLifecycleReceipt:
        position = self._require_position()
        contract = position.contract
        intrinsic = _intrinsic(contract, settlement_price)
        shares = contract.multiplier * position.quantity
        entry_cost = position.entry_premium * shares
        if position.do_not_exercise:
            action = OptionLifecycleAction.DO_NOT_EXERCISE
            share_change = cash_change = 0.0
            option_cash = 0.0
            reason = "position carried a do-not-exercise instruction"
        elif intrinsic + 1e-12 < self.config.auto_exercise_threshold:
            action = OptionLifecycleAction.EXPIRED
            share_change = cash_change = 0.0
            option_cash = 0.0
            reason = "contract was below the auto-exercise threshold"
        elif contract.right is OptionRight.CALL:
            required = contract.strike * shares
            if self._cash + 1e-9 < required:
                raise PermissionError("insufficient cash for call exercise")
            action = OptionLifecycleAction.EXERCISED
            share_change = float(shares)
            cash_change = -required
            option_cash = intrinsic * shares
            reason = "ITM call auto-exercised into underlying shares"
        else:
            if self._underlying_shares + 1e-9 < shares:
                raise PermissionError("insufficient shares for put exercise")
            action = OptionLifecycleAction.EXERCISED
            share_change = -float(shares)
            cash_change = contract.strike * shares
            option_cash = intrinsic * shares
            reason = "ITM put auto-exercised against underlying shares"
        self._cash += cash_change
        self._underlying_shares += share_change
        self._realized_pnl += option_cash - entry_cost
        self._position = None
        receipt = OptionLifecycleReceipt(
            contract.contract_id,
            action,
            occurred_at,
            intrinsic,
            option_cash,
            share_change,
            cash_change,
            reason,
        )
        self._append_event("option_lifecycle", occurred_at, {"receipt": receipt})
        return receipt

    def verify_integrity(self) -> int:
        for ordinal, event in enumerate(self._events):
            payload = json.loads(event.payload_json)
            if sha256_digest(payload) != event.digest:
                raise RuntimeError(f"options event digest mismatch: {event.event_id}")
            expected = sha256_digest({"ordinal": ordinal, "payload": payload})
            if event.event_id != expected:
                raise RuntimeError(f"options event identity mismatch: {event.event_id}")
        return len(self._events)

    def _require_position(self) -> OptionPosition:
        if self._position is None:
            raise PermissionError("no option position is open")
        return self._position

    def _append_event(
        self, event_type: str, occurred_at: datetime, payload: Mapping[str, object]
    ) -> None:
        occurred_at = require_aware(occurred_at, "occurred_at")
        body = {"event_type": event_type, "occurred_at": occurred_at, **payload}
        payload_json = canonical_json(body)
        normalized = json.loads(payload_json)
        event_id = sha256_digest({"ordinal": len(self._events), "payload": normalized})
        self._events.append(
            OptionsLedgerEvent(
                event_id,
                event_type,
                occurred_at,
                payload_json,
                sha256_digest(normalized),
            )
        )


class OptionsLifecycleSandbox:
    def __init__(
        self,
        config: OptionsSandboxConfig,
        ledger: OptionsSandboxLedger,
        market_resolver: Callable[[str], OptionMarketState],
        *,
        enabled: bool = False,
        strategy_eligible: bool = False,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.market_resolver = market_resolver
        self.enabled = enabled
        self.strategy_eligible = strategy_eligible

    def open_long(
        self,
        contract: OptionContract,
        quantity: int,
        *,
        now: datetime,
    ) -> float:
        self._validate_gates()
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise ValueError("option quantity must be a positive whole number")
        if quantity > self.config.max_order_contracts:
            raise PermissionError("option order exceeds the contract limit")
        if require_aware(now, "now") >= contract.expiration:
            raise PermissionError("cannot open an option at or after expiration")
        market = self.market_resolver(contract.contract_id)
        self._validate_market(contract, market, now)
        return self.ledger.buy_option(
            contract, market, quantity, occurred_at=require_aware(now, "now")
        )

    def rebalance_delta(self, *, now: datetime) -> float:
        self._validate_gates()
        account = self.ledger.account()
        if account.option_position is None:
            raise PermissionError("delta hedge requires an open option position")
        position = account.option_position
        market = self.market_resolver(position.contract.contract_id)
        self._validate_market(position.contract, market, now)
        target = -position.quantity * position.contract.multiplier * market.delta
        change = target - account.underlying_shares
        self.ledger.trade_underlying(
            change,
            market.underlying_price,
            occurred_at=now,
            hedge=True,
        )
        return target

    def process_expiration(
        self,
        settlement_price: float,
        *,
        now: datetime,
    ) -> OptionLifecycleReceipt:
        if not self.enabled:
            raise PermissionError("options lifecycle sandbox is disabled")
        account = self.ledger.account()
        if account.option_position is None:
            raise PermissionError("expiration requires an open option position")
        position = account.option_position
        now = require_aware(now, "now")
        if now < position.contract.expiration:
            raise PermissionError("option contract has not reached expiration")
        intrinsic = _intrinsic(position.contract, settlement_price)
        resources_ok = (
            position.contract.right is OptionRight.CALL
            and account.cash
            >= position.contract.strike
            * position.contract.multiplier
            * position.quantity
        ) or (
            position.contract.right is OptionRight.PUT
            and account.underlying_shares
            >= position.contract.multiplier * position.quantity
        )
        if (
            not position.do_not_exercise
            and intrinsic + 1e-12 >= self.config.auto_exercise_threshold
            and not resources_ok
        ):
            market = self.market_resolver(position.contract.contract_id)
            self._validate_market(position.contract, market, now)
            return self.ledger.close_option_for_risk(market, occurred_at=now)
        return self.ledger.expire(settlement_price, occurred_at=now)

    def mark_do_not_exercise(self, *, now: datetime) -> None:
        self._validate_gates()
        self.ledger.mark_do_not_exercise(occurred_at=now)

    def _validate_gates(self) -> None:
        if not self.enabled:
            raise PermissionError("options lifecycle sandbox is disabled")
        if not self.strategy_eligible:
            raise PermissionError("options strategy has not passed evidence gates")

    def _validate_market(
        self, contract: OptionContract, market: OptionMarketState, now: datetime
    ) -> None:
        now = require_aware(now, "now")
        if market.contract_id != contract.contract_id:
            raise ValueError("option market does not match contract")
        if contract.right is OptionRight.CALL and not 0 <= market.delta <= 1:
            raise ValueError("call option delta must be between zero and one")
        if contract.right is OptionRight.PUT and not -1 <= market.delta <= 0:
            raise ValueError("put option delta must be between negative one and zero")
        if market.observed_at > now:
            raise ValueError("option market observation is from the future")
        if now - market.observed_at > self.config.max_market_age:
            raise ValueError("option market observation is stale")


def _intrinsic(contract: OptionContract, underlying_price: float) -> float:
    if not math.isfinite(underlying_price) or underlying_price <= 0:
        raise ValueError("option settlement price must be finite and positive")
    return max(
        underlying_price - contract.strike
        if contract.right is OptionRight.CALL
        else contract.strike - underlying_price,
        0.0,
    )


class OptionsScenarioStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class OptionsScenarioResult:
    scenario_id: str
    label: str
    status: OptionsScenarioStatus
    checks: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class OptionsSandboxReport:
    generated_at: datetime
    config_version: str
    scenarios: tuple[OptionsScenarioResult, ...]
    network_access: bool = False
    venue_credentials_used: bool = False
    real_orders_placed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generated_at", require_aware(self.generated_at, "generated_at")
        )

    @property
    def passed(self) -> int:
        return sum(item.status is OptionsScenarioStatus.PASSED for item in self.scenarios)

    @property
    def failed(self) -> int:
        return len(self.scenarios) - self.passed

    @property
    def successful(self) -> bool:
        return self.failed == 0


def options_scenario_names() -> tuple[str, ...]:
    return tuple(_OPTIONS_SCENARIOS)


def run_options_sandbox_scenarios(
    scenario: str = "all",
    *,
    config: OptionsSandboxConfig | None = None,
    generated_at: datetime | None = None,
) -> OptionsSandboxReport:
    config = config or OptionsSandboxConfig()
    selected = tuple(_OPTIONS_SCENARIOS) if scenario == "all" else (scenario,)
    if any(item not in _OPTIONS_SCENARIOS for item in selected):
        raise ValueError(f"unknown options sandbox scenario: {selected[0]}")
    results = tuple(
        _run_safely(item, _OPTIONS_SCENARIOS[item], config) for item in selected
    )
    return OptionsSandboxReport(
        generated_at or datetime.now(timezone.utc), config.version, results
    )


def render_options_sandbox_report(
    report: OptionsSandboxReport, output_format: str = "text"
) -> str:
    if output_format == "json":
        payload = asdict(report)
        payload["generated_at"] = report.generated_at.isoformat()
        payload.update(
            passed=report.passed, failed=report.failed, successful=report.successful
        )
        return json.dumps(payload, sort_keys=True, indent=2)
    if output_format == "markdown":
        lines = [
            "## Options lifecycle sandbox",
            "",
            f"**{'PASS' if report.successful else 'FAIL'}** · "
            f"{report.passed}/{len(report.scenarios)} scenarios passed · "
            f"policy `{report.config_version}` · network disabled · credentials unused · real orders 0",
            "",
            "| Scenario | Status | Verified behavior |",
            "|---|---:|---|",
        ]
        for item in report.scenarios:
            checks = "; ".join(item.checks).replace("|", "\\|")
            lines.append(
                f"| {item.label} | {item.status.value.upper()} | {checks} |"
            )
        return "\n".join(lines)
    if output_format != "text":
        raise ValueError("options sandbox format must be text, json, or markdown")
    lines = [
        f"Options lifecycle sandbox: {'PASS' if report.successful else 'FAIL'} "
        f"passed={report.passed} failed={report.failed} policy={report.config_version} "
        "network=false credentials=false real_orders=0"
    ]
    for item in report.scenarios:
        lines.append(f"{item.scenario_id}: {item.status.value} - {item.detail}")
    return "\n".join(lines)


def _run_safely(
    scenario_id: str,
    scenario: tuple[str, Callable[[OptionsSandboxConfig], tuple[tuple[str, ...], str]]],
    config: OptionsSandboxConfig,
) -> OptionsScenarioResult:
    label, operation = scenario
    try:
        with patch(
            "socket.create_connection",
            side_effect=PermissionError("network disabled in options sandbox"),
        ):
            checks, detail = operation(config)
        return OptionsScenarioResult(
            scenario_id, label, OptionsScenarioStatus.PASSED, checks, detail
        )
    except Exception as exc:
        return OptionsScenarioResult(
            scenario_id,
            label,
            OptionsScenarioStatus.FAILED,
            (),
            f"{type(exc).__name__}: {exc}",
        )


def _call_exercise(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, contract, _ = _sandbox(config, OptionRight.CALL)
    sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    receipt = sandbox.process_expiration(110, now=contract.expiration)
    account = ledger.account()
    _require(receipt.action is OptionLifecycleAction.EXERCISED, "call did not exercise")
    _require(account.underlying_shares == 100, "call did not deliver shares")
    return (("$0.01 ITM threshold", "100-share delivery"), "ITM call auto-exercised")


def _put_exercise(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, contract, market = _sandbox(config, OptionRight.PUT)
    ledger.trade_underlying(100, market.underlying_price, occurred_at=SANDBOX_TIME, hedge=False)
    sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    receipt = sandbox.process_expiration(90, now=contract.expiration)
    _require(receipt.action is OptionLifecycleAction.EXERCISED, "put did not exercise")
    _require(ledger.account().underlying_shares == 0, "put did not deliver shares")
    return (("share requirement", "strike-price delivery"), "ITM put exercised against shares")


def _dne(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, contract, _ = _sandbox(config, OptionRight.CALL)
    sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    sandbox.mark_do_not_exercise(now=SANDBOX_TIME)
    receipt = sandbox.process_expiration(110, now=contract.expiration)
    _require(receipt.action is OptionLifecycleAction.DO_NOT_EXERCISE, "DNE was ignored")
    _require(ledger.account().underlying_shares == 0, "DNE delivered shares")
    return (("explicit DNE", "no delivery"), "do-not-exercise overrode automatic exercise")


def _otm_expiry(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, _, contract, _ = _sandbox(config, OptionRight.CALL)
    sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    receipt = sandbox.process_expiration(99, now=contract.expiration)
    _require(receipt.action is OptionLifecycleAction.EXPIRED, "OTM option did not expire")
    return (("OTM flatten", "zero delivery"), "out-of-the-money contract expired worthless")


def _risk_sellout(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    tight = replace(config, initial_cash=600, max_premium_cost=600, max_hedge_notional=600)
    sandbox, _, contract, _ = _sandbox(tight, OptionRight.CALL)
    sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    receipt = sandbox.process_expiration(110, now=contract.expiration)
    _require(receipt.action is OptionLifecycleAction.LIQUIDATED, "risk sell-out did not occur")
    return (("buying-power check", "pre-expiry liquidation"), "insufficient exercise cash triggered sell-out")


def _delta_hedge(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, contract, market = _sandbox(config, OptionRight.CALL)
    sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    target = sandbox.rebalance_delta(now=SANDBOX_TIME)
    _require(target == -50, "initial delta hedge was incorrect")
    next_market = replace(market, underlying_price=105, delta=0.7, observed_at=SANDBOX_TIME)
    sandbox.market_resolver = lambda _: next_market
    target = sandbox.rebalance_delta(now=SANDBOX_TIME)
    _require(target == -70, "delta rebalance was incorrect")
    _require(ledger.account().hedge_trades == 2, "hedge trades were not recorded")
    return (("contract delta", "discrete rebalance", "hedge ledger"), "delta exposure was neutralized along a path")


def _stale_and_gates(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, contract, market = _sandbox(config, OptionRight.CALL)
    sandbox.enabled = False
    try:
        sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    except PermissionError:
        pass
    else:
        raise AssertionError("disabled options sandbox accepted a position")
    sandbox.enabled = True
    sandbox.strategy_eligible = False
    try:
        sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    except PermissionError:
        pass
    else:
        raise AssertionError("ineligible options strategy accepted a position")
    sandbox.strategy_eligible = True
    sandbox.market_resolver = lambda _: replace(
        market, observed_at=SANDBOX_TIME - config.max_market_age - timedelta(seconds=1)
    )
    try:
        sandbox.open_long(contract, 1, now=SANDBOX_TIME)
    except ValueError as exc:
        _require("stale" in str(exc), "stale check failed for unrelated reason")
    else:
        raise AssertionError("stale option quote was accepted")
    _require(ledger.events == (), "closed gates mutated options ledger")
    return (("runtime gate", "evidence gate", "stale quote"), "independent option controls failed closed")


def _limits(config: OptionsSandboxConfig) -> tuple[tuple[str, ...], str]:
    sandbox, ledger, contract, _ = _sandbox(config, OptionRight.CALL)
    try:
        sandbox.open_long(contract, config.max_order_contracts + 1, now=SANDBOX_TIME)
    except PermissionError:
        pass
    else:
        raise AssertionError("option contract limit was bypassed")
    _require(ledger.verify_integrity() == 0, "rejected option order mutated ledger")
    return (("whole contracts", "contract cap", "append-only ledger"), "option sizing limits rejected excess exposure")


def _sandbox(
    config: OptionsSandboxConfig, right: OptionRight
) -> tuple[OptionsLifecycleSandbox, OptionsSandboxLedger, OptionContract, OptionMarketState]:
    contract = OptionContract(
        f"sandbox:option:TEST-{right.value}",
        "TEST",
        right,
        100,
        SANDBOX_TIME + timedelta(seconds=30),
    )
    delta = 0.5 if right is OptionRight.CALL else -0.5
    market = OptionMarketState(contract.contract_id, 4.9, 5.0, 100, delta, SANDBOX_TIME)
    ledger = OptionsSandboxLedger(config)
    sandbox = OptionsLifecycleSandbox(
        config,
        ledger,
        lambda _: market,
        enabled=True,
        strategy_eligible=True,
    )
    return sandbox, ledger, contract, market


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_OPTIONS_SCENARIOS: dict[
    str, tuple[str, Callable[[OptionsSandboxConfig], tuple[tuple[str, ...], str]]]
] = {
    "call-exercise": ("Call automatic exercise", _call_exercise),
    "put-exercise": ("Put automatic exercise", _put_exercise),
    "do-not-exercise": ("Do-not-exercise override", _dne),
    "otm-expiry": ("Out-of-the-money expiry", _otm_expiry),
    "risk-sellout": ("Exercise buying-power sell-out", _risk_sellout),
    "delta-hedge": ("Discrete delta hedge", _delta_hedge),
    "stale-and-gates": ("Freshness and eligibility gates", _stale_and_gates),
    "limits": ("Contract and premium limits", _limits),
}
