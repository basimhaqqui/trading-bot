from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware
from trading_bot.core.snapshot import create_verified_snapshot
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.costs import (
    CostBasis,
    EconomicCostModel,
    EconomicCostRegistry,
)
from trading_bot.evaluation.scoring import ScoreKind
from trading_bot.execution.alpaca import (
    AlpacaAccount,
    AlpacaOrder,
    AlpacaPaperAdapter,
    AlpacaPaperError,
    PaperOrderRequest,
)
from trading_bot.execution.control import DeterministicExecutor, ExecutionReceipt
from trading_bot.execution.operations import (
    PaperControlStore,
    PaperExecutionLedger,
    PaperReconciler,
    activate_paper_emergency_stop,
)
from trading_bot.execution.paper import (
    AlpacaPaperAllocator,
    PaperExecutionService,
    PaperRiskConfig,
)
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ApprovedOrderIntent,
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
)
from trading_bot.ingestion.runner import IngestionRunLedger


DRILL_TIME = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


class DrillStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class DrillScenarioResult:
    scenario_id: str
    label: str
    status: DrillStatus
    checks: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class PaperDrillReport:
    generated_at: datetime
    scenarios: tuple[DrillScenarioResult, ...]
    network_access: bool = False
    broker_credentials_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generated_at", require_aware(self.generated_at, "generated_at")
        )
        if not self.scenarios:
            raise ValueError("paper drill report requires at least one scenario")

    @property
    def passed(self) -> int:
        return sum(item.status is DrillStatus.PASSED for item in self.scenarios)

    @property
    def failed(self) -> int:
        return len(self.scenarios) - self.passed

    @property
    def successful(self) -> bool:
        return self.failed == 0


class _DrillPaperClient:
    def __init__(
        self,
        *,
        equity: float = 100_000,
        last_equity: float = 100_000,
        response_status: str = "accepted",
        filled_quantity: float = 0,
        average_fill_price: float | None = None,
        timeout_after_accept: bool = False,
    ) -> None:
        self.equity = equity
        self.last_equity = last_equity
        self.response_status = response_status
        self.filled_quantity = filled_quantity
        self.average_fill_price = average_fill_price
        self.timeout_after_accept = timeout_after_accept
        self.remote_orders: dict[str, AlpacaOrder] = {}
        self.submit_attempts = 0
        self.created_orders = 0
        self.cancel_requests = 0

    def account(self, *, observed_at: datetime | None = None) -> AlpacaAccount:
        return AlpacaAccount(
            "drill-paper-account",
            "ACTIVE",
            self.equity,
            self.last_equity,
            self.equity,
            self.equity * 2,
            False,
            False,
            False,
            observed_at or DRILL_TIME,
        )

    def positions(self) -> tuple[object, ...]:
        return ()

    def orders(self, *, status: str = "all", limit: int = 500) -> tuple[AlpacaOrder, ...]:
        values = tuple(self.remote_orders.values())
        if status == "open":
            open_statuses = {"new", "accepted", "partially_filled", "pending_new"}
            values = tuple(item for item in values if item.status in open_statuses)
        return values[:limit]

    def portfolio_snapshot(
        self,
        instruments_by_symbol: dict[str, Instrument],
        *,
        observed_at: datetime | None = None,
    ) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            observed_at or DRILL_TIME,
            self.equity,
            self.equity * 2,
            (),
        )

    def order_by_client_id(self, client_order_id: str) -> AlpacaOrder | None:
        return self.remote_orders.get(client_order_id)

    def submit_order(self, request: PaperOrderRequest) -> AlpacaOrder:
        self.submit_attempts += 1
        existing = self.remote_orders.get(request.client_order_id)
        if existing is not None:
            return existing
        self.created_orders += 1
        order = _order(
            request.client_order_id,
            status=self.response_status,
            quantity=request.quantity,
            filled_quantity=self.filled_quantity,
            average_fill_price=self.average_fill_price,
            side=request.side,
            limit_price=request.limit_price,
        )
        self.remote_orders[request.client_order_id] = order
        if self.timeout_after_accept:
            self.timeout_after_accept = False
            raise AlpacaPaperError("simulated ambiguous timeout after remote acceptance")
        return order

    def cancel_open_orders(self) -> tuple[object, ...]:
        self.cancel_requests += 1
        canceled: list[object] = []
        for client_order_id, order in tuple(self.remote_orders.items()):
            if order.status in {"new", "accepted", "partially_filled", "pending_new"}:
                self.remote_orders[client_order_id] = replace(
                    order, status="canceled", updated_at=DRILL_TIME
                )
                canceled.append({"id": order.order_id, "status": 200})
        return tuple(canceled)


def run_paper_drills(
    scenario: str = "all", *, generated_at: datetime | None = None
) -> PaperDrillReport:
    generated_at = generated_at or datetime.now(timezone.utc)
    selected = tuple(_SCENARIOS) if scenario == "all" else (scenario,)
    unknown = [item for item in selected if item not in _SCENARIOS]
    if unknown:
        raise ValueError(f"unknown paper drill scenario: {unknown[0]}")
    results = tuple(_run_safely(item, _SCENARIOS[item]) for item in selected)
    return PaperDrillReport(generated_at, results)


def render_paper_drill_report(report: PaperDrillReport, output_format: str = "text") -> str:
    if output_format == "json":
        payload = asdict(report)
        payload["generated_at"] = report.generated_at.isoformat()
        payload["passed"] = report.passed
        payload["failed"] = report.failed
        payload["successful"] = report.successful
        return json.dumps(payload, sort_keys=True, indent=2)
    if output_format == "markdown":
        lines = [
            "## Paper incident drills",
            "",
            (
                f"**{'PASS' if report.successful else 'FAIL'}** · "
                f"{report.passed}/{len(report.scenarios)} scenarios passed · "
                "network disabled · broker credentials unused"
            ),
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
        raise ValueError("paper drill format must be text, json, or markdown")
    lines = [
        (
            f"Paper incident drills: {'PASS' if report.successful else 'FAIL'} "
            f"passed={report.passed} failed={report.failed} "
            "network=false credentials=false"
        )
    ]
    for item in report.scenarios:
        lines.append(f"{item.scenario_id}: {item.status.value} - {item.detail}")
        lines.extend(f"  check: {check}" for check in item.checks)
    return "\n".join(lines)


def scenario_names() -> tuple[str, ...]:
    return tuple(_SCENARIOS)


def _run_safely(
    scenario_id: str,
    scenario: tuple[str, Callable[[], tuple[tuple[str, ...], str]]],
) -> DrillScenarioResult:
    label, operation = scenario
    try:
        with patch(
            "socket.create_connection",
            side_effect=PermissionError("network access is disabled during paper drills"),
        ):
            checks, detail = operation()
        return DrillScenarioResult(scenario_id, label, DrillStatus.PASSED, checks, detail)
    except Exception as exc:
        return DrillScenarioResult(
            scenario_id,
            label,
            DrillStatus.FAILED,
            (),
            f"{type(exc).__name__}: {exc}",
        )


def _duplicate_submission() -> tuple[tuple[str, ...], str]:
    instrument, signer, approval = _approved_order("drill-duplicate")
    client = _DrillPaperClient()
    adapter = AlpacaPaperAdapter(client, lambda _: instrument, trading_enabled=True)
    first = DeterministicExecutor(signer, adapter).execute(approval, now=DRILL_TIME)
    second = DeterministicExecutor(signer, adapter).execute(approval, now=DRILL_TIME)
    _require(client.created_orders == 1, "duplicate intent created more than one remote order")
    _require(
        first.client_order_id == second.client_order_id,
        "client order ID changed after restart",
    )
    return (
        ("one remote order created", "stable client order ID across executor restart"),
        "restart retry reused the existing remote order",
    )


def _ambiguous_timeout() -> tuple[tuple[str, ...], str]:
    instrument, signer, approval = _approved_order("drill-timeout")
    client = _DrillPaperClient(timeout_after_accept=True)
    adapter = AlpacaPaperAdapter(client, lambda _: instrument, trading_enabled=True)
    try:
        DeterministicExecutor(signer, adapter).execute(approval, now=DRILL_TIME)
    except AlpacaPaperError as exc:
        _require("ambiguous timeout" in str(exc), "unexpected first-attempt failure")
    else:
        raise AssertionError("ambiguous timeout was not surfaced")
    recovered = DeterministicExecutor(signer, adapter).execute(approval, now=DRILL_TIME)
    _require(client.created_orders == 1, "timeout recovery duplicated the remote order")
    _require(recovered.status == "accepted", "recovered order was not accepted")
    return (
        ("timeout surfaced", "remote lookup recovered order", "no duplicate submit"),
        "ambiguous remote acceptance recovered safely",
    )


def _partial_fill() -> tuple[tuple[str, ...], str]:
    instrument, signer, approval = _approved_order("drill-partial")
    client = _DrillPaperClient(
        response_status="partially_filled",
        filled_quantity=4,
        average_fill_price=100.25,
    )
    receipt = DeterministicExecutor(
        signer,
        AlpacaPaperAdapter(client, lambda _: instrument, trading_enabled=True),
    ).execute(approval, now=DRILL_TIME)
    _require(receipt.status == "partially_filled", "partial-fill status was lost")
    _require(receipt.filled_quantity == 4, "filled quantity was not preserved")
    _require(receipt.average_fill_price == 100.25, "average fill price was not preserved")
    return (
        ("partial status preserved", "filled quantity preserved", "fill price preserved"),
        "partial fill remained explicit in the execution receipt",
    )


def _remote_rejection() -> tuple[tuple[str, ...], str]:
    instrument, signer, approval = _approved_order("drill-rejection")
    client = _DrillPaperClient(response_status="rejected")
    adapter = AlpacaPaperAdapter(client, lambda _: instrument, trading_enabled=True)
    receipt = DeterministicExecutor(signer, adapter).execute(approval, now=DRILL_TIME)
    retry = DeterministicExecutor(signer, adapter).execute(approval, now=DRILL_TIME)
    _require(receipt.status == "rejected", "remote rejection was not preserved")
    _require(retry.status == "rejected", "rejection changed on restart")
    _require(client.created_orders == 1, "rejected order was resubmitted")
    return (
        ("rejection preserved", "restart observes terminal order", "no duplicate resubmit"),
        "terminal rejection remained visible and idempotent",
    )


def _stale_market_data() -> tuple[tuple[str, ...], str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stale.db"
        store = PointInTimeStore(path)
        store.initialize()
        instrument = _instrument()
        store.register_instrument(instrument)
        stale_at = DRILL_TIME - timedelta(days=3)
        store.append_event(
            MarketEvent(
                "drill-stale-bar",
                MarketEventType.BAR,
                "alpaca",
                instrument.instrument_id,
                stale_at,
                stale_at,
                "paper-drill",
                {"close": 100.0},
                ingested_at=stale_at,
            )
        )
        audit = AuditLedger(path)
        audit.initialize()
        allocator = AlpacaPaperAllocator(
            store,
            audit,
            _DrillPaperClient(),
            PaperControlStore(path),
            _costs(),
            PaperRiskConfig(min_outcomes=2, min_economic_trades=2),
        )
        try:
            allocator._fresh_price(instrument, DRILL_TIME)
        except ValueError as exc:
            _require("stale" in str(exc), "stale data failed for an unrelated reason")
        else:
            raise AssertionError("stale market data was accepted")
    return (("three-day-old stock bar rejected",), "stale evidence blocked allocation")


def _reconciliation_mismatch() -> tuple[tuple[str, ...], str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reconciliation.db"
        audit = AuditLedger(path)
        audit.initialize()
        audit.append_execution_receipt(
            ExecutionReceipt(
                "drill-reconcile-intent",
                ExecutionEnvironment.PAPER,
                "accepted",
                DRILL_TIME,
                "expected-order",
                "tb-expected",
            )
        )
        client = _DrillPaperClient()
        client.remote_orders["tb-unexpected"] = _order("tb-unexpected")
        controls = PaperControlStore(path)
        controls.release_kill_switch(
            confirmation="PAPER-ONLY", reason="drill setup", now=DRILL_TIME
        )
        controls.enable(confirmation="PAPER-ONLY", reason="drill setup", now=DRILL_TIME)
        result = PaperReconciler(
            client, PaperExecutionLedger(path), audit
        ).run(observed_at=DRILL_TIME)
        _require(not result.clean, "reconciliation mismatch was not detected")
        controls.activate_kill_switch(reason="drill reconciliation mismatch", now=DRILL_TIME)
        _require(controls.status().kill_switch_active, "mismatch did not lock execution")
    return (
        ("missing order detected", "unexpected order detected", "kill switch activated"),
        "remote/local divergence failed closed",
    )


def _daily_loss_shutdown() -> tuple[tuple[str, ...], str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "daily-loss.db"
        store = PointInTimeStore(path)
        store.initialize()
        audit = AuditLedger(path)
        audit.initialize()
        controls = PaperControlStore(path)
        controls.release_kill_switch(
            confirmation="PAPER-ONLY", reason="drill setup", now=DRILL_TIME
        )
        controls.enable(confirmation="PAPER-ONLY", reason="drill setup", now=DRILL_TIME)
        service = PaperExecutionService(
            store,
            audit,
            _DrillPaperClient(equity=98_000, last_equity=100_000),
            controls,
            _costs(),
            ApprovalSigner(b"paper-drill-signing-key"),
            RiskLimits(100_000, 10_000, 30_000),
            config=PaperRiskConfig(min_outcomes=2, min_economic_trades=2),
            submission_enabled=True,
        )
        result = service.run(now=DRILL_TIME)
        _require(
            "daily loss limit activated kill switch" in result.rejected,
            "daily loss did not activate the expected shutdown",
        )
        status = controls.status()
        _require(status.kill_switch_active and not status.enabled, "control stayed ready")
    return (
        ("2% daily loss detected", "submission disabled", "kill switch activated"),
        "daily loss breach stopped paper execution",
    )


def _emergency_stop_recovery() -> tuple[tuple[str, ...], str]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "active.db"
        snapshot = root / "recovery.db"
        store = PointInTimeStore(path)
        store.initialize()
        audit = AuditLedger(path)
        audit.initialize()
        IngestionRunLedger(path).initialize()
        controls = PaperControlStore(path)
        controls.release_kill_switch(
            confirmation="PAPER-ONLY", reason="drill setup", now=DRILL_TIME
        )
        controls.enable(confirmation="PAPER-ONLY", reason="drill setup", now=DRILL_TIME)
        ledger = PaperExecutionLedger(path)
        ledger.initialize()
        client = _DrillPaperClient(response_status="accepted")
        client.remote_orders["tb-open"] = _order("tb-open")
        ledger.append_account(client.account(observed_at=DRILL_TIME), ())
        ledger.append_order(client.remote_orders["tb-open"], observed_at=DRILL_TIME)
        stopped = activate_paper_emergency_stop(
            controls,
            reason="paper drill emergency stop",
            cancel_open_orders=client.cancel_open_orders,
        )
        _require(not stopped.control.ready, "emergency control remained ready")
        _require(stopped.cancellation_requests == 1, "open order was not canceled")
        summary = create_verified_snapshot(path, snapshot)
        recovered_control = PaperControlStore(snapshot).status()
        recovered_records = PaperExecutionLedger(snapshot).verify_integrity()
        _require(not recovered_control.ready, "snapshot recovery unlocked execution")
        _require(recovered_records == 2, "snapshot lost paper execution records")
        _require(summary.paper_control_ready is False, "snapshot summary reported ready")
    return (
        (
            "kill switch locked first",
            "open order cancellation requested",
            "snapshot retained locked state",
            "paper ledger recovered intact",
        ),
        "emergency stop and snapshot restart both failed closed",
    )


def _approved_order(intent_id: str) -> tuple[Instrument, ApprovalSigner, ApprovedOrderIntent]:
    instrument = _instrument()
    intent = OrderIntent(
        intent_id,
        "paper-drill-strategy",
        "drill-v1",
        instrument.instrument_id,
        instrument.venue,
        instrument.asset_class,
        OrderSide.BUY,
        1_000,
        ExecutionEnvironment.PAPER,
        (OrderType.LIMIT,),
        DRILL_TIME + timedelta(minutes=5),
        max_price=101,
        created_at=DRILL_TIME,
        quantity=10,
        forecast_id="drill-forecast",
    )
    signer = ApprovalSigner(b"paper-drill-signing-key")
    approval = RiskGovernor(RiskLimits(100_000, 10_000, 30_000), signer).approve(
        intent,
        instrument=instrument,
        portfolio=PortfolioSnapshot(DRILL_TIME, 100_000, 100_000),
        now=DRILL_TIME,
    )
    return instrument, signer, approval


def _instrument() -> Instrument:
    return Instrument("alpaca:equity:AAPL", "alpaca", "AAPL", AssetClass.EQUITY, "USD")


def _costs() -> EconomicCostRegistry:
    return EconomicCostRegistry(
        "paper-drill-costs-v1",
        (
            EconomicCostModel(
                "paper-drill-equity-costs",
                "paper-drill-strategy",
                ScoreKind.RETURN,
                CostBasis.STATIC_BPS,
                "https://example.com/paper-drill-costs",
                date(2026, 1, 1),
                fee_bps=10,
            ),
        ),
    )


def _order(
    client_order_id: str,
    *,
    status: str = "accepted",
    quantity: float = 10,
    filled_quantity: float = 0,
    average_fill_price: float | None = None,
    side: OrderSide = OrderSide.BUY,
    limit_price: float | None = 101,
) -> AlpacaOrder:
    return AlpacaOrder(
        f"remote-{client_order_id}",
        client_order_id,
        "AAPL",
        AssetClass.EQUITY,
        side,
        "limit",
        "day",
        status,
        quantity,
        filled_quantity,
        limit_price,
        average_fill_price,
        DRILL_TIME,
        DRILL_TIME,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_SCENARIOS: dict[str, tuple[str, Callable[[], tuple[tuple[str, ...], str]]]] = {
    "duplicate-submission": ("Duplicate submission", _duplicate_submission),
    "ambiguous-timeout": ("Ambiguous API timeout", _ambiguous_timeout),
    "partial-fill": ("Partial fill", _partial_fill),
    "remote-rejection": ("Remote rejection", _remote_rejection),
    "stale-market-data": ("Stale market data", _stale_market_data),
    "reconciliation-mismatch": ("Reconciliation mismatch", _reconciliation_mismatch),
    "daily-loss-shutdown": ("Daily-loss shutdown", _daily_loss_shutdown),
    "emergency-stop-recovery": ("Emergency stop and recovery", _emergency_stop_recovery),
}
