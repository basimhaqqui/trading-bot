from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import AssetClass, Forecast, ForecastKind, Instrument, MarketEventType
from trading_bot.core.serialization import require_aware, sha256_digest, utc_now
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.costs import EconomicCostRegistry
from trading_bot.evaluation.economics import EconomicGateConfig, EconomicStatus, build_economic_report
from trading_bot.evaluation.reporting import EdgeStatus, EvaluationGateConfig, build_walk_forward_report
from trading_bot.evaluation.scoring import ScoreKind
from trading_bot.execution.alpaca import AlpacaAccount, AlpacaPaperAdapter, AlpacaPaperClient
from trading_bot.execution.control import DeterministicExecutor, ExecutionReceipt
from trading_bot.execution.operations import (
    PaperControlStore,
    PaperExecutionLedger,
    PaperReconciler,
)
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    Position,
    TimeInForce,
)


@dataclass(frozen=True)
class PaperRiskConfig:
    max_daily_loss_pct: float = 0.01
    risk_per_trade_pct: float = 0.0025
    max_plan_notional_pct: float = 0.05
    max_quote_age: timedelta = timedelta(minutes=20)
    max_bar_age: timedelta = timedelta(days=2)
    limit_buffer_bps: float = 10.0
    min_outcomes: int = 30
    min_economic_trades: int = 30
    version: str = "alpaca-paper-v1"

    def __post_init__(self) -> None:
        if not 0 < self.max_daily_loss_pct < 1:
            raise ValueError("max_daily_loss_pct must be between zero and one")
        if not 0 < self.risk_per_trade_pct <= self.max_plan_notional_pct < 1:
            raise ValueError("paper allocation percentages are invalid")
        if self.max_quote_age <= timedelta(0) or self.max_bar_age <= timedelta(0):
            raise ValueError("market data age limits must be positive")
        if not 0 <= self.limit_buffer_bps <= 100:
            raise ValueError("limit buffer must be between zero and 100 bps")
        if self.min_outcomes < 2 or self.min_economic_trades < 2:
            raise ValueError("evidence minimums must be at least two")
        if not self.version:
            raise ValueError("paper risk policy version is required")


@dataclass(frozen=True)
class CandidateEligibility:
    candidates: frozenset[tuple[str, ScoreKind]]
    forecast_status: Mapping[tuple[str, ScoreKind], EdgeStatus]
    economic_status: Mapping[tuple[str, ScoreKind], EconomicStatus]


@dataclass(frozen=True)
class PaperAllocationPlan:
    generated_at: datetime
    account_equity: float
    account_daily_return: float
    intents: tuple[OrderIntent, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class PaperCycleResult:
    plan: PaperAllocationPlan
    receipts: tuple[ExecutionReceipt, ...]
    rejected: tuple[str, ...]


def load_paper_risk_config(path: str | Path) -> PaperRiskConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paper execution policy must be an object")
    expected = {
        "version",
        "max_daily_loss_pct",
        "risk_per_trade_pct",
        "max_plan_notional_pct",
        "max_quote_age_seconds",
        "max_bar_age_seconds",
        "limit_buffer_bps",
        "min_outcomes",
        "min_economic_trades",
    }
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            "paper execution policy keys mismatch: "
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    return PaperRiskConfig(
        max_daily_loss_pct=float(raw["max_daily_loss_pct"]),
        risk_per_trade_pct=float(raw["risk_per_trade_pct"]),
        max_plan_notional_pct=float(raw["max_plan_notional_pct"]),
        max_quote_age=timedelta(seconds=float(raw["max_quote_age_seconds"])),
        max_bar_age=timedelta(seconds=float(raw["max_bar_age_seconds"])),
        limit_buffer_bps=float(raw["limit_buffer_bps"]),
        min_outcomes=int(raw["min_outcomes"]),
        min_economic_trades=int(raw["min_economic_trades"]),
        version=str(raw["version"]),
    )


def candidate_eligibility(
    audit: AuditLedger,
    costs: EconomicCostRegistry,
    config: PaperRiskConfig,
) -> CandidateEligibility:
    forecasts = audit.forecasts()
    scores = audit.forecast_scores()
    forecast_report = build_walk_forward_report(
        forecasts,
        scores,
        EvaluationGateConfig(min_independent_outcomes=config.min_outcomes),
        locked_decisions=audit.evaluation_decisions(),
    )
    economic_report = build_economic_report(
        forecasts,
        scores,
        forecast_report,
        costs,
        EconomicGateConfig(min_trades=config.min_economic_trades),
    )
    forecast_status = {
        (group.specialist_id, group.kind): group.status for group in forecast_report.groups
    }
    economic_status = {
        (group.specialist_id, group.kind): group.status
        for group in economic_report.evaluations
    }
    candidates = frozenset(
        key
        for key, status in forecast_status.items()
        if status is EdgeStatus.CANDIDATE
        and economic_status.get(key) is EconomicStatus.CANDIDATE
    )
    return CandidateEligibility(candidates, forecast_status, economic_status)


class AlpacaPaperAllocator:
    def __init__(
        self,
        store: PointInTimeStore,
        audit: AuditLedger,
        client: AlpacaPaperClient,
        controls: PaperControlStore,
        costs: EconomicCostRegistry,
        config: PaperRiskConfig | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.client = client
        self.controls = controls
        self.costs = costs
        self.config = config or PaperRiskConfig()

    def plan(self, *, now: datetime | None = None) -> PaperAllocationPlan:
        now = require_aware(now or utc_now(), "now")
        account = self.client.account(observed_at=now)
        skipped: list[str] = []
        if not account.can_trade:
            skipped.append("paper account is not active for trading")
        if account.daily_return <= -self.config.max_daily_loss_pct:
            skipped.append("paper account breached the daily loss limit")
        control = self.controls.status()
        if not control.ready:
            skipped.append("paper execution control is locked")
        eligibility = candidate_eligibility(self.audit, self.costs, self.config)
        if not eligibility.candidates:
            skipped.append("no strategy passed both forecast and after-cost gates")

        latest = self._latest_forecasts(now)
        intents: list[OrderIntent] = []
        total_budget = account.equity * self.config.max_plan_notional_pct
        per_trade = account.equity * self.config.risk_per_trade_pct
        for forecast in latest:
            key = (forecast.specialist_id, _score_kind(forecast))
            if key[1] is None or key not in eligibility.candidates:
                continue
            try:
                instrument = self.store.instrument(forecast.instrument_id)
            except KeyError:
                skipped.append(f"{forecast.forecast_id}: instrument is missing")
                continue
            if instrument.venue != "alpaca" or instrument.asset_class not in {
                AssetClass.EQUITY,
                AssetClass.OPTION,
            }:
                continue
            try:
                intent = self._intent(forecast, instrument, now, min(per_trade, total_budget))
            except ValueError as exc:
                skipped.append(f"{forecast.forecast_id}: {exc}")
                continue
            if intent is not None:
                intents.append(intent)
                total_budget -= intent.notional
            if total_budget <= 0:
                break
        if not account.can_trade or account.daily_return <= -self.config.max_daily_loss_pct:
            intents = []
        if not control.ready:
            intents = []
        return PaperAllocationPlan(
            now,
            account.equity,
            account.daily_return,
            tuple(intents),
            tuple(dict.fromkeys(skipped)),
        )

    def _latest_forecasts(self, now: datetime) -> tuple[Forecast, ...]:
        latest: dict[tuple[str, str], Forecast] = {}
        for forecast in self.audit.forecasts():
            if forecast.valid_until <= now:
                continue
            key = (forecast.specialist_id, forecast.instrument_id)
            existing = latest.get(key)
            if existing is None or (forecast.generated_at, forecast.forecast_id) > (
                existing.generated_at,
                existing.forecast_id,
            ):
                latest[key] = forecast
        return tuple(sorted(latest.values(), key=lambda item: item.forecast_id))

    def _intent(
        self,
        forecast: Forecast,
        instrument: Instrument,
        now: datetime,
        budget: float,
    ) -> OrderIntent | None:
        if budget <= 0:
            return None
        if forecast.kind is not ForecastKind.RETURN_DISTRIBUTION:
            raise ValueError("forecast kind has no Alpaca paper payoff mapping")
        predicted = forecast.values.get("predicted_return")
        if not isinstance(predicted, (int, float)) or not math.isfinite(float(predicted)):
            raise ValueError("return forecast lacks a finite predicted_return")
        if float(predicted) == 0:
            return None
        price = self._fresh_price(instrument, now)
        multiplier = instrument.multiplier
        raw_quantity = budget / (price * multiplier)
        if instrument.asset_class is AssetClass.OPTION:
            quantity = float(math.floor(raw_quantity))
        else:
            quantity = math.floor(raw_quantity * 1_000_000) / 1_000_000
        if quantity <= 0:
            raise ValueError("allocation is too small for one executable unit")
        notional = quantity * price * multiplier
        side = OrderSide.BUY if float(predicted) > 0 else OrderSide.SELL
        buffer = self.config.limit_buffer_bps / 10_000
        max_price = price * (1 + buffer) if side is OrderSide.BUY else None
        min_price = price * (1 - buffer) if side is OrderSide.SELL else None
        intent_id = "paper-" + sha256_digest(
            {
                "forecast_id": forecast.forecast_id,
                "side": side.value,
                "quantity": quantity,
                "price": price,
            }
        )[:32]
        return OrderIntent(
            intent_id=intent_id,
            strategy_id=forecast.specialist_id,
            model_version=forecast.model_version,
            instrument_id=instrument.instrument_id,
            venue=instrument.venue,
            asset_class=instrument.asset_class,
            side=side,
            notional=notional,
            environment=ExecutionEnvironment.PAPER,
            allowed_order_types=(OrderType.LIMIT,),
            expires_at=min(forecast.valid_until, now + timedelta(minutes=5)),
            max_price=max_price,
            min_price=min_price,
            created_at=now,
            quantity=quantity,
            time_in_force=TimeInForce.DAY,
            forecast_id=forecast.forecast_id,
        )

    def _fresh_price(self, instrument: Instrument, now: datetime) -> float:
        if instrument.asset_class is AssetClass.OPTION:
            events = self.store.latest_events(
                now,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.QUOTE,
                limit=1,
            )
            maximum_age = self.config.max_quote_age
            if not events:
                raise ValueError("no option quote is available")
            bid = _finite_number(events[-1].payload.get("bid_price"))
            ask = _finite_number(events[-1].payload.get("ask_price"))
            if bid is None or ask is None or bid <= 0 or ask < bid:
                raise ValueError("option quote is not executable")
            price = (bid + ask) / 2
        else:
            events = self.store.latest_events(
                now,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.BAR,
                limit=1,
            )
            maximum_age = self.config.max_bar_age
            if not events:
                raise ValueError("no stock bar is available")
            price = _finite_number(events[-1].payload.get("close"))
            if price is None or price <= 0:
                raise ValueError("stock bar lacks a valid close")
        if now - events[-1].available_at > maximum_age:
            raise ValueError("market data is stale")
        return price


class PaperExecutionService:
    def __init__(
        self,
        store: PointInTimeStore,
        audit: AuditLedger,
        client: AlpacaPaperClient,
        controls: PaperControlStore,
        costs: EconomicCostRegistry,
        signer: ApprovalSigner,
        risk_limits: RiskLimits,
        *,
        config: PaperRiskConfig | None = None,
        submission_enabled: bool = False,
        execution_ledger: PaperExecutionLedger | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.client = client
        self.controls = controls
        self.costs = costs
        self.signer = signer
        self.risk_limits = risk_limits
        self.config = config or PaperRiskConfig()
        self.submission_enabled = submission_enabled
        self.execution_ledger = execution_ledger or PaperExecutionLedger(store.path)

    def run(self, *, now: datetime | None = None) -> PaperCycleResult:
        now = require_aware(now or utc_now(), "now")
        allocator = AlpacaPaperAllocator(
            self.store,
            self.audit,
            self.client,
            self.controls,
            self.costs,
            self.config,
        )
        plan = allocator.plan(now=now)
        if not self.submission_enabled:
            reasons = tuple(
                dict.fromkeys((*plan.skipped, "paper submission interlock is disabled"))
            )
            return PaperCycleResult(plan, (), reasons)
        control = self.controls.status()
        if not control.ready:
            return PaperCycleResult(plan, (), ("paper execution control is locked",))

        account = self.client.account(observed_at=now)
        if not account.can_trade:
            return PaperCycleResult(plan, (), ("paper account is not active for trading",))
        if account.daily_return <= -self.config.max_daily_loss_pct:
            self.controls.activate_kill_switch(
                reason="automatic daily loss limit breach", now=now
            )
            return PaperCycleResult(plan, (), ("daily loss limit activated kill switch",))
        if not plan.intents:
            return PaperCycleResult(plan, (), tuple(plan.skipped))

        reconciliation = PaperReconciler(
            self.client, self.execution_ledger, self.audit
        ).run(observed_at=now)
        if not reconciliation.clean:
            self.controls.activate_kill_switch(
                reason="automatic reconciliation mismatch", now=now
            )
            return PaperCycleResult(
                plan,
                (),
                (
                    "reconciliation mismatch activated kill switch: "
                    f"missing={len(reconciliation.missing_remote_client_order_ids)} "
                    f"unexpected={len(reconciliation.unexpected_remote_client_order_ids)}",
                ),
            )

        instruments_by_symbol = {
            item.symbol: item for item in self.store.instruments() if item.venue == "alpaca"
        }
        portfolio = self.client.portfolio_snapshot(instruments_by_symbol, observed_at=now)
        governor = RiskGovernor(self.risk_limits, self.signer)
        governor.kill_switch_active = not control.ready
        adapter = AlpacaPaperAdapter(
            self.client,
            self.store.instrument,
            trading_enabled=True,
        )
        executor = DeterministicExecutor(self.signer, adapter)
        eligibility = candidate_eligibility(self.audit, self.costs, self.config)
        forecasts = {item.forecast_id: item for item in self.audit.forecasts()}
        receipts: list[ExecutionReceipt] = []
        rejected: list[str] = []
        for intent in plan.intents:
            forecast = forecasts.get(intent.forecast_id or "")
            if forecast is None:
                rejected.append(f"{intent.intent_id}: linked forecast is missing")
                continue
            key = (forecast.specialist_id, _score_kind(forecast))
            if key[1] is None or key not in eligibility.candidates:
                rejected.append(f"{intent.intent_id}: strategy is no longer eligible")
                continue
            if (
                forecast.specialist_id != intent.strategy_id
                or forecast.model_version != intent.model_version
                or forecast.instrument_id != intent.instrument_id
                or forecast.valid_until <= now
            ):
                rejected.append(f"{intent.intent_id}: forecast linkage is invalid or expired")
                continue
            evidence_error = self._evidence_error(forecast)
            if evidence_error:
                rejected.append(f"{intent.intent_id}: {evidence_error}")
                continue
            instrument = self.store.instrument(intent.instrument_id)
            decision = governor.evaluate(
                intent, instrument=instrument, portfolio=portfolio, now=now
            )
            self.audit.append_order_intent(intent)
            if not decision.approved:
                self.audit.append_risk_decision(decision)
                rejected.append(f"{intent.intent_id}: {'; '.join(decision.reasons)}")
                continue
            approval = governor.approve(
                intent, instrument=instrument, portfolio=portfolio, now=now
            )
            self.audit.append_risk_decision(approval.decision)
            self.audit.append_approval(approval)
            try:
                receipt = executor.execute(approval, now=now)
            except Exception as exc:
                rejected.append(f"{intent.intent_id}: {type(exc).__name__}: {exc}")
                continue
            self.audit.append_execution_receipt(receipt)
            receipts.append(receipt)
            portfolio = _portfolio_after_intent(portfolio, intent)
        return PaperCycleResult(plan, tuple(receipts), tuple(rejected))

    def _evidence_error(self, forecast: Forecast) -> str | None:
        for event_id in forecast.evidence_event_ids:
            try:
                event = self.store.event(event_id)
            except KeyError:
                return f"forecast evidence {event_id} is missing"
            if event.available_at > forecast.generated_at:
                return f"forecast evidence {event_id} was unavailable at decision time"
        return None


def _score_kind(forecast: Forecast) -> ScoreKind | None:
    return {
        ForecastKind.BINARY_PROBABILITY: ScoreKind.BINARY,
        ForecastKind.FUNDING_RATE: ScoreKind.FUNDING,
        ForecastKind.VOLATILITY: ScoreKind.VOLATILITY,
        ForecastKind.RETURN_DISTRIBUTION: ScoreKind.RETURN,
    }.get(forecast.kind)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _portfolio_after_intent(
    portfolio: PortfolioSnapshot, intent: OrderIntent
) -> PortfolioSnapshot:
    positions = {item.instrument_id: item for item in portfolio.positions}
    existing = positions.get(intent.instrument_id)
    current = existing.signed_notional if existing else 0.0
    change = intent.notional if intent.side is OrderSide.BUY else -intent.notional
    positions[intent.instrument_id] = Position(
        intent.instrument_id,
        intent.venue,
        intent.asset_class,
        current + change,
    )
    return PortfolioSnapshot(
        portfolio.snapshot_at,
        portfolio.equity,
        max(0.0, portfolio.available_cash - (0 if intent.reduce_only else intent.notional)),
        tuple(sorted(positions.values(), key=lambda item: item.instrument_id)),
    )
