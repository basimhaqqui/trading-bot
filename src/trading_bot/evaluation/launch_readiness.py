from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from trading_bot.agents.hypotheses import SUPERSEDED_PAPER_REVIEW_SPECIALIST_IDS
from trading_bot.core.audit import AuditLedger
from trading_bot.core.database import DatabaseLocation, postgres_integrity_ok
from trading_bot.core.schemas import AssetClass, Instrument
from trading_bot.core.serialization import require_aware
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.costs import load_cost_registry
from trading_bot.evaluation.economics import EconomicStatus
from trading_bot.evaluation.reporting import EdgeStatus
from trading_bot.evaluation.scorecard import (
    FastPredictionCadenceSummary,
    RapidCryptoCadenceSummary,
    build_daily_scorecard,
    rapid_lane_continuity_passes,
)
from trading_bot.execution.crypto_sandbox import (
    load_crypto_sandbox_config,
    run_crypto_sandbox_scenarios,
)
from trading_bot.execution.drills import run_paper_drills
from trading_bot.execution.memecoin_sandbox import (
    load_memecoin_sandbox_config,
    run_memecoin_sandbox_scenarios,
)
from trading_bot.execution.operations import PaperControlStore, PaperExecutionLedger
from trading_bot.execution.options_sandbox import (
    load_options_sandbox_config,
    run_options_sandbox_scenarios,
)
from trading_bot.execution.prediction_sandbox import (
    load_prediction_sandbox_config,
    run_prediction_sandbox_scenarios,
)
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
)
from trading_bot.ingestion.plan import load_plan
from trading_bot.ingestion.runner import IngestionRunLedger


@dataclass(frozen=True)
class LaunchReadinessConfig:
    version: str = "launch-readiness-v1"
    roadmap_milestones: int = 18
    min_independent_outcomes: int = 30
    min_after_cost_trades: int = 30
    max_ingestion_age: timedelta = timedelta(minutes=90)
    max_consecutive_ingestion_failures: int = 0
    min_forecast_candidates: int = 1
    min_economic_candidates: int = 1
    allow_live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.version or self.roadmap_milestones != 18:
            raise ValueError("launch readiness requires the versioned 18-milestone roadmap")
        if self.min_independent_outcomes < 30 or self.min_after_cost_trades < 30:
            raise ValueError("launch evidence floors cannot be lower than 30")
        if self.max_ingestion_age <= timedelta(0):
            raise ValueError("launch ingestion age must be positive")
        if self.max_consecutive_ingestion_failures < 0:
            raise ValueError("launch ingestion failure limit cannot be negative")
        if self.min_forecast_candidates < 1 or self.min_economic_candidates < 1:
            raise ValueError("launch readiness requires candidate evidence")
        if self.allow_live_execution:
            raise ValueError("launch readiness cannot authorize live execution")


def load_launch_readiness_config(path: str | Path) -> LaunchReadinessConfig:
    config_path = Path(path)
    if config_path.stat().st_size > 1_000_000:
        raise ValueError("launch readiness config exceeds the 1 MB safety limit")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "version",
        "roadmap_milestones",
        "min_independent_outcomes",
        "min_after_cost_trades",
        "max_ingestion_age_minutes",
        "max_consecutive_ingestion_failures",
        "min_forecast_candidates",
        "min_economic_candidates",
        "allow_live_execution",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            "launch readiness config keys mismatch: "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )
    integer_fields = (
        "roadmap_milestones",
        "min_independent_outcomes",
        "min_after_cost_trades",
        "max_consecutive_ingestion_failures",
        "min_forecast_candidates",
        "min_economic_candidates",
    )
    if any(type(raw[field]) is not int for field in integer_fields):
        raise ValueError("launch readiness counts must be whole numbers")
    if type(raw["allow_live_execution"]) is not bool:
        raise ValueError("launch live-execution flag must be boolean")
    max_age = raw["max_ingestion_age_minutes"]
    if (
        type(max_age) not in (int, float)
        or isinstance(max_age, bool)
        or not math.isfinite(float(max_age))
    ):
        raise ValueError("launch ingestion age must be finite")
    return LaunchReadinessConfig(
        version=str(raw["version"]),
        roadmap_milestones=raw["roadmap_milestones"],
        min_independent_outcomes=raw["min_independent_outcomes"],
        min_after_cost_trades=raw["min_after_cost_trades"],
        max_ingestion_age=timedelta(
            minutes=float(raw["max_ingestion_age_minutes"])
        ),
        max_consecutive_ingestion_failures=raw[
            "max_consecutive_ingestion_failures"
        ],
        min_forecast_candidates=raw["min_forecast_candidates"],
        min_economic_candidates=raw["min_economic_candidates"],
        allow_live_execution=raw["allow_live_execution"],
    )


class LaunchGateCategory(StrEnum):
    SANDBOX = "sandbox"
    INTEGRITY = "integrity"
    OPERATIONS = "operations"
    EVIDENCE = "evidence"
    EXECUTION = "execution"


@dataclass(frozen=True)
class LaunchGate:
    gate_id: str
    category: LaunchGateCategory
    passed: bool
    blocking: bool
    detail: str


class LaunchReadinessStatus(StrEnum):
    NO_GO = "no_go"
    PAPER_REVIEW = "paper_review"


@dataclass(frozen=True)
class LaunchReadinessReport:
    generated_at: datetime
    config_version: str
    roadmap_completed: int
    roadmap_total: int
    status: LaunchReadinessStatus
    gates: tuple[LaunchGate, ...]
    forecast_candidates: int
    economic_candidates: int
    live_execution_authorized: bool = False
    real_orders_placed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generated_at", require_aware(self.generated_at, "generated_at")
        )
        if self.live_execution_authorized or self.real_orders_placed:
            raise ValueError("launch report cannot authorize or place live orders")

    @property
    def blockers(self) -> tuple[LaunchGate, ...]:
        return tuple(item for item in self.gates if item.blocking and not item.passed)

    @property
    def technical_successful(self) -> bool:
        technical = {
            LaunchGateCategory.SANDBOX,
            LaunchGateCategory.INTEGRITY,
            LaunchGateCategory.EXECUTION,
        }
        return all(item.passed for item in self.gates if item.category in technical)


def build_launch_readiness_report(
    database_path: str | Path,
    *,
    plan_path: str | Path,
    costs_path: str | Path,
    config_path: str | Path = "config/launch-readiness.json",
    crypto_policy_path: str | Path = "config/crypto-sandbox.json",
    prediction_policy_path: str | Path = "config/prediction-sandbox.json",
    options_policy_path: str | Path = "config/options-sandbox.json",
    memecoin_policy_path: str | Path = "config/memecoin-sandbox.json",
    as_of: datetime,
) -> LaunchReadinessReport:
    as_of = require_aware(as_of, "as_of")
    config = load_launch_readiness_config(config_path)
    database = Path(database_path)
    _initialize_readiness_stores(database)

    paper_drills = run_paper_drills("all", generated_at=as_of)
    crypto = run_crypto_sandbox_scenarios(
        config=load_crypto_sandbox_config(crypto_policy_path), generated_at=as_of
    )
    prediction = run_prediction_sandbox_scenarios(
        config=load_prediction_sandbox_config(prediction_policy_path),
        generated_at=as_of,
    )
    options = run_options_sandbox_scenarios(
        config=load_options_sandbox_config(options_policy_path), generated_at=as_of
    )
    memecoin = run_memecoin_sandbox_scenarios(
        config=load_memecoin_sandbox_config(memecoin_policy_path), generated_at=as_of
    )
    scorecard = build_daily_scorecard(
        database,
        load_plan(plan_path),
        load_cost_registry(costs_path),
        as_of=as_of,
        min_outcomes=config.min_independent_outcomes,
        min_trades=config.min_after_cost_trades,
        max_age=config.max_ingestion_age,
        max_consecutive_failures=config.max_consecutive_ingestion_failures,
        environment={},
    )
    forecast_candidates, economic_candidates = _paper_review_candidate_counts(scorecard)
    paper = PaperControlStore(database).status()
    integrity = _database_integrity(database)
    live_rejected = _probe_live_rejection(as_of)
    gates = (
        _suite_gate("paper-drills", paper_drills.successful, paper_drills.passed, 8),
        _suite_gate("crypto-sandbox", crypto.successful, crypto.passed, 8),
        _suite_gate(
            "prediction-sandbox", prediction.successful, prediction.passed, 8
        ),
        _suite_gate("options-sandbox", options.successful, options.passed, 8),
        _suite_gate("memecoin-sandbox", memecoin.successful, memecoin.passed, 8),
        LaunchGate(
            "database-integrity",
            LaunchGateCategory.INTEGRITY,
            integrity,
            True,
            "database, audit, ingestion, and paper ledgers verified"
            if integrity
            else "one or more local ledgers failed integrity verification",
        ),
        LaunchGate(
            "paper-control-locked",
            LaunchGateCategory.OPERATIONS,
            not paper.enabled and paper.kill_switch_active,
            True,
            "paper execution remains locked with kill switch active"
            if not paper.enabled and paper.kill_switch_active
            else "paper execution controls are not in the required locked state",
        ),
        LaunchGate(
            "ingestion-health",
            LaunchGateCategory.OPERATIONS,
            scorecard.ingestion.healthy,
            True,
            "all required ingestion jobs are current"
            if scorecard.ingestion.healthy
            else "required ingestion jobs are missing, stale, or failing",
        ),
        _cadence_gate(
            "rapid-crypto-continuity",
            "rapid crypto",
            scorecard.rapid_crypto_cadence,
        ),
        _cadence_gate(
            "fast-prediction-continuity",
            "fast prediction",
            scorecard.fast_prediction_cadence,
        ),
        LaunchGate(
            "forecast-candidates",
            LaunchGateCategory.EVIDENCE,
            forecast_candidates >= config.min_forecast_candidates,
            True,
            f"{forecast_candidates}/{config.min_forecast_candidates} eligible forecast candidates",
        ),
        LaunchGate(
            "after-cost-candidates",
            LaunchGateCategory.EVIDENCE,
            economic_candidates >= config.min_economic_candidates,
            True,
            f"{economic_candidates}/{config.min_economic_candidates} eligible after-cost candidates",
        ),
        LaunchGate(
            "live-intent-rejection",
            LaunchGateCategory.EXECUTION,
            live_rejected and not config.allow_live_execution,
            True,
            "independent risk governor rejected a signed live intent",
        ),
        LaunchGate(
            "live-authorization",
            LaunchGateCategory.EXECUTION,
            True,
            False,
            "live execution is structurally unavailable; strongest result is paper review",
        ),
    )
    blockers = tuple(item for item in gates if item.blocking and not item.passed)
    status = (
        LaunchReadinessStatus.PAPER_REVIEW
        if not blockers
        else LaunchReadinessStatus.NO_GO
    )
    return LaunchReadinessReport(
        as_of,
        config.version,
        config.roadmap_milestones,
        config.roadmap_milestones,
        status,
        gates,
        forecast_candidates,
        economic_candidates,
    )


def render_launch_readiness_report(
    report: LaunchReadinessReport, output_format: str = "text"
) -> str:
    if output_format == "json":
        payload = asdict(report)
        payload["generated_at"] = report.generated_at.isoformat()
        payload["blockers"] = [item.gate_id for item in report.blockers]
        payload["technical_successful"] = report.technical_successful
        return json.dumps(payload, sort_keys=True, indent=2)
    if output_format == "markdown":
        lines = [
            "## Launch readiness",
            "",
            f"**{report.status.value.upper()}** · roadmap "
            f"{report.roadmap_completed}/{report.roadmap_total} · "
            f"forecast candidates {report.forecast_candidates} · "
            f"after-cost candidates {report.economic_candidates} · live authorization false",
            "",
            "| Gate | Category | Status | Detail |",
            "|---|---|---:|---|",
        ]
        for item in report.gates:
            lines.append(
                f"| {item.gate_id} | {item.category.value} | "
                f"{'PASS' if item.passed else 'BLOCK'} | {item.detail.replace('|', '/')} |"
            )
        return "\n".join(lines)
    if output_format != "text":
        raise ValueError("launch readiness format must be text, json, or markdown")
    lines = [
        f"Launch readiness: {report.status.value.upper()} "
        f"roadmap={report.roadmap_completed}/{report.roadmap_total} "
        f"forecast_candidates={report.forecast_candidates} "
        f"economic_candidates={report.economic_candidates} live=false real_orders=0"
    ]
    for item in report.gates:
        lines.append(
            f"{item.gate_id}: {'pass' if item.passed else 'block'} - {item.detail}"
        )
    return "\n".join(lines)


def _suite_gate(
    name: str, successful: bool, passed: int, expected: int
) -> LaunchGate:
    return LaunchGate(
        name,
        LaunchGateCategory.SANDBOX,
        successful and passed == expected,
        True,
        f"{passed}/{expected} isolated scenarios passed",
    )


def _paper_review_candidate_counts(scorecard: object) -> tuple[int, int]:
    """Count only cohorts that remain eligible for supervised paper review.

    Superseded fast-settlement studies remain visible in scorecards and cannot
    be deleted, but their results cannot be pooled with v15's fresh
    preregistered cohort to unlock a readiness gate.
    """
    strategies = getattr(scorecard, "strategies")
    economics = getattr(scorecard, "economics")
    forecast_candidates = sum(
        item.status is EdgeStatus.CANDIDATE
        and item.specialist_id not in SUPERSEDED_PAPER_REVIEW_SPECIALIST_IDS
        for item in strategies
    )
    economic_candidates = sum(
        item.status is EconomicStatus.CANDIDATE
        and item.specialist_id not in SUPERSEDED_PAPER_REVIEW_SPECIALIST_IDS
        for item in economics
    )
    return forecast_candidates, economic_candidates


def _cadence_gate(
    gate_id: str,
    lane: str,
    cadence: RapidCryptoCadenceSummary | FastPredictionCadenceSummary,
) -> LaunchGate:
    observed = cadence.observed_cycles
    largest_gap = cadence.largest_gap_minutes
    bound = cadence.max_allowed_gap_minutes
    passed = rapid_lane_continuity_passes(cadence)
    if observed == 0 or largest_gap is None:
        detail = f"{lane} has no observed collection cycles in its telemetry window"
    elif passed:
        detail = (
            f"{lane} largest collection gap {largest_gap:.1f} minutes is within "
            f"the {bound:.0f}-minute bound"
        )
    else:
        detail = (
            f"{lane} largest collection gap {largest_gap:.1f} minutes exceeds "
            f"the {bound:.0f}-minute bound"
        )
    return LaunchGate(gate_id, LaunchGateCategory.OPERATIONS, passed, True, detail)


def _initialize_readiness_stores(path: DatabaseLocation) -> None:
    PointInTimeStore(path).initialize()
    AuditLedger(path).initialize()
    IngestionRunLedger(path).initialize()
    PaperControlStore(path).initialize()
    PaperExecutionLedger(path).initialize()


def _database_integrity(path: DatabaseLocation) -> bool:
    try:
        database_ok = postgres_integrity_ok(path)
        AuditLedger(path).verify_integrity()
        IngestionRunLedger(path).verify_integrity()
        PaperExecutionLedger(path).verify_integrity()
        return database_ok
    except Exception:
        return False


def _probe_live_rejection(now: datetime) -> bool:
    instrument = Instrument(
        "readiness:equity:TEST", "readiness", "TEST", AssetClass.EQUITY, "USD"
    )
    intent = OrderIntent(
        "readiness-live-probe",
        "readiness-probe",
        "v1",
        instrument.instrument_id,
        instrument.venue,
        instrument.asset_class,
        OrderSide.BUY,
        100,
        ExecutionEnvironment.LIVE,
        (OrderType.LIMIT,),
        now + timedelta(minutes=1),
        max_price=100,
        created_at=now,
        quantity=1,
    )
    governor = RiskGovernor(
        RiskLimits(
            1_000,
            1_000,
            1_000,
            {AssetClass.EQUITY: 1_000},
            allow_live=False,
        ),
        ApprovalSigner(b"launch-readiness-probe-key"),
    )
    decision = governor.evaluate(
        intent,
        instrument=instrument,
        portfolio=PortfolioSnapshot(now, 10_000, 10_000),
        now=now,
    )
    return not decision.approved and "live execution is disabled" in decision.reasons
