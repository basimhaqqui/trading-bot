from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from trading_bot.agents.hypotheses import (
    BASELINE_HYPOTHESES,
    BASELINE_HYPOTHESIS_SPECIALIST_IDS,
)
from trading_bot.agents.market_math import prediction_book_payload
from trading_bot.agents.prediction import (
    FastPredictionSettlementV7Specialist,
    FastPredictionSettlementV8Specialist,
    is_quarantined_prediction_identity_collision,
    prediction_forecast_target_time,
    prediction_settlement_event_ticker,
)
from trading_bot.core.audit import AuditLedger, AuditRecordType
from trading_bot.core.database import connect_database
from trading_bot.core.schemas import AssetClass, Forecast, MarketEvent, MarketEventType
from trading_bot.core.serialization import canonical_json, parse_datetime, require_aware
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.costs import EconomicCostRegistry
from trading_bot.evaluation.economics import (
    EconomicGateConfig,
    EconomicStatus,
    build_economic_report,
)
from trading_bot.evaluation.checkpoint import checkpointed_walk_forward_report
from trading_bot.evaluation.outcomes import forecast_label_deadline
from trading_bot.evaluation.reporting import (
    EdgeStatus,
    EvaluationGateConfig,
)
from trading_bot.evaluation.shadow import (
    FastPredictionEligibilitySummary,
    IntradayMomentumEligibilitySummary,
    ShadowResearchRunner,
)
from trading_bot.evaluation.scoring import ForecastScore
from trading_bot.ingestion.health import JobHealth, IngestionHealthReport, ingestion_health
from trading_bot.ingestion.plan import ShadowIngestionPlan
from trading_bot.execution.operations import PaperControlStore, PaperExecutionLedger


class ScorecardStatus(StrEnum):
    COLLECTING = "collecting"
    ATTENTION = "attention"
    CANDIDATE = "candidate"
    CRITICAL = "critical"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ACTION = "action"
    CRITICAL = "critical"


@dataclass(frozen=True)
class OperationalAlert:
    code: str
    severity: AlertSeverity
    message: str


@dataclass(frozen=True)
class SystemTotals:
    instruments: int
    events: int
    ingestion_runs: int
    forecasts: int
    scores: int
    order_intents: int
    approvals: int
    execution_receipts: int


@dataclass(frozen=True)
class CoverageSummary:
    asset_class: AssetClass
    instruments: int
    events: int
    forecasts: int
    scores: int


@dataclass(frozen=True)
class ResearchLaneSummary:
    specialist_id: str
    market: AssetClass
    proposed_at: datetime
    forecasts: int
    scores: int
    latest_forecast_at: datetime | None


@dataclass(frozen=True)
class MemecoinResearchSummary:
    discovered_tokens: int
    latest_profile_observations: int
    latest_pool_observations: int
    latest_authority_observations: int
    transfer_control_observations: int
    holder_concentration_observations: int
    holder_activity_observations: int
    latest_profile_observed_at: datetime | None
    latest_pool_observed_at: datetime | None
    latest_authority_observed_at: datetime | None
    latest_transfer_control_observed_at: datetime | None
    latest_holder_concentration_observed_at: datetime | None
    latest_holder_activity_observed_at: datetime | None
    blocked_unverified_tokens: int
    safety_eligible_tokens: int
    missing_hard_gates: tuple[str, ...]


@dataclass(frozen=True)
class StrategySummary:
    specialist_id: str
    kind: str
    status: EdgeStatus
    forecasts: int
    outcomes: int
    required_outcomes: int
    unique_instruments: int
    largest_instrument_share: float | None
    mean_improvement: float | None
    lower_confidence_bound: float | None
    win_rate: float | None
    reasons: tuple[str, ...]
    locked_status: EdgeStatus | None = None
    locked_at: datetime | None = None
    locked_outcomes: int | None = None
    monitoring_status: EdgeStatus | None = None


@dataclass(frozen=True)
class EconomicSummary:
    specialist_id: str
    kind: str
    status: EconomicStatus
    trades: int
    required_trades: int
    mean_net_return: float | None
    doubled_cost_mean_return: float | None
    max_drawdown: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PaperOperationsSummary:
    enabled: bool
    kill_switch_active: bool
    ready: bool
    reason: str
    reconciliation_records: int


@dataclass(frozen=True)
class OutcomeQueueSummary:
    unscored: int
    not_due: int
    due_unmatched: int
    quarantined: int
    policy_inconsistent_early_labels: int
    next_due_at: datetime | None
    oldest_due_at: datetime | None


@dataclass(frozen=True)
class PredictionOutcomePollingSummary:
    """Latest bounded Kalshi outcome-poll receipt, not a settlement assertion."""

    job_ids: tuple[str, ...]
    requested_instruments: int | None
    returned_instruments: int | None
    missing_instruments: int | None
    finished_at: datetime | None


@dataclass(frozen=True)
class StrategyOutcomeQueue:
    specialist_id: str
    kind: str
    pending: int
    not_due: int
    due_unmatched: int
    next_due_at: datetime | None
    oldest_due_at: datetime | None


@dataclass(frozen=True)
class PredictionCalibrationReadiness:
    eligible_independent_events: int
    eligible_open_events: int
    strongest_bucket_events: int
    required_bucket_events: int
    probability_bucket_radius: float
    ready: bool


@dataclass(frozen=True)
class RapidCryptoCadenceSummary:
    job_ids: tuple[str, ...]
    observed_cycles: int
    latest_started_at: datetime | None
    largest_gap_minutes: float | None
    max_allowed_gap_minutes: float
    lookback_hours: float


@dataclass(frozen=True)
class FastPredictionCadenceSummary:
    job_ids: tuple[str, ...]
    observed_cycles: int
    latest_started_at: datetime | None
    largest_gap_minutes: float | None
    max_allowed_gap_minutes: float
    lookback_hours: float


@dataclass(frozen=True)
class DailyScorecard:
    generated_at: datetime
    status: ScorecardStatus
    totals: SystemTotals
    coverage: tuple[CoverageSummary, ...]
    research_lanes: tuple[ResearchLaneSummary, ...]
    memecoin_research: MemecoinResearchSummary
    strategies: tuple[StrategySummary, ...]
    economics: tuple[EconomicSummary, ...]
    paper: PaperOperationsSummary
    outcome_queue: OutcomeQueueSummary
    prediction_outcome_polling: PredictionOutcomePollingSummary
    strategy_outcome_queues: tuple[StrategyOutcomeQueue, ...]
    prediction_calibration: PredictionCalibrationReadiness
    fast_prediction_eligibility: FastPredictionEligibilitySummary
    intraday_momentum_eligibility: IntradayMomentumEligibilitySummary
    rapid_crypto_cadence: RapidCryptoCadenceSummary
    fast_prediction_cadence: FastPredictionCadenceSummary
    ingestion: IngestionHealthReport
    alerts: tuple[OperationalAlert, ...]


def rapid_lane_continuity_passes(
    cadence: RapidCryptoCadenceSummary | FastPredictionCadenceSummary,
) -> bool:
    """Return whether a rapid lane has an observed, in-bound telemetry window."""
    return (
        cadence.observed_cycles > 0
        and cadence.largest_gap_minutes is not None
        and cadence.largest_gap_minutes <= cadence.max_allowed_gap_minutes
    )


def build_daily_scorecard(
    path: str | Path,
    plan: ShadowIngestionPlan,
    cost_registry: EconomicCostRegistry,
    *,
    as_of: datetime,
    min_outcomes: int = 30,
    min_trades: int = 30,
    max_age: timedelta = timedelta(minutes=90),
    max_consecutive_failures: int = 0,
    environment: Mapping[str, str] | None = None,
) -> DailyScorecard:
    as_of = require_aware(as_of, "as_of")
    store = PointInTimeStore(path)
    audit = AuditLedger(path)
    forecasts = audit.forecasts()
    scores = audit.forecast_scores()
    evidence_forecasts = tuple(
        forecast
        for forecast in forecasts
        if not is_quarantined_prediction_identity_collision(forecast)
    )
    evidence_forecast_ids = {forecast.forecast_id for forecast in evidence_forecasts}
    evidence_scores = tuple(
        score for score in scores if score.forecast_id in evidence_forecast_ids
    )
    evaluation, _ = checkpointed_walk_forward_report(
        audit,
        EvaluationGateConfig(min_independent_outcomes=min_outcomes),
        as_of=as_of,
    )
    economic = build_economic_report(
        forecasts,
        scores,
        evaluation,
        cost_registry,
        EconomicGateConfig(min_trades=min_trades),
    )
    health = ingestion_health(
        path,
        plan,
        as_of=as_of,
        max_age=max_age,
        max_consecutive_failures=max_consecutive_failures,
        environment=environment,
    )
    counts = audit.counts()
    database_totals, event_counts = _database_counts(path)
    instruments = store.instruments()
    asset_by_instrument = {
        instrument.instrument_id: instrument.asset_class for instrument in instruments
    }
    forecast_by_id = {forecast.forecast_id: forecast for forecast in forecasts}
    forecast_counts = {asset_class: 0 for asset_class in AssetClass}
    score_counts = {asset_class: 0 for asset_class in AssetClass}
    instrument_counts = {asset_class: 0 for asset_class in AssetClass}
    for instrument in instruments:
        instrument_counts[instrument.asset_class] += 1
    for forecast in forecasts:
        asset_class = asset_by_instrument.get(forecast.instrument_id)
        if asset_class is not None:
            forecast_counts[asset_class] += 1
    for score in scores:
        forecast = forecast_by_id.get(score.forecast_id)
        asset_class = (
            asset_by_instrument.get(forecast.instrument_id) if forecast is not None else None
        )
        if asset_class is not None:
            score_counts[asset_class] += 1

    coverage = tuple(
        CoverageSummary(
            asset_class,
            instrument_counts[asset_class],
            event_counts.get(asset_class, 0),
            forecast_counts[asset_class],
            score_counts[asset_class],
        )
        for asset_class in AssetClass
    )
    research_lanes = _research_lane_summaries(evidence_forecasts, evidence_scores)
    memecoin_research = _memecoin_research_summary(path, as_of, max_age=max_age)
    totals = SystemTotals(
        instruments=database_totals["instruments"],
        events=database_totals["events"],
        ingestion_runs=database_totals["ingestion_runs"],
        forecasts=counts[AuditRecordType.FORECAST],
        scores=counts[AuditRecordType.FORECAST_SCORE],
        order_intents=counts[AuditRecordType.ORDER_INTENT],
        approvals=counts[AuditRecordType.APPROVAL],
        execution_receipts=counts[AuditRecordType.EXECUTION_RECEIPT],
    )
    strategies = tuple(
        StrategySummary(
            item.specialist_id,
            item.kind.value,
            item.status,
            item.forecasts,
            item.independent_outcomes,
            min_outcomes,
            item.unique_instruments,
            _finite(item.largest_instrument_share),
            _finite(item.mean_improvement),
            _finite(item.lower_confidence_bound),
            _finite(item.win_rate),
            item.reasons,
            item.locked_status,
            item.locked_at,
            item.locked_outcomes,
            item.monitoring_status,
        )
        for item in evaluation.groups
    )
    economics = tuple(
        EconomicSummary(
            item.specialist_id,
            item.kind.value,
            item.status,
            item.trades,
            min_trades,
            _finite(item.mean_net_return),
            _finite(item.doubled_cost_mean_return),
            _finite(item.max_full_notional_drawdown),
            item.reasons,
        )
        for item in economic.evaluations
    )
    control = PaperControlStore(path).status()
    reconciliation_records = PaperExecutionLedger(path).verify_integrity()
    paper = PaperOperationsSummary(
        control.enabled,
        control.kill_switch_active,
        control.ready,
        control.reason,
        reconciliation_records,
    )
    outcome_queue = _outcome_queue(store, evidence_forecasts, evidence_scores, as_of)
    prediction_outcome_polling = _prediction_outcome_polling(path, plan, as_of=as_of)
    strategy_outcome_queues = _strategy_outcome_queues(
        evidence_forecasts,
        evidence_scores,
        as_of,
    )
    prediction_calibration = _prediction_calibration_readiness(path, as_of)
    shadow_research = ShadowResearchRunner(store, audit)
    fast_prediction_eligibility = shadow_research.fast_prediction_eligibility(as_of=as_of)
    intraday_momentum_eligibility = shadow_research.intraday_momentum_eligibility(
        as_of=as_of
    )
    rapid_crypto_cadence = _rapid_crypto_cadence(path, plan, as_of=as_of)
    fast_prediction_cadence = _fast_prediction_cadence(path, plan, as_of=as_of)
    alerts = _build_alerts(
        health,
        strategies,
        economics,
        totals,
        paper,
        outcome_queue,
        prediction_outcome_polling,
        fast_prediction_eligibility,
        rapid_crypto_cadence,
        fast_prediction_cadence,
    )
    return DailyScorecard(
        as_of,
        _scorecard_status(alerts),
        totals,
        coverage,
        research_lanes,
        memecoin_research,
        strategies,
        economics,
        paper,
        outcome_queue,
        prediction_outcome_polling,
        strategy_outcome_queues,
        prediction_calibration,
        fast_prediction_eligibility,
        intraday_momentum_eligibility,
        rapid_crypto_cadence,
        fast_prediction_cadence,
        health,
        alerts,
    )


def render_scorecard(scorecard: DailyScorecard, output_format: str = "text") -> str:
    if output_format == "json":
        return canonical_json(scorecard)
    if output_format == "markdown":
        return _render_markdown(scorecard)
    if output_format != "text":
        raise ValueError("output_format must be text, json, or markdown")
    lines = [
        f"daily-scorecard: {scorecard.status.value} at {scorecard.generated_at.isoformat()}",
        (
            f"totals: instruments={scorecard.totals.instruments} "
            f"events={scorecard.totals.events} forecasts={scorecard.totals.forecasts} "
            f"scores={scorecard.totals.scores} ingestion_runs={scorecard.totals.ingestion_runs}"
        ),
        (
            f"paper: enabled={str(scorecard.paper.enabled).lower()} "
            f"kill_switch={str(scorecard.paper.kill_switch_active).lower()} "
            f"ready={str(scorecard.paper.ready).lower()} "
            f"reconciliation_records={scorecard.paper.reconciliation_records}"
        ),
        (
            f"outcome_queue: unscored={scorecard.outcome_queue.unscored} "
            f"not_due={scorecard.outcome_queue.not_due} "
            f"due_unmatched={scorecard.outcome_queue.due_unmatched} "
            f"quarantined={scorecard.outcome_queue.quarantined} "
            "policy_inconsistent_early_labels="
            f"{scorecard.outcome_queue.policy_inconsistent_early_labels}"
        ),
        (
            "prediction_outcome_polling: "
            f"requested={scorecard.prediction_outcome_polling.requested_instruments} "
            f"returned={scorecard.prediction_outcome_polling.returned_instruments} "
            f"missing={scorecard.prediction_outcome_polling.missing_instruments}"
        ),
        (
            "prediction_calibration: "
            f"eligible_events={scorecard.prediction_calibration.eligible_independent_events} "
            f"open_events={scorecard.prediction_calibration.eligible_open_events} "
            f"strongest_bucket={scorecard.prediction_calibration.strongest_bucket_events}/"
            f"{scorecard.prediction_calibration.required_bucket_events} "
            f"ready={str(scorecard.prediction_calibration.ready).lower()}"
        ),
        (
            "fast_prediction_eligibility: "
            f"paired={scorecard.fast_prediction_eligibility.paired_markets} "
            f"fresh={scorecard.fast_prediction_eligibility.fresh_book_markets} "
            f"active={scorecard.fast_prediction_eligibility.active_markets} "
            f"documented_close_policy={scorecard.fast_prediction_eligibility.documented_close_policy_markets} "
            f"early_close_enabled={scorecard.fast_prediction_eligibility.early_close_enabled_markets} "
            f"early_close_disabled={scorecard.fast_prediction_eligibility.early_close_disabled_markets} "
            f"close_policy_missing={scorecard.fast_prediction_eligibility.missing_close_policy_markets} "
            f"close_policy_invalid={scorecard.fast_prediction_eligibility.invalid_close_policy_markets} "
            f"short_timer={scorecard.fast_prediction_eligibility.short_timer_markets} "
            f"horizon={scorecard.fast_prediction_eligibility.horizon_markets} "
            f"executable={scorecard.fast_prediction_eligibility.executable_markets} "
            f"unforecasted_events={scorecard.fast_prediction_eligibility.unforecasted_event_candidates} "
            f"selected={scorecard.fast_prediction_eligibility.selected_events}"
        ),
        (
            "intraday_momentum_eligibility: "
            f"observed={scorecard.intraday_momentum_eligibility.observed_instruments} "
            f"fresh={scorecard.intraday_momentum_eligibility.fresh_instruments} "
            f"lookback={scorecard.intraday_momentum_eligibility.adequate_lookback_instruments} "
            f"signals={scorecard.intraday_momentum_eligibility.signal_instruments} "
            f"v2_assigned={scorecard.intraday_momentum_eligibility.v2_assigned_instruments} "
            f"v2_signals={scorecard.intraday_momentum_eligibility.v2_signal_instruments}"
        ),
        (
            "rapid_crypto_cadence: "
            f"jobs={len(scorecard.rapid_crypto_cadence.job_ids)} "
            f"cycles={scorecard.rapid_crypto_cadence.observed_cycles} "
            f"largest_gap_minutes="
            f"{scorecard.rapid_crypto_cadence.largest_gap_minutes} "
            f"bound_minutes={scorecard.rapid_crypto_cadence.max_allowed_gap_minutes} "
            f"lookback_hours={scorecard.rapid_crypto_cadence.lookback_hours}"
        ),
        (
            "fast_prediction_cadence: "
            f"jobs={len(scorecard.fast_prediction_cadence.job_ids)} "
            f"cycles={scorecard.fast_prediction_cadence.observed_cycles} "
            f"largest_gap_minutes="
            f"{scorecard.fast_prediction_cadence.largest_gap_minutes} "
            f"bound_minutes={scorecard.fast_prediction_cadence.max_allowed_gap_minutes} "
            f"lookback_hours={scorecard.fast_prediction_cadence.lookback_hours}"
        ),
        (
            "memecoin_research: "
            f"tokens={scorecard.memecoin_research.discovered_tokens} "
            f"profiles={scorecard.memecoin_research.latest_profile_observations} "
            f"pools={scorecard.memecoin_research.latest_pool_observations} "
            f"blocked_unverified={scorecard.memecoin_research.blocked_unverified_tokens} "
            f"safety_eligible={scorecard.memecoin_research.safety_eligible_tokens}"
        ),
    ]
    for lane in scorecard.research_lanes:
        latest = lane.latest_forecast_at.isoformat() if lane.latest_forecast_at else "—"
        lines.append(
            f"research_lane: {lane.specialist_id} market={lane.market.value} "
            f"forecasts={lane.forecasts} scores={lane.scores} latest={latest}"
        )
    for alert in scorecard.alerts:
        lines.append(f"{alert.severity.value}: {alert.code}: {alert.message}")
    for strategy in scorecard.strategies:
        lines.append(
            f"{strategy.specialist_id}/{strategy.kind}: {strategy.status.value} "
            f"outcomes={strategy.outcomes}/{strategy.required_outcomes}"
        )
        if strategy.locked_status is not None:
            lines.append(
                f"  decision: locked {strategy.locked_status.value} at "
                f"{strategy.locked_outcomes} outcomes on "
                f"{strategy.locked_at.isoformat()}; "
                f"monitoring={strategy.monitoring_status.value}"
            )
    for queue in scorecard.strategy_outcome_queues:
        lines.append(
            f"{queue.specialist_id}/{queue.kind} pending: "
            f"not_due={queue.not_due} due_unmatched={queue.due_unmatched}"
        )
    return "\n".join(lines)


def render_github_alerts(scorecard: DailyScorecard) -> str:
    commands = {
        AlertSeverity.WARNING: "warning",
        AlertSeverity.ACTION: "notice",
        AlertSeverity.CRITICAL: "error",
    }
    lines = []
    for alert in scorecard.alerts:
        command = commands.get(alert.severity)
        if command is None:
            continue
        title = _github_escape(f"Trading bot: {alert.code.replace('_', ' ')}")
        message = _github_escape(alert.message)
        lines.append(f"::{command} title={title}::{message}")
    return "\n".join(lines)


def _database_counts(path: str | Path) -> tuple[dict[str, int], dict[AssetClass, int]]:
    with connect_database(path) as connection:
        totals = {
            "instruments": int(connection.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]),
            "events": int(connection.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]),
            "ingestion_runs": int(
                connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0]
            ),
        }
        rows = connection.execute(
            """
            SELECT instruments.asset_class, COUNT(market_events.event_id)
            FROM instruments
            LEFT JOIN market_events
              ON market_events.instrument_id = instruments.instrument_id
            GROUP BY instruments.asset_class
            """
        ).fetchall()
    return totals, {AssetClass(asset_class): int(count) for asset_class, count in rows}


def _research_lane_summaries(
    forecasts: tuple[Forecast, ...], scores: tuple[ForecastScore, ...]
) -> tuple[ResearchLaneSummary, ...]:
    score_counts: dict[str, int] = {}
    for score in scores:
        score_counts[score.specialist_id] = score_counts.get(score.specialist_id, 0) + 1
    summaries: list[ResearchLaneSummary] = []
    for hypothesis in BASELINE_HYPOTHESES:
        specialist_ids = BASELINE_HYPOTHESIS_SPECIALIST_IDS[hypothesis.hypothesis_id]
        lane_forecasts = tuple(
            item for item in forecasts if item.specialist_id in specialist_ids
        )
        latest = max(
            (item.generated_at for item in lane_forecasts), default=None
        )
        summaries.append(
            ResearchLaneSummary(
                hypothesis.hypothesis_id,
                hypothesis.market,
                hypothesis.proposed_at,
                len(lane_forecasts),
                sum(score_counts.get(specialist_id, 0) for specialist_id in specialist_ids),
                latest,
            )
        )
    return tuple(summaries)


def _memecoin_research_summary(
    path: str | Path, as_of: datetime, *, max_age: timedelta
) -> MemecoinResearchSummary:
    with connect_database(path) as connection:
        rows = connection.execute(
            """
            SELECT event.instrument_id, event.source, event.available_at, event.event_id,
                   event.payload_json
            FROM market_events AS event
            JOIN instruments USING (instrument_id)
            WHERE instruments.asset_class = ?
              AND event.event_type = ?
              AND event.available_at <= ?
              AND event.venue IN ('dexscreener', 'solana')
            ORDER BY event.instrument_id, event.available_at, event.event_id
            """,
            (
                AssetClass.MEMECOIN.value,
                "onchain_state",
                as_of.isoformat(),
            ),
        ).fetchall()

    latest: dict[tuple[str, str], tuple[datetime, str, Mapping[str, object]]] = {}
    for instrument_id, source, available_at, event_id, payload_json in rows:
        category = _memecoin_observation_category(source)
        if category is None:
            continue
        try:
            observed_at = parse_datetime(available_at)
            payload = json.loads(payload_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        key = (instrument_id, category)
        candidate = (observed_at, event_id, payload)
        existing = latest.get(key)
        if existing is None or candidate[:2] > existing[:2]:
            latest[key] = candidate

    by_token: dict[str, list[tuple[datetime, Mapping[str, object]]]] = {}
    profiles = 0
    pools = 0
    authorities = 0
    transfer_controls = 0
    holder_concentrations = 0
    holder_activity = 0
    latest_profile_observed_at: datetime | None = None
    latest_pool_observed_at: datetime | None = None
    latest_authority_observed_at: datetime | None = None
    latest_transfer_control_observed_at: datetime | None = None
    latest_holder_concentration_observed_at: datetime | None = None
    latest_holder_activity_observed_at: datetime | None = None
    for (instrument_id, category), (observed_at, _, payload) in latest.items():
        by_token.setdefault(instrument_id, []).append((observed_at, payload))
        if category == "profile":
            profiles += 1
            if (
                latest_profile_observed_at is None
                or observed_at > latest_profile_observed_at
            ):
                latest_profile_observed_at = observed_at
        elif category == "pool":
            pools += 1
            if latest_pool_observed_at is None or observed_at > latest_pool_observed_at:
                latest_pool_observed_at = observed_at
        elif category == "authority":
            authorities += 1
            if latest_authority_observed_at is None or observed_at > latest_authority_observed_at:
                latest_authority_observed_at = observed_at
            if payload.get("transfer_behavior_observed") is True:
                transfer_controls += 1
                if (
                    latest_transfer_control_observed_at is None
                    or observed_at > latest_transfer_control_observed_at
                ):
                    latest_transfer_control_observed_at = observed_at
        elif category == "holder_concentration":
            if payload.get("holder_concentration_observed") is True:
                holder_concentrations += 1
                if (
                    latest_holder_concentration_observed_at is None
                    or observed_at > latest_holder_concentration_observed_at
                ):
                    latest_holder_concentration_observed_at = observed_at
        else:
            holder_activity += 1
            if (
                latest_holder_activity_observed_at is None
                or observed_at > latest_holder_activity_observed_at
            ):
                latest_holder_activity_observed_at = observed_at
    blocked = 0
    eligible = 0
    missing: set[str] = set()
    hard_gates = {
        "onchain_authorities_observed": "onchain authorities",
        "holder_concentration_observed": "holder concentration",
        "transfer_behavior_observed": "transfer behavior",
        "round_trip_simulation_observed": "round-trip simulation",
    }
    for observations in by_token.values():
        # A scorecard is an operational view, not an archive lookup. A hard
        # gate outside its active health window cannot make a current token
        # look eligible; sandbox execution applies its own tighter check.
        payloads = tuple(
            payload
            for observed_at, payload in observations
            if timedelta(0) <= as_of - observed_at <= max_age
        )
        statuses = {str(payload.get("safety_status", "")) for payload in payloads}
        token_missing = False
        for field, label in hard_gates.items():
            if not any(payload.get(field) is True for payload in payloads):
                missing.add(label)
                token_missing = True
        if statuses == {"sandbox_eligible"} and not token_missing:
            eligible += 1
        else:
            blocked += 1
    return MemecoinResearchSummary(
        len(by_token),
        profiles,
        pools,
        authorities,
        transfer_controls,
        holder_concentrations,
        holder_activity,
        latest_profile_observed_at,
        latest_pool_observed_at,
        latest_authority_observed_at,
        latest_transfer_control_observed_at,
        latest_holder_concentration_observed_at,
        latest_holder_activity_observed_at,
        blocked,
        eligible,
        tuple(sorted(missing)),
    )


def _memecoin_observation_category(source: str) -> str | None:
    if source == "dexscreener-public-token-profile-v1":
        return "profile"
    if source == "dexscreener-public-token-pairs-v1":
        return "pool"
    if source in {
        "solana-rpc-get-multiple-accounts-finalized-v1",
        "solana-rpc-get-multiple-accounts-finalized-v2",
    }:
        return "authority"
    if source == "solana-rpc-token-holder-concentration-finalized-v1":
        return "holder_concentration"
    if source == "solana-rpc-finalized-holder-activity-v1":
        return "holder_activity"
    return None


def _build_alerts(
    health: IngestionHealthReport,
    strategies: tuple[StrategySummary, ...],
    economics: tuple[EconomicSummary, ...],
    totals: SystemTotals,
    paper: PaperOperationsSummary,
    outcome_queue: OutcomeQueueSummary,
    prediction_outcome_polling: PredictionOutcomePollingSummary,
    fast_prediction_eligibility: FastPredictionEligibilitySummary,
    rapid_crypto_cadence: RapidCryptoCadenceSummary,
    fast_prediction_cadence: FastPredictionCadenceSummary,
) -> tuple[OperationalAlert, ...]:
    alerts: list[OperationalAlert] = []
    unhealthy = [job.job_id for job in health.jobs if not job.healthy]
    if unhealthy:
        alerts.append(
            OperationalAlert(
                "ingestion_unhealthy",
                AlertSeverity.CRITICAL,
                f"{len(unhealthy)} ingestion job(s) need attention: {', '.join(unhealthy)}",
            )
        )
    waiting = [job for job in health.jobs if job.status == "waiting_credentials"]
    if waiting:
        alerts.append(
            OperationalAlert(
                "market_data_credentials_waiting",
                AlertSeverity.WARNING,
                _waiting_credentials_message(waiting),
            )
        )
    economic_candidates = [
        item for item in economics if item.status is EconomicStatus.CANDIDATE
    ]
    for item in economic_candidates:
        alerts.append(
            OperationalAlert(
                "economic_candidate",
                AlertSeverity.ACTION,
                f"{item.specialist_id}/{item.kind} cleared forecast and doubled-cost gates",
            )
        )
    economic_candidate_keys = {
        (item.specialist_id, item.kind) for item in economic_candidates
    }
    if paper.ready and not economic_candidates:
        alerts.append(
            OperationalAlert(
                "paper_execution_unlocked_without_candidate",
                AlertSeverity.WARNING,
                "Paper execution is unlocked while no strategy has cleared the economic gate",
            )
        )
    for item in strategies:
        if item.status is EdgeStatus.CANDIDATE and (
            item.specialist_id,
            item.kind,
        ) not in economic_candidate_keys:
            alerts.append(
                OperationalAlert(
                    "forecast_candidate",
                    AlertSeverity.ACTION,
                    f"{item.specialist_id}/{item.kind} cleared forecast gates and awaits economic eligibility",
                )
            )
    if totals.execution_receipts:
        alerts.append(
            OperationalAlert(
                "execution_receipts_present",
                AlertSeverity.WARNING,
                f"{totals.execution_receipts} controlled execution receipt(s) exist; live adapters remain disabled",
            )
        )
    if outcome_queue.due_unmatched:
        oldest = (
            outcome_queue.oldest_due_at.isoformat()
            if outcome_queue.oldest_due_at is not None
            else "unknown"
        )
        alerts.append(
            OperationalAlert(
                "outcomes_awaiting_settlement",
                AlertSeverity.INFO,
                f"{outcome_queue.due_unmatched} due forecast(s) await a public outcome; oldest target {oldest}",
            )
        )
    if outcome_queue.policy_inconsistent_early_labels:
        alerts.append(
            OperationalAlert(
                "fast_prediction_policy_inconsistent_labels",
                AlertSeverity.WARNING,
                f"{outcome_queue.policy_inconsistent_early_labels} fast prediction label(s) "
                "finalized before expected expiration without the registered "
                "early-close policy and venue corroboration; excluded from evidence",
            )
        )
    if (
        prediction_outcome_polling.missing_instruments is not None
        and prediction_outcome_polling.missing_instruments > 0
    ):
        alerts.append(
            OperationalAlert(
                "prediction_outcome_polling_incomplete",
                AlertSeverity.WARNING,
                "latest bounded Kalshi outcome poll requested "
                f"{prediction_outcome_polling.requested_instruments} tracked ticker(s) but "
                f"returned {prediction_outcome_polling.returned_instruments}; "
                "missing responses are not treated as settlements",
            )
        )
    if rapid_crypto_cadence.job_ids and rapid_crypto_cadence.observed_cycles == 0:
        alerts.append(
            OperationalAlert(
                "rapid_crypto_observation_cadence_gap",
                AlertSeverity.WARNING,
                "rapid crypto has no scheduled observations in its telemetry window; "
                "manual and legacy cycles are not prospective evidence",
            )
        )
    elif (
        rapid_crypto_cadence.largest_gap_minutes is not None
        and rapid_crypto_cadence.largest_gap_minutes
        > rapid_crypto_cadence.max_allowed_gap_minutes
    ):
        alerts.append(
            OperationalAlert(
                "rapid_crypto_observation_cadence_gap",
                AlertSeverity.WARNING,
                (
                    "rapid crypto observation gap "
                    f"{rapid_crypto_cadence.largest_gap_minutes:.1f} minutes exceeds "
                    f"the {rapid_crypto_cadence.max_allowed_gap_minutes:.0f}-minute "
                    "collection bound; missed cycles are not prospective evidence"
                ),
            )
        )
    if fast_prediction_cadence.job_ids and fast_prediction_cadence.observed_cycles == 0:
        alerts.append(
            OperationalAlert(
                "fast_prediction_observation_cadence_gap",
                AlertSeverity.WARNING,
                "fast prediction has no scheduled observations in its telemetry window; "
                "manual and legacy cycles are not prospective evidence",
            )
        )
    elif (
        fast_prediction_cadence.largest_gap_minutes is not None
        and fast_prediction_cadence.largest_gap_minutes
        > fast_prediction_cadence.max_allowed_gap_minutes
    ):
        alerts.append(
            OperationalAlert(
                "fast_prediction_observation_cadence_gap",
                AlertSeverity.WARNING,
                (
                    "fast prediction observation gap "
                    f"{fast_prediction_cadence.largest_gap_minutes:.1f} minutes exceeds "
                    f"the {fast_prediction_cadence.max_allowed_gap_minutes:.0f}-minute "
                    "collection bound; missed cycles are not prospective evidence"
                ),
            )
        )
    if (
        fast_prediction_eligibility.active_markets
        and not fast_prediction_eligibility.documented_close_policy_markets
    ):
        close_constraint_reasons: list[str] = []
        if fast_prediction_eligibility.missing_close_policy_markets:
            close_constraint_reasons.append(
                f"{fast_prediction_eligibility.missing_close_policy_markets} omit the field"
            )
        if fast_prediction_eligibility.invalid_close_policy_markets:
            close_constraint_reasons.append(
                f"{fast_prediction_eligibility.invalid_close_policy_markets} have invalid values"
            )
        alerts.append(
            OperationalAlert(
                "fast_prediction_close_policy_unavailable",
                AlertSeverity.INFO,
                (
                    f"{fast_prediction_eligibility.active_markets} active fast-settlement "
                    "market(s) lacked a documented boolean can_close_early "
                    f"constraint ({'; '.join(close_constraint_reasons)}); "
                    "no fast-lane forecast was selected"
                ),
            )
        )
    if not alerts:
        alerts.append(
            OperationalAlert(
                "evidence_collecting",
                AlertSeverity.INFO,
                "Data is healthy and no strategy has cleared the evidence gates yet",
            )
        )
    return tuple(alerts)


def _rapid_crypto_cadence(
    path: str | Path,
    plan: ShadowIngestionPlan,
    *,
    as_of: datetime,
    max_gap: timedelta = timedelta(minutes=30),
    lookback: timedelta = timedelta(hours=24),
) -> RapidCryptoCadenceSummary:
    """Report recent gaps for the fixed fifteen-minute crypto collection lane.

    This is operational telemetry only: it never fills a missing candle, produces a
    forecast, or changes any evidence or eligibility threshold. It includes the
    intervals from the rolling-window boundary to the first observed cycle and from
    the most recent cycle through ``as_of``. A newly started lane therefore cannot
    be mistaken for continuously collected evidence.
    """
    if max_gap <= timedelta(0):
        raise ValueError("rapid crypto cadence bound must be positive")
    if lookback <= timedelta(0):
        raise ValueError("rapid crypto cadence lookback must be positive")
    as_of = require_aware(as_of, "as_of")
    job_ids = tuple(
        job.job_id
        for job in plan.jobs
        if job.enabled
        and job.venue == "coinbase"
        and job.dataset == "candles"
        and job.granularity == "FIFTEEN_MINUTE"
    )
    if not job_ids:
        return RapidCryptoCadenceSummary(
            (), 0, None, None, max_gap.total_seconds() / 60, lookback.total_seconds() / 3600
        )

    observed_at: list[datetime] = []
    cycle_counts: list[int] = []
    gaps: list[float] = []
    with connect_database(path) as connection:
        for job_id in job_ids:
            rows = connection.execute(
                """
                SELECT started_at
                FROM ingestion_runs
                WHERE plan_name = ? AND job_id = ? AND status IN (?, ?)
                  AND json_extract(record_json, '$.observation_origin') = 'scheduled'
                  AND started_at >= ? AND started_at <= ?
                ORDER BY started_at ASC, run_id ASC
                """,
                (
                    plan.name,
                    job_id,
                    "success",
                    "degraded",
                    (as_of - lookback).isoformat(),
                    as_of.isoformat(),
                ),
            ).fetchall()
            timestamps = [parse_datetime(str(row[0])) for row in rows]
            cycle_counts.append(len(timestamps))
            if timestamps:
                observed_at.append(timestamps[-1])
                gaps.append(
                    (timestamps[0] - (as_of - lookback)).total_seconds() / 60
                )
            gaps.extend(
                (later - earlier).total_seconds() / 60
                for earlier, later in zip(timestamps, timestamps[1:])
            )
            if timestamps and timestamps[-1] <= as_of:
                gaps.append((as_of - timestamps[-1]).total_seconds() / 60)
    return RapidCryptoCadenceSummary(
        job_ids,
        min(cycle_counts, default=0),
        min(observed_at) if len(observed_at) == len(job_ids) else None,
        max(gaps, default=None),
        max_gap.total_seconds() / 60,
        lookback.total_seconds() / 3600,
    )


def _fast_prediction_cadence(
    path: str | Path,
    plan: ShadowIngestionPlan,
    *,
    as_of: datetime,
    max_gap: timedelta = timedelta(minutes=30),
    lookback: timedelta = timedelta(hours=24),
) -> FastPredictionCadenceSummary:
    """Report gaps for the pre-registered public fast-settlement market page.

    This is operational telemetry only: it cannot create a forecast, restore a
    skipped page, or alter the immutable candidate or evidence rules. Both the
    interval from the rolling-window boundary to the first observation and the
    current interval through ``as_of`` are included, so a newly started or stopped
    lane is visible.
    """
    if max_gap <= timedelta(0):
        raise ValueError("fast prediction cadence bound must be positive")
    if lookback <= timedelta(0):
        raise ValueError("fast prediction cadence lookback must be positive")
    as_of = require_aware(as_of, "as_of")
    job_ids = tuple(
        job.job_id
        for job in plan.jobs
        if job.enabled and job.job_id == "kalshi-fast-settling-markets"
    )
    if not job_ids:
        return FastPredictionCadenceSummary(
            (), 0, None, None, max_gap.total_seconds() / 60, lookback.total_seconds() / 3600
        )

    observed_at: list[datetime] = []
    cycle_counts: list[int] = []
    gaps: list[float] = []
    with connect_database(path) as connection:
        for job_id in job_ids:
            rows = connection.execute(
                """
                SELECT started_at
                FROM ingestion_runs
                WHERE plan_name = ? AND job_id = ? AND status IN (?, ?)
                  AND json_extract(record_json, '$.observation_origin') = 'scheduled'
                  AND started_at >= ? AND started_at <= ?
                ORDER BY started_at ASC, run_id ASC
                """,
                (
                    plan.name,
                    job_id,
                    "success",
                    "degraded",
                    (as_of - lookback).isoformat(),
                    as_of.isoformat(),
                ),
            ).fetchall()
            timestamps = [parse_datetime(str(row[0])) for row in rows]
            cycle_counts.append(len(timestamps))
            if timestamps:
                observed_at.append(timestamps[-1])
                gaps.append(
                    (timestamps[0] - (as_of - lookback)).total_seconds() / 60
                )
            gaps.extend(
                (later - earlier).total_seconds() / 60
                for earlier, later in zip(timestamps, timestamps[1:])
            )
            if timestamps and timestamps[-1] <= as_of:
                gaps.append((as_of - timestamps[-1]).total_seconds() / 60)
    return FastPredictionCadenceSummary(
        job_ids,
        min(cycle_counts, default=0),
        min(observed_at) if len(observed_at) == len(job_ids) else None,
        max(gaps, default=None),
        max_gap.total_seconds() / 60,
        lookback.total_seconds() / 3600,
    )


def _waiting_credentials_message(waiting: Sequence[JobHealth]) -> str:
    """Describe optional read-only feeds without conflating their activation needs."""
    by_venue: dict[str, int] = {}
    for job in waiting:
        venue = job.venue
        by_venue[venue] = by_venue.get(venue, 0) + 1
    descriptions: list[str] = []
    for venue, count in sorted(by_venue.items()):
        if venue == "alpaca":
            descriptions.append(f"{count} Alpaca stock/options job(s)")
        elif venue == "solana":
            descriptions.append(
                f"{count} Solana safety-observation job(s) "
                "(memecoin research remains blocked-unverified)"
            )
        else:
            descriptions.append(f"{count} {venue} job(s)")
    return (
        f"{len(waiting)} read-only collection job(s) await activation: "
        f"{'; '.join(descriptions)}"
    )


def _scorecard_status(alerts: tuple[OperationalAlert, ...]) -> ScorecardStatus:
    severities = {alert.severity for alert in alerts}
    if AlertSeverity.CRITICAL in severities:
        return ScorecardStatus.CRITICAL
    if AlertSeverity.ACTION in severities:
        return ScorecardStatus.CANDIDATE
    if AlertSeverity.WARNING in severities:
        return ScorecardStatus.ATTENTION
    return ScorecardStatus.COLLECTING


def _outcome_queue(
    store: PointInTimeStore,
    forecasts: tuple[Forecast, ...],
    scores: tuple[ForecastScore, ...],
    as_of: datetime,
) -> OutcomeQueueSummary:
    scored_ids = {score.forecast_id for score in scores}
    unscored = [
        forecast for forecast in forecasts if forecast.forecast_id not in scored_ids
    ]
    targets = [(forecast, forecast_label_deadline(forecast)) for forecast in unscored]
    future = [target for _, target in targets if target is not None and target > as_of]
    due = [target for _, target in targets if target is not None and target <= as_of]
    quarantined = sum(target is None for _, target in targets)
    return OutcomeQueueSummary(
        len(unscored),
        len(future),
        len(due),
        quarantined,
        _policy_inconsistent_fast_labels(store, unscored, as_of),
        min(future) if future else None,
        min(due) if due else None,
    )


def _policy_inconsistent_fast_labels(
    store: PointInTimeStore,
    forecasts: Sequence[Forecast],
    as_of: datetime,
) -> int:
    """Count early labels that the active fast-lane policy excludes.

    Kalshi documents that an earlier close is allowed only when
    ``can_close_early`` is true.  These observations remain in the immutable
    store for audit, but must stay outside the prospective score set when a
    forecast recorded the opposite policy.
    """
    excluded = 0
    for forecast in forecasts:
        if forecast.specialist_id not in {
            FastPredictionSettlementV7Specialist.agent_id,
            FastPredictionSettlementV8Specialist.agent_id,
        }:
            continue
        expected_event_ticker = forecast.values.get("event_ticker")
        target_time = prediction_forecast_target_time(forecast)
        if (
            not isinstance(expected_event_ticker, str)
            or not expected_event_ticker
            or target_time is None
        ):
            continue
        settlements = store.events_available_at(
            as_of,
            instrument_id=forecast.instrument_id,
            event_type=MarketEventType.SETTLEMENT,
        )
        if any(
            _early_fast_label_is_excluded(forecast, event, target_time, expected_event_ticker)
            for event in settlements
        ):
            excluded += 1
    return excluded


def _early_fast_label_is_excluded(
    forecast: Forecast,
    event: MarketEvent,
    target_time: datetime,
    expected_event_ticker: str,
) -> bool:
    if (
        event.available_at <= forecast.generated_at
        or not forecast.generated_at <= event.event_time < target_time
        or str(event.payload.get("result", "")).lower() not in {"yes", "no"}
        or prediction_settlement_event_ticker(event) != expected_event_ticker
    ):
        return False
    if forecast.specialist_id == FastPredictionSettlementV7Specialist.agent_id:
        return forecast.values.get("can_close_early") is not True
    if forecast.values.get("can_close_early") is not True:
        return True
    raw_market = event.payload.get("raw_market")
    if not isinstance(raw_market, Mapping):
        return True
    close_value = raw_market.get("close_time")
    if not isinstance(close_value, str) or not close_value:
        return True
    try:
        close_time = parse_datetime(close_value)
    except (TypeError, ValueError):
        return True
    return not (
        forecast.generated_at < close_time <= event.event_time
        and close_time < target_time
    )


def _prediction_outcome_polling(
    path: str | Path,
    plan: ShadowIngestionPlan,
    *,
    as_of: datetime,
) -> PredictionOutcomePollingSummary:
    """Read the newest bounded outcome-poll receipt without inferring a label."""
    job_ids = tuple(
        job.job_id
        for job in plan.jobs
        if job.venue == "kalshi" and job.dataset == "forecast_outcomes" and job.enabled
    )
    if not job_ids:
        return PredictionOutcomePollingSummary(job_ids, None, None, None, None)
    placeholders = ", ".join("?" for _ in job_ids)
    with connect_database(path) as connection:
        row = connection.execute(
            f"""
            SELECT finished_at, record_json
            FROM ingestion_runs
            WHERE plan_name = ?
              AND job_id IN ({placeholders})
              AND status IN ('success', 'degraded')
              AND finished_at <= ?
            ORDER BY finished_at DESC, run_id DESC
            LIMIT 1
            """,
            (plan.name, *job_ids, as_of.isoformat()),
        ).fetchone()
    if row is None:
        return PredictionOutcomePollingSummary(job_ids, None, None, None, None)
    payload = json.loads(row[1])
    requested = payload.get("requested_instruments")
    returned = payload.get("instruments_seen")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 0
        or isinstance(returned, bool)
        or not isinstance(returned, int)
        or returned < 0
    ):
        return PredictionOutcomePollingSummary(job_ids, None, None, None, parse_datetime(row[0]))
    return PredictionOutcomePollingSummary(
        job_ids,
        requested,
        returned,
        max(0, requested - returned),
        parse_datetime(row[0]),
    )


def _strategy_outcome_queues(
    forecasts: tuple[Forecast, ...],
    scores: tuple[ForecastScore, ...],
    as_of: datetime,
) -> tuple[StrategyOutcomeQueue, ...]:
    scored_ids = {score.forecast_id for score in scores}
    grouped: dict[tuple[str, str], list[datetime]] = {}
    for forecast in forecasts:
        if forecast.forecast_id in scored_ids:
            continue
        target = forecast_label_deadline(forecast)
        if target is None:
            continue
        key = (forecast.specialist_id, forecast.kind.value)
        grouped.setdefault(key, []).append(target)
    summaries: list[StrategyOutcomeQueue] = []
    for (specialist_id, kind), targets in sorted(grouped.items()):
        future = [target for target in targets if target > as_of]
        due = [target for target in targets if target <= as_of]
        summaries.append(
            StrategyOutcomeQueue(
                specialist_id,
                kind,
                len(targets),
                len(future),
                len(due),
                min(future) if future else None,
                min(due) if due else None,
            )
        )
    return tuple(summaries)


def _prediction_calibration_readiness(
    path: str | Path,
    as_of: datetime,
    *,
    probability_bucket_radius: float = 0.10,
    required_bucket_events: int = 5,
) -> PredictionCalibrationReadiness:
    with connect_database(path) as connection:
        rows = connection.execute(
            """
            WITH ranked_settlements AS (
                SELECT
                    instrument_id,
                    event_id,
                    available_at,
                    payload_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY instrument_id
                        ORDER BY available_at, event_id
                    ) AS settlement_rank
                FROM market_events
                WHERE event_type = 'settlement'
                  AND available_at <= ?
                  AND LOWER(json_extract(payload_json, '$.result')) IN ('yes', 'no')
            )
            SELECT
                settlements.instrument_id,
                settlements.available_at,
                settlements.payload_json,
                books.event_id,
                books.event_time,
                books.available_at,
                books.payload_json
            FROM ranked_settlements AS settlements
            JOIN market_events AS books
              ON books.instrument_id = settlements.instrument_id
             AND books.event_type = 'book_snapshot'
             AND books.available_at <= ?
            JOIN instruments
              ON instruments.instrument_id = settlements.instrument_id
            WHERE settlements.settlement_rank = 1
              AND instruments.asset_class = 'prediction'
            ORDER BY settlements.instrument_id, books.available_at, books.event_id
            """,
            (as_of.isoformat(), as_of.isoformat()),
        ).fetchall()
        open_rows = connection.execute(
            """
            WITH ranked_books AS (
                SELECT
                    instrument_id,
                    available_at,
                    payload_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY instrument_id
                        ORDER BY available_at DESC, event_time DESC, event_id DESC
                    ) AS event_rank
                FROM market_events
                WHERE event_type = 'book_snapshot' AND available_at <= ?
            ),
            ranked_rules AS (
                SELECT
                    instrument_id,
                    payload_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY instrument_id
                        ORDER BY available_at DESC, event_time DESC, event_id DESC
                    ) AS event_rank
                FROM market_events
                WHERE event_type = 'contract_rule' AND available_at <= ?
            ),
            settled AS (
                SELECT DISTINCT instrument_id
                FROM market_events
                WHERE event_type = 'settlement'
                  AND available_at <= ?
                  AND LOWER(json_extract(payload_json, '$.result')) IN ('yes', 'no')
            )
            SELECT
                books.instrument_id,
                books.available_at,
                books.payload_json,
                rules.payload_json
            FROM ranked_books AS books
            JOIN ranked_rules AS rules USING (instrument_id)
            JOIN instruments USING (instrument_id)
            LEFT JOIN settled USING (instrument_id)
            WHERE books.event_rank = 1
              AND rules.event_rank = 1
              AND instruments.asset_class = 'prediction'
              AND settled.instrument_id IS NULL
            """,
            (as_of.isoformat(), as_of.isoformat(), as_of.isoformat()),
        ).fetchall()
        forecasted_event_keys = {
            str(event_key)
            for (event_key,) in connection.execute(
                """
                SELECT COALESCE(
                    json_extract(payload_json, '$.values.event_ticker'),
                    json_extract(payload_json, '$.values.outcome_cluster'),
                    json_extract(payload_json, '$.instrument_id')
                )
                FROM audit_records
                WHERE record_type = 'forecast'
                  AND json_extract(payload_json, '$.specialist_id') =
                      'prediction-market-calibration-adjusted-v1'
                """
            ).fetchall()
        }

    candidates_by_event: dict[str, list[tuple[float, datetime]]] = {}
    latest_by_instrument: dict[
        str, tuple[datetime, str, str, float, datetime]
    ] = {}
    for (
        instrument_id,
        settlement_available_at,
        settlement_json,
        book_event_id,
        book_event_time,
        book_available_at,
        book_json,
    ) in rows:
        settlement = json.loads(settlement_json)
        raw_market = settlement.get("raw_market")
        raw_market = raw_market if isinstance(raw_market, dict) else {}
        occurrence_value = settlement.get("occurrence_datetime") or raw_market.get(
            "occurrence_datetime"
        )
        if not isinstance(occurrence_value, str) or not occurrence_value:
            continue
        try:
            occurrence = parse_datetime(occurrence_value)
            event_time = parse_datetime(book_event_time)
            available_at = parse_datetime(book_available_at)
            label_available_at = parse_datetime(settlement_available_at)
        except (TypeError, ValueError):
            continue
        time_to_occurrence = occurrence - available_at
        if (
            event_time > occurrence
            or time_to_occurrence <= timedelta(hours=1)
            or time_to_occurrence > timedelta(hours=8)
        ):
            continue
        executable = prediction_book_payload(json.loads(book_json))
        if executable is None or executable[3] > probability_bucket_radius:
            continue
        event_ticker = settlement.get("event_ticker") or raw_market.get("event_ticker")
        if isinstance(event_ticker, str) and event_ticker:
            event_key = f"event:{event_ticker}"
        else:
            event_key = f"occurrence:{occurrence_value}"
        latest_by_instrument[instrument_id] = (
            available_at,
            book_event_id,
            event_key,
            executable[2],
            label_available_at,
        )

    for _, _, event_key, probability, label_available_at in latest_by_instrument.values():
        candidates_by_event.setdefault(event_key, []).append(
            (probability, label_available_at)
        )

    open_by_event: dict[str, tuple[float, float, str]] = {}
    open_probability_by_event: dict[str, float] = {}
    open_decision_by_event: dict[str, datetime] = {}
    for instrument_id, book_available_at, book_json, rule_json in open_rows:
        try:
            decision_time = parse_datetime(book_available_at)
        except (TypeError, ValueError):
            continue
        if as_of - decision_time > timedelta(minutes=15):
            continue
        rule = json.loads(rule_json)
        occurrence_value = rule.get("occurrence_datetime")
        event_key = rule.get("event_ticker")
        if (
            not isinstance(occurrence_value, str)
            or not occurrence_value
            or not isinstance(event_key, str)
            or not event_key
            or event_key in forecasted_event_keys
        ):
            continue
        try:
            occurrence = parse_datetime(occurrence_value)
        except (TypeError, ValueError):
            continue
        time_to_occurrence = occurrence - decision_time
        if (
            time_to_occurrence <= timedelta(hours=1)
            or time_to_occurrence > timedelta(hours=8)
        ):
            continue
        executable = prediction_book_payload(json.loads(book_json))
        if executable is None or executable[3] > probability_bucket_radius:
            continue
        candidate = (executable[3], -decision_time.timestamp(), instrument_id)
        existing = open_by_event.get(event_key)
        if existing is None or candidate < existing:
            open_by_event[event_key] = candidate
            open_probability_by_event[event_key] = executable[2]
            open_decision_by_event[event_key] = decision_time

    strongest_bucket = max(
        (
            sum(
                any(
                    label_available_at <= open_decision_by_event[target_event]
                    and abs(probability - target_probability)
                    <= probability_bucket_radius + 1e-12
                    for probability, label_available_at in event_probabilities
                )
                for event_probabilities in candidates_by_event.values()
            )
            for target_event, target_probability in open_probability_by_event.items()
        ),
        default=0,
    )
    return PredictionCalibrationReadiness(
        len(candidates_by_event),
        len(open_by_event),
        strongest_bucket,
        required_bucket_events,
        probability_bucket_radius,
        strongest_bucket >= required_bucket_events,
    )


def _render_markdown(scorecard: DailyScorecard) -> str:
    totals = scorecard.totals
    lines = [
        "## Daily shadow scorecard",
        "",
        f"**{scorecard.status.value}** — generated at `{scorecard.generated_at.isoformat()}`",
        "",
        (
            f"**{totals.events:,}** events · **{totals.instruments:,}** instruments · "
            f"**{totals.forecasts:,}** forecasts · **{totals.scores:,}** scored outcomes · "
            f"**{totals.ingestion_runs:,}** ingestion runs"
        ),
        "",
        "### Alerts",
        "",
    ]
    for alert in scorecard.alerts:
        lines.append(
            f"- **{alert.severity.value}** `{alert.code}` — {_markdown_escape(alert.message)}"
        )
    lines.extend(
        [
            "",
            "### Market coverage",
            "",
            "| Market | Instruments | Events | Forecasts | Scores |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in scorecard.coverage:
        lines.append(
            f"| {item.asset_class.value} | {item.instruments:,} | {item.events:,} | "
            f"{item.forecasts:,} | {item.scores:,} |"
        )
    lines.extend(
        [
            "",
            "### Pre-registered research lanes",
            "",
            "| Specialist | Market | Proposed | Forecasts | Scores | Latest forecast |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for lane in scorecard.research_lanes:
        latest = (
            f"`{lane.latest_forecast_at.isoformat()}`"
            if lane.latest_forecast_at is not None
            else "—"
        )
        lines.append(
            f"| `{lane.specialist_id}` | {lane.market.value} | "
            f"`{lane.proposed_at.isoformat()}` | {lane.forecasts} | {lane.scores} | {latest} |"
        )
    fast = scorecard.fast_prediction_eligibility
    lines.extend(
        [
            "",
            "### Fast-settlement prediction eligibility",
            "",
            (
                "Current, read-only funnel (not evidence): "
                f"**{fast.paired_markets}** paired rule/book markets → "
                f"**{fast.fresh_book_markets}** fresh books → "
                f"**{fast.active_markets}** active → "
                f"**{fast.documented_close_policy_markets}** documented close policies → "
                f"({fast.early_close_enabled_markets} early-close enabled, "
                f"{fast.early_close_disabled_markets} early-close disabled, "
                f"{fast.missing_close_policy_markets} missing, "
                f"{fast.invalid_close_policy_markets} invalid) → "
                f"**{fast.short_timer_markets}** short-timer → "
                f"**{fast.horizon_markets}** in horizon → "
                f"**{fast.executable_markets}** executable → "
                f"**{fast.unforecasted_event_candidates}** unforecasted events → "
                f"**{fast.selected_events}** selected."
            ),
        ]
    )
    intraday = scorecard.intraday_momentum_eligibility
    lines.extend(
        [
            "",
            "### Fifteen-minute crypto momentum eligibility",
            "",
            (
                "Current, fixed-parameter funnel (not evidence): "
                f"**{intraday.observed_instruments}** observed instruments → "
                f"**{intraday.fresh_instruments}** fresh → "
                f"**{intraday.adequate_lookback_instruments}** with eight completed bars → "
                f"**{intraday.signal_instruments}** current fixed-threshold signal(s)."
            ),
            (
                "Preregistered v2 fixed-assignment funnel (not evidence): "
                f"**{intraday.v2_assigned_instruments}** target-time assigned instrument(s) → "
                f"**{intraday.v2_signal_instruments}** fixed-threshold signal(s)."
            ),
        ]
    )
    cadence = scorecard.rapid_crypto_cadence
    latest_cycle = (
        f"`{cadence.latest_started_at.isoformat()}`"
        if cadence.latest_started_at is not None
        else "—"
    )
    largest_gap = (
        f"{cadence.largest_gap_minutes:.1f} minutes"
        if cadence.largest_gap_minutes is not None
        else "insufficient history"
    )
    lines.extend(
        [
            "",
            "### Rapid crypto collection cadence",
            "",
            (
                f"Fixed fifteen-minute jobs: **{len(cadence.job_ids)}** · "
                f"shared observed cycles: **{cadence.observed_cycles}** · "
                f"latest shared cycle: {latest_cycle} · "
                f"largest observed gap: **{largest_gap}** "
                f"in the trailing {cadence.lookback_hours:.0f} hours "
                f"(bound: {cadence.max_allowed_gap_minutes:.0f} minutes)."
            ),
        ]
    )
    fast_cadence = scorecard.fast_prediction_cadence
    fast_latest_cycle = (
        f"`{fast_cadence.latest_started_at.isoformat()}`"
        if fast_cadence.latest_started_at is not None
        else "—"
    )
    fast_largest_gap = (
        f"{fast_cadence.largest_gap_minutes:.1f} minutes"
        if fast_cadence.largest_gap_minutes is not None
        else "insufficient history"
    )
    lines.extend(
        [
            "",
            "### Fast prediction collection cadence",
            "",
            (
                f"Cursor-resuming public market jobs: **{len(fast_cadence.job_ids)}** · "
                f"observed cycles: **{fast_cadence.observed_cycles}** · "
                f"latest cycle: {fast_latest_cycle} · "
                f"largest collection gap: **{fast_largest_gap}** "
                f"in the trailing {fast_cadence.lookback_hours:.0f} hours "
                f"(bound: {fast_cadence.max_allowed_gap_minutes:.0f} minutes)."
            ),
        ]
    )
    memecoin = scorecard.memecoin_research
    missing_gates = ", ".join(memecoin.missing_hard_gates) or "none"
    lines.extend(
        [
            "",
            "### Memecoin shadow research",
            "",
            (
                f"Recorded public discoveries: **{memecoin.discovered_tokens}** token(s) · "
                f"**{memecoin.latest_profile_observations}** profile(s) · "
                f"**{memecoin.latest_pool_observations}** pool snapshot(s) · "
                f"**{memecoin.latest_authority_observations}** finalized authority observation(s) · "
                f"**{memecoin.transfer_control_observations}** transfer-control parse(s) · "
                f"**{memecoin.holder_concentration_observations}** holder-concentration observation(s) · "
                f"**{memecoin.holder_activity_observations}** aggregate holder-activity observation(s)."
            ),
            (
                "Most recent profile: "
                f"`{memecoin.latest_profile_observed_at.isoformat()}`"
                if memecoin.latest_profile_observed_at is not None
                else "Most recent profile: —"
            ),
            (
                "Most recent pool snapshot: "
                f"`{memecoin.latest_pool_observed_at.isoformat()}`"
                if memecoin.latest_pool_observed_at is not None
                else "Most recent pool snapshot: —"
            ),
            (
                "Most recent finalized authority observation: "
                f"`{memecoin.latest_authority_observed_at.isoformat()}`"
                if memecoin.latest_authority_observed_at is not None
                else "Most recent finalized authority observation: —"
            ),
            (
                "Most recent transfer-control parse: "
                f"`{memecoin.latest_transfer_control_observed_at.isoformat()}`"
                if memecoin.latest_transfer_control_observed_at is not None
                else "Most recent transfer-control parse: —"
            ),
            (
                "Most recent holder-concentration observation: "
                f"`{memecoin.latest_holder_concentration_observed_at.isoformat()}`"
                if memecoin.latest_holder_concentration_observed_at is not None
                else "Most recent holder-concentration observation: —"
            ),
            (
                "Most recent aggregate holder-activity observation: "
                f"`{memecoin.latest_holder_activity_observed_at.isoformat()}`"
                if memecoin.latest_holder_activity_observed_at is not None
                else "Most recent aggregate holder-activity observation: —"
            ),
            "",
            (
                f"Safety filter: **{memecoin.blocked_unverified_tokens}** blocked-unverified · "
                f"**{memecoin.safety_eligible_tokens}** sandbox-eligible. "
                f"Missing hard-gate evidence: {_markdown_escape(missing_gates)}."
            ),
            "",
            "### Outcome queue",
            "",
            (
                f"Unscored: **{scorecard.outcome_queue.unscored:,}** · "
                f"not due: **{scorecard.outcome_queue.not_due:,}** · "
                f"due without outcome: **{scorecard.outcome_queue.due_unmatched:,}** · "
                f"quarantined legacy/invalid: **{scorecard.outcome_queue.quarantined:,}** · "
                "policy-inconsistent or uncorroborated early labels excluded: "
                f"**{scorecard.outcome_queue.policy_inconsistent_early_labels:,}**"
            ),
            "",
            (
                "Next due: "
                f"`{scorecard.outcome_queue.next_due_at.isoformat()}`"
                if scorecard.outcome_queue.next_due_at is not None
                else "Next due: —"
            ),
            "",
            (
                "Oldest due without outcome: "
                f"`{scorecard.outcome_queue.oldest_due_at.isoformat()}`"
                if scorecard.outcome_queue.oldest_due_at is not None
                else "Oldest due without outcome: —"
            ),
            "",
            "### Prediction outcome polling",
            "",
            (
                "Latest bounded poll: "
                f"**{scorecard.prediction_outcome_polling.requested_instruments:,}** requested → "
                f"**{scorecard.prediction_outcome_polling.returned_instruments:,}** returned → "
                f"**{scorecard.prediction_outcome_polling.missing_instruments:,}** missing."
                if scorecard.prediction_outcome_polling.requested_instruments is not None
                and scorecard.prediction_outcome_polling.returned_instruments is not None
                and scorecard.prediction_outcome_polling.missing_instruments is not None
                else "Latest bounded poll: no attested tracked-ticker receipt."
            ),
            (
                "Poll completed: "
                f"`{scorecard.prediction_outcome_polling.finished_at.isoformat()}`"
                if scorecard.prediction_outcome_polling.finished_at is not None
                else "Poll completed: —"
            ),
            "",
            "### Pending outcomes by strategy",
            "",
            "| Specialist | Forecast | Pending | Not due | Due | Next due | Oldest due |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if not scorecard.strategy_outcome_queues:
        lines.append("| none | — | 0 | 0 | 0 | — | — |")
    for queue in scorecard.strategy_outcome_queues:
        next_due = (
            f"`{queue.next_due_at.isoformat()}`"
            if queue.next_due_at is not None
            else "—"
        )
        oldest_due = (
            f"`{queue.oldest_due_at.isoformat()}`"
            if queue.oldest_due_at is not None
            else "—"
        )
        lines.append(
            f"| `{queue.specialist_id}` | {queue.kind} | {queue.pending} | "
            f"{queue.not_due} | {queue.due_unmatched} | {next_due} | "
            f"{oldest_due} |"
        )
    lines.extend(
        [
            "",
            "### Prediction calibration readiness",
            "",
            (
                "Eligible independent resolved events: "
                f"**{scorecard.prediction_calibration.eligible_independent_events}** · "
                "eligible open events: "
                f"**{scorecard.prediction_calibration.eligible_open_events}** · "
                "strongest fixed ten-cent bucket: "
                f"**{scorecard.prediction_calibration.strongest_bucket_events}/"
                f"{scorecard.prediction_calibration.required_bucket_events}** · "
                f"v4 adjustment ready: **{str(scorecard.prediction_calibration.ready).lower()}**"
            ),
            "",
            "### Strategy evidence",
            "",
            "| Specialist | Score | Status | Locked decision | Outcomes | Instruments | Largest share | Improvement | Lower bound | Win rate |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not scorecard.strategies:
        lines.append("| none | — | collecting | — | 0 | 0 | — | — | — | — |")
    for item in scorecard.strategies:
        locked = "—"
        if item.locked_status is not None:
            locked = (
                f"{item.locked_status.value} @ {item.locked_outcomes} "
                f"(monitoring {item.monitoring_status.value})"
            )
        lines.append(
            f"| `{item.specialist_id}` | {item.kind} | **{item.status.value}** | "
            f"{locked} | "
            f"{item.outcomes}/{item.required_outcomes} | {item.unique_instruments} | "
            f"{_percent(item.largest_instrument_share)} | {_number(item.mean_improvement)} | "
            f"{_number(item.lower_confidence_bound)} | {_percent(item.win_rate)} |"
        )
    lines.extend(
        [
            "",
            "### After-cost eligibility",
            "",
            "| Specialist | Score | Status | Trades | Net return | Doubled-cost net | Drawdown |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if not scorecard.economics:
        lines.append("| none | — | blocked | 0 | — | — | — |")
    for item in scorecard.economics:
        lines.append(
            f"| `{item.specialist_id}` | {item.kind} | **{item.status.value}** | "
            f"{item.trades}/{item.required_trades} | {_percent(item.mean_net_return)} | "
            f"{_percent(item.doubled_cost_mean_return)} | {_percent(item.max_drawdown)} |"
        )
    lines.extend(
        [
            "",
            "### Paper execution",
            "",
            (
                f"Control: **{'ready' if scorecard.paper.ready else 'locked'}** · "
                f"enabled={str(scorecard.paper.enabled).lower()} · "
                f"kill switch={str(scorecard.paper.kill_switch_active).lower()} · "
                f"reconciliation records={scorecard.paper.reconciliation_records:,} · "
                f"reason={_markdown_escape(scorecard.paper.reason)}"
            ),
            "",
            "### Ingestion health",
            "",
            "| Job | Feed | Status | Age | Failures | Diagnostics |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for job in scorecard.ingestion.jobs:
        age = "—" if job.age_minutes is None else f"{job.age_minutes:.1f}m"
        status = job.status
        if job.reasons:
            status += f": {'; '.join(job.reasons)}"
        lines.append(
            f"| `{job.job_id}` | {job.venue}/{job.dataset} | "
            f"{_markdown_escape(status)} | {age} | {job.consecutive_failures} | "
            f"{job.diagnostics} |"
        )
    lines.extend(
        [
            "",
            (
                f"Execution audit: {totals.order_intents} intents, {totals.approvals} approvals, "
                f"{totals.execution_receipts} receipts. **Live venue adapters remain disabled.**"
            ),
        ]
    )
    return "\n".join(lines)


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
