from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from trading_bot.agents.hypotheses import (
    BASELINE_HYPOTHESES,
    BASELINE_HYPOTHESIS_SPECIALIST_IDS,
)
from trading_bot.agents.market_math import prediction_book_payload
from trading_bot.core.audit import AuditLedger, AuditRecordType
from trading_bot.core.schemas import AssetClass, Forecast
from trading_bot.core.serialization import canonical_json, parse_datetime, require_aware
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.costs import EconomicCostRegistry
from trading_bot.evaluation.economics import (
    EconomicGateConfig,
    EconomicStatus,
    build_economic_report,
)
from trading_bot.evaluation.checkpoint import checkpointed_walk_forward_report
from trading_bot.evaluation.outcomes import forecast_outcome_target_time
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
from trading_bot.ingestion.health import IngestionHealthReport, ingestion_health
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
    latest_profile_observed_at: datetime | None
    latest_pool_observed_at: datetime | None
    latest_authority_observed_at: datetime | None
    latest_transfer_control_observed_at: datetime | None
    latest_holder_concentration_observed_at: datetime | None
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
    next_due_at: datetime | None
    oldest_due_at: datetime | None


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
    strategy_outcome_queues: tuple[StrategyOutcomeQueue, ...]
    prediction_calibration: PredictionCalibrationReadiness
    fast_prediction_eligibility: FastPredictionEligibilitySummary
    intraday_momentum_eligibility: IntradayMomentumEligibilitySummary
    ingestion: IngestionHealthReport
    alerts: tuple[OperationalAlert, ...]


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
    research_lanes = _research_lane_summaries(forecasts, scores)
    memecoin_research = _memecoin_research_summary(path, as_of)
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
    outcome_queue = _outcome_queue(forecasts, scores, as_of)
    strategy_outcome_queues = _strategy_outcome_queues(
        forecasts,
        scores,
        as_of,
    )
    prediction_calibration = _prediction_calibration_readiness(path, as_of)
    shadow_research = ShadowResearchRunner(store, audit)
    fast_prediction_eligibility = shadow_research.fast_prediction_eligibility(as_of=as_of)
    intraday_momentum_eligibility = shadow_research.intraday_momentum_eligibility(
        as_of=as_of
    )
    alerts = _build_alerts(
        health, strategies, economics, totals, paper, outcome_queue
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
        strategy_outcome_queues,
        prediction_calibration,
        fast_prediction_eligibility,
        intraday_momentum_eligibility,
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
            f"quarantined={scorecard.outcome_queue.quarantined}"
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
            f"fixed_close={scorecard.fast_prediction_eligibility.fixed_close_markets} "
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
    with sqlite3.connect(path) as connection:
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
    path: str | Path, as_of: datetime
) -> MemecoinResearchSummary:
    with sqlite3.connect(path) as connection:
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

    by_token: dict[str, list[Mapping[str, object]]] = {}
    profiles = 0
    pools = 0
    authorities = 0
    transfer_controls = 0
    holder_concentrations = 0
    latest_profile_observed_at: datetime | None = None
    latest_pool_observed_at: datetime | None = None
    latest_authority_observed_at: datetime | None = None
    latest_transfer_control_observed_at: datetime | None = None
    latest_holder_concentration_observed_at: datetime | None = None
    for (instrument_id, category), (observed_at, _, payload) in latest.items():
        by_token.setdefault(instrument_id, []).append(payload)
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
        else:
            if payload.get("holder_concentration_observed") is True:
                holder_concentrations += 1
                if (
                    latest_holder_concentration_observed_at is None
                    or observed_at > latest_holder_concentration_observed_at
                ):
                    latest_holder_concentration_observed_at = observed_at
    blocked = 0
    eligible = 0
    missing: set[str] = set()
    hard_gates = {
        "onchain_authorities_observed": "onchain authorities",
        "holder_concentration_observed": "holder concentration",
        "transfer_behavior_observed": "transfer behavior",
        "round_trip_simulation_observed": "round-trip simulation",
    }
    for payloads in by_token.values():
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
        latest_profile_observed_at,
        latest_pool_observed_at,
        latest_authority_observed_at,
        latest_transfer_control_observed_at,
        latest_holder_concentration_observed_at,
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
    return None


def _build_alerts(
    health: IngestionHealthReport,
    strategies: tuple[StrategySummary, ...],
    economics: tuple[EconomicSummary, ...],
    totals: SystemTotals,
    paper: PaperOperationsSummary,
    outcome_queue: OutcomeQueueSummary,
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
    waiting = [job.job_id for job in health.jobs if job.status == "waiting_credentials"]
    if waiting:
        alerts.append(
            OperationalAlert(
                "market_data_credentials_waiting",
                AlertSeverity.WARNING,
                f"{len(waiting)} Alpaca stock/options job(s) will activate after read-only secrets are set",
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
    if not alerts:
        alerts.append(
            OperationalAlert(
                "evidence_collecting",
                AlertSeverity.INFO,
                "Data is healthy and no strategy has cleared the evidence gates yet",
            )
        )
    return tuple(alerts)


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
    forecasts: tuple[Forecast, ...],
    scores: tuple[ForecastScore, ...],
    as_of: datetime,
) -> OutcomeQueueSummary:
    scored_ids = {score.forecast_id for score in scores}
    unscored = [
        forecast for forecast in forecasts if forecast.forecast_id not in scored_ids
    ]
    targets = [
        (forecast, forecast_outcome_target_time(forecast)) for forecast in unscored
    ]
    future = [target for _, target in targets if target is not None and target > as_of]
    due = [target for _, target in targets if target is not None and target <= as_of]
    quarantined = sum(target is None for _, target in targets)
    return OutcomeQueueSummary(
        len(unscored),
        len(future),
        len(due),
        quarantined,
        min(future) if future else None,
        min(due) if due else None,
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
        target = forecast_outcome_target_time(forecast)
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
    with sqlite3.connect(path) as connection:
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
                      'prediction-market-calibration-baseline-v4'
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
                f"**{fast.fixed_close_markets}** fixed-close → "
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
                f"**{memecoin.holder_concentration_observations}** holder-concentration observation(s)."
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
                f"quarantined legacy/invalid: **{scorecard.outcome_queue.quarantined:,}**"
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
