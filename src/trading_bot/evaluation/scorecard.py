from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Mapping

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
from trading_bot.evaluation.outcomes import forecast_outcome_target_time
from trading_bot.evaluation.reporting import (
    EdgeStatus,
    EvaluationGateConfig,
    build_walk_forward_report,
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
class StrategySummary:
    specialist_id: str
    kind: str
    status: EdgeStatus
    forecasts: int
    outcomes: int
    required_outcomes: int
    mean_improvement: float | None
    lower_confidence_bound: float | None
    win_rate: float | None
    reasons: tuple[str, ...]


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
    strategies: tuple[StrategySummary, ...]
    economics: tuple[EconomicSummary, ...]
    paper: PaperOperationsSummary
    outcome_queue: OutcomeQueueSummary
    prediction_calibration: PredictionCalibrationReadiness
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
    evaluation = build_walk_forward_report(
        forecasts,
        scores,
        EvaluationGateConfig(min_independent_outcomes=min_outcomes),
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
            _finite(item.mean_improvement),
            _finite(item.lower_confidence_bound),
            _finite(item.win_rate),
            item.reasons,
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
    prediction_calibration = _prediction_calibration_readiness(path, as_of)
    alerts = _build_alerts(
        health, strategies, economics, totals, paper, outcome_queue
    )
    return DailyScorecard(
        as_of,
        _scorecard_status(alerts),
        totals,
        coverage,
        strategies,
        economics,
        paper,
        outcome_queue,
        prediction_calibration,
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
    ]
    for alert in scorecard.alerts:
        lines.append(f"{alert.severity.value}: {alert.code}: {alert.message}")
    for strategy in scorecard.strategies:
        lines.append(
            f"{strategy.specialist_id}/{strategy.kind}: {strategy.status.value} "
            f"outcomes={strategy.outcomes}/{strategy.required_outcomes}"
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

    candidates_by_event: dict[str, list[float]] = {}
    latest_by_instrument: dict[str, tuple[datetime, str, str, float]] = {}
    for (
        instrument_id,
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
        )

    for _, _, event_key, probability in latest_by_instrument.values():
        candidates_by_event.setdefault(event_key, []).append(probability)

    open_by_event: dict[str, tuple[float, float, str]] = {}
    open_probability_by_event: dict[str, float] = {}
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

    target_probabilities = list(open_probability_by_event.values())
    strongest_bucket = max(
        (
            sum(
                any(
                    abs(probability - target) <= probability_bucket_radius + 1e-12
                    for probability in event_probabilities
                )
                for event_probabilities in candidates_by_event.values()
            )
            for target in target_probabilities
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
            "| Specialist | Score | Status | Outcomes | Improvement | Lower bound | Win rate |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if not scorecard.strategies:
        lines.append("| none | — | collecting | 0 | — | — | — |")
    for item in scorecard.strategies:
        lines.append(
            f"| `{item.specialist_id}` | {item.kind} | **{item.status.value}** | "
            f"{item.outcomes}/{item.required_outcomes} | {_number(item.mean_improvement)} | "
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
