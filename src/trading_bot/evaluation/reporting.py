from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.evaluation.scoring import ForecastScore, ScoreKind


class EdgeStatus(StrEnum):
    COLLECTING = "collecting"
    REJECTED = "rejected"
    CANDIDATE = "candidate"


class CohortDimension(StrEnum):
    FORECAST_HORIZON = "forecast_horizon"
    OUTCOME_HORIZON = "outcome_horizon"


@dataclass(frozen=True)
class EvaluationGateConfig:
    min_independent_outcomes: int = 30
    familywise_alpha: float = 0.05
    minimum_win_rate: float = 0.50

    def __post_init__(self) -> None:
        if self.min_independent_outcomes < 2:
            raise ValueError("minimum outcomes must be at least two")
        if not 0 < self.familywise_alpha < 0.5:
            raise ValueError("familywise alpha must be between zero and 0.5")
        if not 0 <= self.minimum_win_rate <= 1:
            raise ValueError("minimum win rate must be between zero and one")


@dataclass(frozen=True)
class SpecialistEvaluation:
    specialist_id: str
    kind: ScoreKind
    status: EdgeStatus
    forecasts: int
    raw_scores: int
    independent_outcomes: int
    unique_instruments: int
    mean_loss: float | None
    mean_benchmark_loss: float | None
    loss_ratio: float | None
    mean_improvement: float | None
    median_improvement: float | None
    lower_confidence_bound: float | None
    win_rate: float | None
    delayed_control_improvement: float | None
    shuffled_control_improvement: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CohortEvaluation:
    dimension: CohortDimension
    label: str
    evaluation: SpecialistEvaluation


@dataclass(frozen=True)
class WalkForwardReport:
    groups: tuple[SpecialistEvaluation, ...]
    cohorts: tuple[CohortEvaluation, ...]
    familywise_alpha: float
    confidence_tests: int


@dataclass(frozen=True)
class _Observation:
    forecast: Forecast
    score: ForecastScore


_HORIZON_BUCKETS = (
    (60 * 60, "<=1h"),
    (8 * 60 * 60, "1h-8h"),
    (24 * 60 * 60, "8h-1d"),
    (7 * 24 * 60 * 60, "1d-7d"),
    (30 * 24 * 60 * 60, "7d-30d"),
    (math.inf, ">30d"),
)
_HORIZON_BUCKET_ORDER = {
    label: index for index, (_, label) in enumerate(_HORIZON_BUCKETS)
}


def build_walk_forward_report(
    forecasts: tuple[Forecast, ...],
    scores: tuple[ForecastScore, ...],
    config: EvaluationGateConfig | None = None,
) -> WalkForwardReport:
    config = config or EvaluationGateConfig()
    forecasts_by_id = {item.forecast_id: item for item in forecasts}
    forecast_groups: dict[tuple[str, ScoreKind], list[Forecast]] = {}
    observations: dict[tuple[str, ScoreKind], list[_Observation]] = {}
    cohort_forecasts: dict[
        tuple[str, ScoreKind, CohortDimension, str], list[Forecast]
    ] = {}
    cohort_observations: dict[
        tuple[str, ScoreKind, CohortDimension, str], list[_Observation]
    ] = {}

    for forecast in forecasts:
        score_kind = _score_kind(forecast)
        if score_kind is not None:
            specialist_key = (forecast.specialist_id, score_kind)
            forecast_groups.setdefault(specialist_key, []).append(forecast)
            cohort_key = (
                *specialist_key,
                CohortDimension.FORECAST_HORIZON,
                _horizon_bucket(forecast.valid_until, forecast.generated_at),
            )
            cohort_forecasts.setdefault(cohort_key, []).append(forecast)
    for score in scores:
        forecast = forecasts_by_id.get(score.forecast_id)
        if forecast is None or forecast.specialist_id != score.specialist_id:
            continue
        key = (score.specialist_id, score.kind)
        forecast_groups.setdefault(key, []).append(forecast)
        observation = _Observation(forecast, score)
        observations.setdefault(key, []).append(observation)
        for dimension, label in (
            (
                CohortDimension.FORECAST_HORIZON,
                _horizon_bucket(forecast.valid_until, forecast.generated_at),
            ),
            (
                CohortDimension.OUTCOME_HORIZON,
                _horizon_bucket(score.target_time, forecast.generated_at),
            ),
        ):
            cohort_key = (*key, dimension, label)
            cohort_forecasts.setdefault(cohort_key, []).append(forecast)
            cohort_observations.setdefault(cohort_key, []).append(observation)

    keys = sorted(set(forecast_groups) | set(observations))
    cohort_keys = sorted(
        set(cohort_forecasts) | set(cohort_observations),
        key=lambda item: (
            item[0],
            item[1].value,
            item[2].value,
            _HORIZON_BUCKET_ORDER[item[3]],
        ),
    )
    confidence_tests = max(
        1,
        len([key for key in keys if observations.get(key)])
        + len([key for key in cohort_keys if cohort_observations.get(key)]),
    )
    critical_value = statistics.NormalDist().inv_cdf(
        1 - config.familywise_alpha / confidence_tests
    )
    cohort_results = tuple(
        CohortEvaluation(
            key[2],
            key[3],
            _evaluate_group(
                (key[0], key[1]),
                cohort_forecasts.get(key, []),
                cohort_observations.get(key, []),
                config,
                critical_value,
            ),
        )
        for key in cohort_keys
    )
    raw_groups = tuple(
        _evaluate_group(
            key,
            forecast_groups.get(key, []),
            observations.get(key, []),
            config,
            critical_value,
        )
        for key in keys
    )
    groups = tuple(
        _apply_cohort_gate(group, cohort_results, config) for group in raw_groups
    )
    return WalkForwardReport(
        groups,
        cohort_results,
        config.familywise_alpha,
        confidence_tests,
    )


def _horizon_bucket(target: datetime, origin: datetime) -> str:
    seconds = (target - origin).total_seconds()
    if seconds < 0:
        raise ValueError("horizon target cannot precede its origin")
    for maximum, label in _HORIZON_BUCKETS:
        if seconds <= maximum:
            return label
    raise AssertionError("horizon bucket is not exhaustive")


def _apply_cohort_gate(
    group: SpecialistEvaluation,
    cohorts: tuple[CohortEvaluation, ...],
    config: EvaluationGateConfig,
) -> SpecialistEvaluation:
    if group.independent_outcomes < config.min_independent_outcomes:
        return group
    relevant = [
        item
        for item in cohorts
        if item.evaluation.specialist_id == group.specialist_id
        and item.evaluation.kind is group.kind
        and item.evaluation.raw_scores > 0
    ]
    underpowered = [
        item
        for item in relevant
        if item.evaluation.independent_outcomes < config.min_independent_outcomes
    ]
    failing = [
        item
        for item in relevant
        if item.evaluation.independent_outcomes >= config.min_independent_outcomes
        and item.evaluation.status is not EdgeStatus.CANDIDATE
    ]
    cohort_reasons = tuple(
        f"{item.dimension.value}={item.label} needs "
        f"{config.min_independent_outcomes - item.evaluation.independent_outcomes} "
        "more outcome clusters"
        for item in underpowered
    ) + tuple(
        f"{item.dimension.value}={item.label} fails its edge gate" for item in failing
    )
    if not cohort_reasons:
        return group
    if group.status is EdgeStatus.REJECTED or failing:
        status = EdgeStatus.REJECTED
    else:
        status = EdgeStatus.COLLECTING
    return replace(group, status=status, reasons=group.reasons + cohort_reasons)


def _score_kind(forecast: Forecast) -> ScoreKind | None:
    return {
        ForecastKind.BINARY_PROBABILITY: ScoreKind.BINARY,
        ForecastKind.FUNDING_RATE: ScoreKind.FUNDING,
        ForecastKind.VOLATILITY: ScoreKind.VOLATILITY,
        ForecastKind.RETURN_DISTRIBUTION: ScoreKind.RETURN,
    }.get(forecast.kind)


def _evaluate_group(
    key: tuple[str, ScoreKind],
    forecasts: list[Forecast],
    raw_observations: list[_Observation],
    config: EvaluationGateConfig,
    critical_value: float,
) -> SpecialistEvaluation:
    specialist_id, kind = key
    unique_forecasts = {item.forecast_id: item for item in forecasts}
    observations = _latest_independent_outcomes(raw_observations)
    if not observations:
        return SpecialistEvaluation(
            specialist_id,
            kind,
            EdgeStatus.COLLECTING,
            len(unique_forecasts),
            len(raw_observations),
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (f"needs {config.min_independent_outcomes} outcome clusters",),
        )

    improvements = [item.score.benchmark_loss - item.score.loss for item in observations]
    mean_loss = statistics.fmean(item.score.loss for item in observations)
    mean_benchmark = statistics.fmean(
        item.score.benchmark_loss for item in observations
    )
    mean_improvement = statistics.fmean(improvements)
    median_improvement = statistics.median(improvements)
    win_rate = sum(value > 0 for value in improvements) / len(improvements)
    loss_ratio = mean_loss / mean_benchmark if mean_benchmark > 0 else None
    lower_bound = _lower_confidence_bound(improvements, critical_value)
    delayed = _control_improvement(observations, shift=1, circular=False)
    shuffled = _control_improvement(
        observations,
        shift=max(1, len(observations) // 3),
        circular=True,
    )

    reasons: list[str] = []
    if len(observations) < config.min_independent_outcomes:
        reasons.append(
            f"needs {config.min_independent_outcomes - len(observations)} more outcome clusters"
        )
    if mean_improvement <= 0:
        reasons.append("mean loss does not beat benchmark")
    if lower_bound <= 0:
        reasons.append("family-wise confidence bound is not positive")
    if win_rate < config.minimum_win_rate:
        reasons.append("paired win rate is below the gate")
    if delayed is not None and mean_improvement <= delayed:
        reasons.append("does not beat delayed-prediction control")
    if shuffled is not None and mean_improvement <= shuffled:
        reasons.append("does not beat shuffled-prediction control")

    enough = len(observations) >= config.min_independent_outcomes
    status = EdgeStatus.CANDIDATE if enough and not reasons else (
        EdgeStatus.REJECTED if enough else EdgeStatus.COLLECTING
    )
    return SpecialistEvaluation(
        specialist_id,
        kind,
        status,
        len(unique_forecasts),
        len(raw_observations),
        len(observations),
        len({item.forecast.instrument_id for item in observations}),
        mean_loss,
        mean_benchmark,
        loss_ratio,
        mean_improvement,
        median_improvement,
        lower_bound,
        win_rate,
        delayed,
        shuffled,
        tuple(reasons),
    )


def _latest_independent_outcomes(
    observations: list[_Observation],
) -> list[_Observation]:
    latest: dict[str, _Observation] = {}
    for item in observations:
        key = independent_outcome_key(item.forecast, item.score.target_time)
        existing = latest.get(key)
        if existing is None or (
            item.forecast.generated_at,
            item.forecast.forecast_id,
        ) > (
            existing.forecast.generated_at,
            existing.forecast.forecast_id,
        ):
            latest[key] = item
    return sorted(
        latest.values(),
        key=lambda item: (
            item.score.target_time,
            item.forecast.instrument_id,
            item.forecast.forecast_id,
        ),
    )


def outcome_cluster_id(forecast: Forecast) -> str:
    if forecast.kind is ForecastKind.BINARY_PROBABILITY:
        event_ticker = forecast.values.get("event_ticker")
        if isinstance(event_ticker, str) and event_ticker:
            return f"prediction-event:{event_ticker}"
    cluster = forecast.values.get("outcome_cluster")
    if isinstance(cluster, str) and cluster:
        return f"{forecast.kind.value}:{cluster}"
    if forecast.specialist_id == "crypto-range-breakout-continuation-baseline":
        return f"crypto-market:{forecast.valid_until.isoformat()}"
    return f"instrument:{forecast.instrument_id}"


def independent_outcome_key(forecast: Forecast, target_time: datetime) -> str:
    cluster = forecast.values.get("outcome_cluster")
    if (
        forecast.kind is ForecastKind.BINARY_PROBABILITY
        or (isinstance(cluster, str) and cluster)
        or forecast.specialist_id == "crypto-range-breakout-continuation-baseline"
    ):
        return outcome_cluster_id(forecast)
    return f"{outcome_cluster_id(forecast)}:{target_time.isoformat()}"


def _lower_confidence_bound(values: list[float], critical_value: float) -> float:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return -math.inf
    standard_error = _hac_standard_error(values)
    return mean - critical_value * standard_error


def _hac_standard_error(values: list[float]) -> float:
    count = len(values)
    mean = statistics.fmean(values)
    demeaned = [value - mean for value in values]
    bandwidth = min(
        count - 1,
        max(1, int(4 * (count / 100) ** (2 / 9))),
    )
    long_run_variance = sum(value**2 for value in demeaned) / count
    for lag in range(1, bandwidth + 1):
        covariance = sum(
            demeaned[index] * demeaned[index - lag]
            for index in range(lag, count)
        ) / count
        weight = 1 - lag / (bandwidth + 1)
        long_run_variance += 2 * weight * covariance
    return math.sqrt(max(0.0, long_run_variance) / count)


def _control_improvement(
    observations: list[_Observation], *, shift: int, circular: bool
) -> float | None:
    if len(observations) < 2:
        return None
    shift %= len(observations)
    if shift == 0:
        return None
    improvements: list[float] = []
    start = 0 if circular else shift
    for index in range(start, len(observations)):
        item = observations[index]
        control_prediction = observations[(index - shift) % len(observations)].score.predicted
        control_loss = (control_prediction - item.score.actual) ** 2
        improvements.append(item.score.benchmark_loss - control_loss)
    return statistics.fmean(improvements)
