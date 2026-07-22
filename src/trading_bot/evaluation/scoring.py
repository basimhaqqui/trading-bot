from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.core.serialization import require_aware


class ScoreKind(StrEnum):
    BINARY = "binary"
    FUNDING = "funding"
    VOLATILITY = "volatility"
    RETURN = "return"


@dataclass(frozen=True)
class ForecastScore:
    score_id: str
    forecast_id: str
    specialist_id: str
    kind: ScoreKind
    scored_at: datetime
    target_time: datetime
    predicted: float
    actual: float
    benchmark: float
    loss: float
    benchmark_loss: float
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scored_at", require_aware(self.scored_at, "scored_at"))
        object.__setattr__(self, "target_time", require_aware(self.target_time, "target_time"))
        for name in ("predicted", "actual", "benchmark", "loss", "benchmark_loss"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


def score_binary_forecast(
    forecast: Forecast,
    *,
    outcome: bool,
    target_time: datetime,
    scored_at: datetime,
) -> ForecastScore:
    if forecast.kind is not ForecastKind.BINARY_PROBABILITY:
        raise ValueError("forecast is not a binary probability")
    probability = _field(forecast, "probability")
    market_probability = _field(forecast, "market_probability", probability)
    actual = 1.0 if outcome else 0.0
    brier = (probability - actual) ** 2
    market_brier = (market_probability - actual) ** 2
    epsilon = 1e-12
    bounded = min(1 - epsilon, max(epsilon, probability))
    market_bounded = min(1 - epsilon, max(epsilon, market_probability))
    log_loss = -(actual * math.log(bounded) + (1 - actual) * math.log(1 - bounded))
    market_log_loss = -(
        actual * math.log(market_bounded)
        + (1 - actual) * math.log(1 - market_bounded)
    )
    return _score(
        forecast,
        ScoreKind.BINARY,
        target_time,
        scored_at,
        probability,
        actual,
        market_probability,
        brier,
        market_brier,
        {
            "brier_improvement_vs_market": market_brier - brier,
            "log_loss": log_loss,
            "market_log_loss": market_log_loss,
            "log_loss_improvement_vs_market": market_log_loss - log_loss,
        },
    )


def score_funding_forecast(
    forecast: Forecast,
    *,
    actual_rate: float,
    target_time: datetime,
    scored_at: datetime,
) -> ForecastScore:
    if forecast.kind is not ForecastKind.FUNDING_RATE:
        raise ValueError("forecast is not a funding-rate forecast")
    predicted = _field(forecast, "predicted_funding_rate")
    benchmark = _field(forecast, "current_funding_rate")
    return _numeric_score(
        forecast,
        ScoreKind.FUNDING,
        predicted,
        actual_rate,
        benchmark,
        target_time,
        scored_at,
    )


def score_volatility_forecast(
    forecast: Forecast,
    *,
    actual_implied_volatility: float,
    target_time: datetime,
    scored_at: datetime,
) -> ForecastScore:
    if forecast.kind is not ForecastKind.VOLATILITY:
        raise ValueError("forecast is not a volatility forecast")
    predicted = _field(forecast, "expected_implied_volatility")
    benchmark = _field(forecast, "current_implied_volatility")
    return _numeric_score(
        forecast,
        ScoreKind.VOLATILITY,
        predicted,
        actual_implied_volatility,
        benchmark,
        target_time,
        scored_at,
    )


def score_return_forecast(
    forecast: Forecast,
    *,
    actual_return: float,
    target_time: datetime,
    scored_at: datetime,
) -> ForecastScore:
    if forecast.kind is not ForecastKind.RETURN_DISTRIBUTION:
        raise ValueError("forecast is not a return forecast")
    predicted = _field(forecast, "predicted_return")
    benchmark = _field(forecast, "benchmark_return", 0.0)
    return _numeric_score(
        forecast,
        ScoreKind.RETURN,
        predicted,
        actual_return,
        benchmark,
        target_time,
        scored_at,
    )


def _numeric_score(
    forecast: Forecast,
    kind: ScoreKind,
    predicted: float,
    actual: float,
    benchmark: float,
    target_time: datetime,
    scored_at: datetime,
) -> ForecastScore:
    if not math.isfinite(actual):
        raise ValueError("actual value must be finite")
    error = predicted - actual
    benchmark_error = benchmark - actual
    return _score(
        forecast,
        kind,
        target_time,
        scored_at,
        predicted,
        actual,
        benchmark,
        error**2,
        benchmark_error**2,
        {
            "absolute_error": abs(error),
            "benchmark_absolute_error": abs(benchmark_error),
            "squared_error_improvement_vs_benchmark": benchmark_error**2 - error**2,
            "squared_error_improvement_vs_latest": benchmark_error**2 - error**2,
        },
    )


def _score(
    forecast: Forecast,
    kind: ScoreKind,
    target_time: datetime,
    scored_at: datetime,
    predicted: float,
    actual: float,
    benchmark: float,
    loss: float,
    benchmark_loss: float,
    metrics: Mapping[str, float],
) -> ForecastScore:
    target_time = require_aware(target_time, "target_time")
    scored_at = require_aware(scored_at, "scored_at")
    if target_time < forecast.generated_at:
        raise ValueError("target_time cannot precede forecast generation")
    return ForecastScore(
        score_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"score:{forecast.forecast_id}:{target_time}")),
        forecast_id=forecast.forecast_id,
        specialist_id=forecast.specialist_id,
        kind=kind,
        scored_at=scored_at,
        target_time=target_time,
        predicted=predicted,
        actual=actual,
        benchmark=benchmark,
        loss=loss,
        benchmark_loss=benchmark_loss,
        metrics=metrics,
    )


def _field(forecast: Forecast, name: str, default: float | None = None) -> float:
    value = forecast.values.get(name, default)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"forecast is missing finite numeric field {name}")
    return float(value)
