from __future__ import annotations

from datetime import datetime

from trading_bot.agents.prediction import (
    TIMING_GUARDED_PREDICTION_SPECIALISTS,
    fast_prediction_settlement_deadline,
    prediction_forecast_target_time,
)
from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.core.serialization import require_aware


def forecast_outcome_target_time(forecast: Forecast) -> datetime | None:
    if forecast.kind is ForecastKind.BINARY_PROBABILITY:
        if forecast.specialist_id not in TIMING_GUARDED_PREDICTION_SPECIALISTS:
            return None
        target_time = prediction_forecast_target_time(forecast)
    else:
        target_time = forecast.valid_until
    if target_time is None or target_time <= forecast.generated_at:
        return None
    return target_time


def forecast_label_deadline(forecast: Forecast) -> datetime | None:
    """Return when an unscored forecast can honestly be called overdue.

    Fast Kalshi forecasts begin outcome polling at their expected expiration,
    but remain eligible for a public finalization through their immutable
    settlement deadline.  Keep that polling target separate from the point at
    which a missing label becomes operationally overdue.
    """
    target_time = forecast_outcome_target_time(forecast)
    if target_time is None:
        return None
    settlement_deadline = fast_prediction_settlement_deadline(forecast)
    if settlement_deadline is None:
        return target_time
    if settlement_deadline < target_time:
        return None
    return settlement_deadline


def evaluation_outcome_target_time(
    forecast: Forecast, recorded_target_time: datetime
) -> datetime:
    recorded_target_time = require_aware(
        recorded_target_time, "recorded_target_time"
    )
    if forecast.kind is ForecastKind.BINARY_PROBABILITY:
        forecast_target = forecast_outcome_target_time(forecast)
        if forecast_target is not None:
            return forecast_target
    return recorded_target_time
