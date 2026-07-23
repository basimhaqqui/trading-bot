from __future__ import annotations

from datetime import datetime

from trading_bot.agents.prediction import (
    TIMING_GUARDED_PREDICTION_SPECIALISTS,
    prediction_forecast_target_time,
)
from trading_bot.core.schemas import Forecast, ForecastKind


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
