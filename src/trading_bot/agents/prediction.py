from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.hypotheses import PREDICTION_CALIBRATION_HYPOTHESIS
from trading_bot.agents.market_math import prediction_book, recent_events
from trading_bot.core.schemas import AssetClass, Forecast, ForecastKind, MarketEvent, MarketEventType
from trading_bot.core.serialization import parse_datetime


@dataclass(frozen=True)
class PredictionCalibrationConfig:
    probability_bucket_radius: float = 0.10
    min_calibration_cohort: int = 5
    shrinkage_observations: float = 20.0
    max_book_age: timedelta = timedelta(minutes=15)
    max_book_spread: float = 0.10
    min_forecast_horizon: timedelta = timedelta(hours=1)
    forecast_horizon: timedelta = timedelta(hours=8)

    def __post_init__(self) -> None:
        if not 0 < self.probability_bucket_radius < 0.5:
            raise ValueError("probability bucket radius must be between 0 and 0.5")
        if self.min_calibration_cohort < 1 or self.shrinkage_observations <= 0:
            raise ValueError("calibration cohort and shrinkage must be positive")
        if not 0 < self.max_book_spread < 1:
            raise ValueError("maximum book spread must be between zero and one")
        if not timedelta(0) < self.min_forecast_horizon < self.forecast_horizon:
            raise ValueError("forecast horizon must have positive ordered bounds")


class PredictionMarketCalibrationSpecialist:
    agent_id = "prediction-market-calibration-baseline-v3"
    model_version = "baseline-v3"
    supported_asset_classes = frozenset({AssetClass.PREDICTION})
    hypothesis = PREDICTION_CALIBRATION_HYPOTHESIS

    def __init__(self, config: PredictionCalibrationConfig | None = None) -> None:
        self.config = config or PredictionCalibrationConfig()

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        primary_id = context.instrument.instrument_id
        books = recent_events(
            context.events,
            instrument_id=primary_id,
            event_type=MarketEventType.BOOK_SNAPSHOT,
            decision_time=context.decision_time,
            max_age=self.config.max_book_age,
        )
        rules = recent_events(
            context.events,
            instrument_id=primary_id,
            event_type=MarketEventType.CONTRACT_RULE,
            decision_time=context.decision_time,
        )
        if not books or not rules:
            return None
        executable = prediction_book(books[-1])
        if executable is None:
            return None
        yes_bid, yes_ask, market_probability, spread = executable
        if spread > self.config.max_book_spread:
            return None
        target_time = prediction_occurrence_time(rules[-1])
        time_to_occurrence = (
            target_time - context.decision_time if target_time is not None else None
        )
        if (
            target_time is None
            or time_to_occurrence is None
            or time_to_occurrence <= self.config.min_forecast_horizon
            or time_to_occurrence > self.config.forecast_horizon
        ):
            return None
        event_ticker = rules[-1].payload.get("event_ticker")
        if not isinstance(event_ticker, str) or not event_ticker:
            return None
        cohort = self._calibration_cohort(context, market_probability)
        if len(cohort) >= self.config.min_calibration_cohort:
            empirical = sum(outcome for _, _, outcome in cohort) / len(cohort)
            weight = len(cohort) / (len(cohort) + self.config.shrinkage_observations)
            probability = market_probability * (1 - weight) + empirical * weight
            state = "cohort_adjusted"
        else:
            empirical = market_probability
            weight = 0.0
            probability = market_probability
            state = "executable_market_prior"
        confidence = min(0.75, 0.35 + len(cohort) * 0.01)
        confidence *= max(0.2, 1.0 - min(0.8, spread * 5))
        evidence = [rules[-1].event_id, books[-1].event_id]
        for historical_book, settlement, _ in cohort:
            evidence.extend((historical_book.event_id, settlement.event_id))
        values: dict[str, float | str | bool] = {
            "probability": probability,
            "market_probability": market_probability,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "spread": spread,
            "calibration_cohort_size": float(len(cohort)),
            "cohort_empirical_frequency": empirical,
            "calibration_weight": weight,
            "state": state,
            "event_ticker": event_ticker,
            "outcome_cluster": event_ticker,
            "target_time": target_time.isoformat(),
        }
        return Forecast(
            forecast_id=str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent_id}:{primary_id}:{context.decision_time}")
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=primary_id,
            kind=ForecastKind.BINARY_PROBABILITY,
            generated_at=context.decision_time,
            valid_until=target_time,
            values=values,
            confidence=confidence,
            uncertainty={
                "market_spread": spread,
                "calibration_sample_size": float(len(cohort)),
            },
            evidence_event_ids=tuple(dict.fromkeys(evidence)),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )

    def _calibration_cohort(
        self, context: ReplayContext, target_probability: float
    ) -> list[tuple[MarketEvent, MarketEvent, float]]:
        books_by_instrument: dict[str, list[MarketEvent]] = defaultdict(list)
        settlements_by_instrument: dict[str, MarketEvent] = {}
        primary_id = context.instrument.instrument_id
        related_ids = {item.instrument_id for item in context.related_instruments}
        for event in context.events:
            if event.instrument_id not in related_ids or event.instrument_id == primary_id:
                continue
            if event.event_type is MarketEventType.BOOK_SNAPSHOT:
                books_by_instrument[event.instrument_id].append(event)
            elif event.event_type is MarketEventType.SETTLEMENT:
                existing = settlements_by_instrument.get(event.instrument_id)
                if existing is None or (event.available_at, event.event_id) < (
                    existing.available_at,
                    existing.event_id,
                ):
                    settlements_by_instrument[event.instrument_id] = event
        cohort_by_event: dict[
            str, tuple[MarketEvent, MarketEvent, float, float]
        ] = {}
        for settlement in settlements_by_instrument.values():
            result = str(settlement.payload.get("result", "")).lower()
            if result not in {"yes", "no"}:
                continue
            eligible = [
                book
                for book in books_by_instrument[settlement.instrument_id]
                if book.available_at <= settlement.event_time
            ]
            if not eligible:
                continue
            eligible.sort(key=lambda event: (event.available_at, event.event_id))
            historical_book = eligible[-1]
            executable = prediction_book(historical_book)
            if executable is None:
                continue
            historical_probability = executable[2]
            if abs(historical_probability - target_probability) <= self.config.probability_bucket_radius:
                event_key = _settlement_event_key(settlement)
                candidate = (
                    historical_book,
                    settlement,
                    1.0 if result == "yes" else 0.0,
                    abs(historical_probability - target_probability),
                )
                existing = cohort_by_event.get(event_key)
                if existing is None or (
                    candidate[3],
                    candidate[0].available_at,
                    candidate[0].event_id,
                ) < (
                    existing[3],
                    existing[0].available_at,
                    existing[0].event_id,
                ):
                    cohort_by_event[event_key] = candidate
        return [
            (book, settlement, outcome)
            for book, settlement, outcome, _ in (
                cohort_by_event[key] for key in sorted(cohort_by_event)
            )
        ]


def _settlement_event_key(settlement: MarketEvent) -> str:
    occurrence = settlement.payload.get("occurrence_datetime")
    event_ticker = settlement.payload.get("event_ticker")
    if not isinstance(occurrence, str) or not occurrence:
        raw_market = settlement.payload.get("raw_market")
        if isinstance(raw_market, dict):
            occurrence = raw_market.get("occurrence_datetime")
            if not isinstance(event_ticker, str) or not event_ticker:
                event_ticker = raw_market.get("event_ticker")
    if isinstance(event_ticker, str) and event_ticker:
        return f"event:{event_ticker}"
    if isinstance(occurrence, str) and occurrence:
        return f"occurrence:{occurrence}"
    return f"instrument:{settlement.instrument_id}"


def prediction_occurrence_time(rule: MarketEvent) -> datetime | None:
    value = rule.payload.get("occurrence_datetime")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def prediction_forecast_target_time(forecast: Forecast) -> datetime | None:
    value = forecast.values.get("target_time")
    if not isinstance(value, str) or not value:
        value = forecast.values.get("outcome_cluster")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


TIMING_GUARDED_PREDICTION_SPECIALISTS = frozenset(
    {
        "prediction-market-calibration-baseline-v2",
        PredictionMarketCalibrationSpecialist.agent_id,
    }
)
