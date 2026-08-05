from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.hypotheses import (
    PREDICTION_CALIBRATION_ADJUSTED_HYPOTHESIS,
    PREDICTION_CALIBRATION_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V5_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V6_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V7_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V8_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V9_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V10_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V11_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V12_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V13_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V14_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V15_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V16_HYPOTHESIS,
)
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
            occurrence_time = prediction_settlement_occurrence_time(settlement)
            if occurrence_time is None:
                continue
            eligible: list[tuple[MarketEvent, tuple[float, float, float, float]]] = []
            for book in books_by_instrument[settlement.instrument_id]:
                time_to_occurrence = occurrence_time - book.available_at
                executable = prediction_book(book)
                if (
                    book.event_time <= occurrence_time
                    and self.config.min_forecast_horizon
                    < time_to_occurrence
                    <= self.config.forecast_horizon
                    and executable is not None
                    and executable[3] <= self.config.max_book_spread
                ):
                    eligible.append((book, executable))
            if not eligible:
                continue
            eligible.sort(key=lambda item: (item[0].available_at, item[0].event_id))
            historical_book, executable = eligible[-1]
            historical_probability = executable[2]
            if abs(historical_probability - target_probability) <= self.config.probability_bucket_radius:
                event_key = prediction_settlement_event_key(settlement)
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


class AdjustedPredictionMarketCalibrationSpecialist(
    PredictionMarketCalibrationSpecialist
):
    agent_id = "prediction-market-calibration-adjusted-v1"
    model_version = "adjusted-v1"
    hypothesis = PREDICTION_CALIBRATION_ADJUSTED_HYPOTHESIS

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        forecast = super().evaluate(context)
        if forecast is None or forecast.values.get("state") != "cohort_adjusted":
            return None
        probability = float(forecast.values["probability"])
        market_probability = float(forecast.values["market_probability"])
        if math.isclose(probability, market_probability, rel_tol=0.0, abs_tol=1e-12):
            return None
        return forecast


@dataclass(frozen=True)
class FastPredictionSettlementConfig:
    max_book_age: timedelta = timedelta(minutes=15)
    max_book_spread: float = 0.10
    min_forecast_horizon: timedelta = timedelta(minutes=20)
    forecast_horizon: timedelta = timedelta(hours=2)
    max_settlement_timer_seconds: int = 15 * 60
    max_finalization_lag: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        if not 0 < self.max_book_spread < 1:
            raise ValueError("maximum book spread must be between zero and one")
        if not timedelta(0) < self.min_forecast_horizon < self.forecast_horizon:
            raise ValueError("forecast horizon must have positive ordered bounds")
        if self.max_settlement_timer_seconds < 1:
            raise ValueError("settlement timer must be positive")
        if self.max_finalization_lag <= timedelta(0):
            raise ValueError("fast finalization lag must be positive")


class FastPredictionSettlementSpecialist:
    """Pre-registered, unadjusted short-horizon Kalshi calibration baseline."""

    agent_id = "prediction-market-fast-settlement-baseline-v2"
    model_version = "baseline-v2"
    supported_asset_classes = frozenset({AssetClass.PREDICTION})
    hypothesis = PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS

    def __init__(self, config: FastPredictionSettlementConfig | None = None) -> None:
        self.config = config or FastPredictionSettlementConfig()

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
        rule = rules[-1]
        if str(rule.payload.get("status", "")).lower() != "active":
            return None
        if rule.payload.get("can_close_early") is not False:
            return None
        timer = _positive_int(rule.payload.get("settlement_timer_seconds"))
        if timer is None or timer > self.config.max_settlement_timer_seconds:
            return None
        event_ticker = rule.payload.get("event_ticker")
        if not isinstance(event_ticker, str) or not event_ticker:
            return None
        target_time = prediction_expected_expiration_time(rule)
        time_to_target = (
            target_time - context.decision_time if target_time is not None else None
        )
        if (
            target_time is None
            or time_to_target is None
            or time_to_target <= self.config.min_forecast_horizon
            or time_to_target > self.config.forecast_horizon
        ):
            return None
        executable = prediction_book(books[-1])
        if executable is None:
            return None
        yes_bid, yes_ask, market_probability, spread = executable
        if spread > self.config.max_book_spread:
            return None
        settlement_deadline = target_time + timedelta(seconds=timer)
        return Forecast(
            forecast_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.agent_id}:{primary_id}:{context.decision_time}",
                )
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=primary_id,
            kind=ForecastKind.BINARY_PROBABILITY,
            generated_at=context.decision_time,
            valid_until=target_time,
            values={
                "probability": market_probability,
                "market_probability": market_probability,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "spread": spread,
                "state": "executable_market_prior",
                "event_ticker": event_ticker,
                "outcome_cluster": event_ticker,
                "target_time": target_time.isoformat(),
                "settlement_deadline": settlement_deadline.isoformat(),
            },
            confidence=max(0.2, 0.45 - min(0.25, spread * 2.5)),
            uncertainty={
                "market_spread": spread,
                "settlement_timer_seconds": float(timer),
                "time_to_expected_expiration_seconds": time_to_target.total_seconds(),
            },
            evidence_event_ids=(rule.event_id, books[-1].event_id),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )

class FastPredictionSettlementV3Specialist(FastPredictionSettlementSpecialist):
    """Prospective fast lane with a preregistered settlement label window."""

    agent_id = "prediction-market-fast-settlement-baseline-v3"
    model_version = "baseline-v3"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS


class FastPredictionSettlementV4Specialist(FastPredictionSettlementV3Specialist):
    """Prospective fast lane that advances the public market-list cursor."""

    agent_id = "prediction-market-fast-settlement-baseline-v4"
    model_version = "baseline-v4"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS


class FastPredictionSettlementV5Specialist:
    """Prospective lane bounded by Kalshi's documented latest-expiration field."""

    agent_id = "prediction-market-fast-settlement-baseline-v5"
    model_version = "baseline-v5"
    supported_asset_classes = frozenset({AssetClass.PREDICTION})
    hypothesis = PREDICTION_FAST_SETTLEMENT_V5_HYPOTHESIS

    def __init__(self, config: FastPredictionSettlementConfig | None = None) -> None:
        self.config = config or FastPredictionSettlementConfig()

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
        rule = rules[-1]
        if str(rule.payload.get("status", "")).lower() != "active":
            return None
        if not isinstance(rule.payload.get("can_close_early"), bool):
            return None
        timer = _positive_int(rule.payload.get("settlement_timer_seconds"))
        if timer is None or timer > self.config.max_settlement_timer_seconds:
            return None
        event_ticker = rule.payload.get("event_ticker")
        if not isinstance(event_ticker, str) or not event_ticker:
            return None
        expected_expiration = prediction_expected_expiration_time(rule)
        latest_expiration = prediction_latest_expiration_time(rule)
        if expected_expiration is None or latest_expiration is None:
            return None
        expected_horizon = expected_expiration - context.decision_time
        latest_horizon = latest_expiration - context.decision_time
        if (
            latest_expiration < expected_expiration
            or not self.config.min_forecast_horizon < expected_horizon
            or expected_horizon > self.config.forecast_horizon
            or latest_horizon > self.config.forecast_horizon
        ):
            return None
        executable = prediction_book(books[-1])
        if executable is None:
            return None
        yes_bid, yes_ask, market_probability, spread = executable
        if spread > self.config.max_book_spread:
            return None
        settlement_deadline = latest_expiration + timedelta(seconds=timer)
        return Forecast(
            forecast_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.agent_id}:{primary_id}:{context.decision_time}",
                )
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=primary_id,
            kind=ForecastKind.BINARY_PROBABILITY,
            generated_at=context.decision_time,
            valid_until=latest_expiration,
            values={
                "probability": market_probability,
                "market_probability": market_probability,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "spread": spread,
                "state": "executable_market_prior",
                "event_ticker": event_ticker,
                "outcome_cluster": event_ticker,
                "can_close_early": rule.payload["can_close_early"],
                "target_time": latest_expiration.isoformat(),
                "expected_expiration_time": expected_expiration.isoformat(),
                "latest_expiration_time": latest_expiration.isoformat(),
                "settlement_deadline": settlement_deadline.isoformat(),
                "venue_lifecycle_source": "https://docs.kalshi.com/getting_started/market_lifecycle",
            },
            confidence=max(0.2, 0.45 - min(0.25, spread * 2.5)),
            uncertainty={
                "market_spread": spread,
                "settlement_timer_seconds": float(timer),
                "time_to_expected_expiration_seconds": expected_horizon.total_seconds(),
                "time_to_latest_expiration_seconds": latest_horizon.total_seconds(),
            },
            evidence_event_ids=(rule.event_id, books[-1].event_id),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )

class FastPredictionSettlementV6Specialist:
    """Prospective fast-finalization lane with a fixed label deadline."""

    agent_id = "prediction-market-fast-settlement-baseline-v6"
    model_version = "baseline-v6"
    supported_asset_classes = frozenset({AssetClass.PREDICTION})
    hypothesis = PREDICTION_FAST_SETTLEMENT_V6_HYPOTHESIS

    def __init__(self, config: FastPredictionSettlementConfig | None = None) -> None:
        self.config = config or FastPredictionSettlementConfig()

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
        rule = rules[-1]
        if str(rule.payload.get("status", "")).lower() != "active":
            return None
        if not isinstance(rule.payload.get("can_close_early"), bool):
            return None
        timer = _positive_int(rule.payload.get("settlement_timer_seconds"))
        if timer is None or timer > self.config.max_settlement_timer_seconds:
            return None
        event_ticker = rule.payload.get("event_ticker")
        if not isinstance(event_ticker, str) or not event_ticker:
            return None
        expected_expiration = prediction_expected_expiration_time(rule)
        latest_expiration = prediction_latest_expiration_time(rule)
        if expected_expiration is None or latest_expiration is None:
            return None
        expected_horizon = expected_expiration - context.decision_time
        if (
            latest_expiration < expected_expiration
            or not self.config.min_forecast_horizon < expected_horizon
            or expected_horizon > self.config.forecast_horizon
        ):
            return None
        executable = prediction_book(books[-1])
        if executable is None:
            return None
        yes_bid, yes_ask, market_probability, spread = executable
        if spread > self.config.max_book_spread:
            return None
        settlement_deadline = (
            expected_expiration
            + timedelta(seconds=timer)
            + self.config.max_finalization_lag
        )
        return Forecast(
            forecast_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.agent_id}:{primary_id}:{context.decision_time}",
                )
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=primary_id,
            kind=ForecastKind.BINARY_PROBABILITY,
            generated_at=context.decision_time,
            valid_until=expected_expiration,
            values={
                "probability": market_probability,
                "market_probability": market_probability,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "spread": spread,
                "state": "executable_market_prior",
                "event_ticker": event_ticker,
                "outcome_cluster": event_ticker,
                "can_close_early": rule.payload["can_close_early"],
                "target_time": expected_expiration.isoformat(),
                "expected_expiration_time": expected_expiration.isoformat(),
                "latest_expiration_time": latest_expiration.isoformat(),
                "settlement_deadline": settlement_deadline.isoformat(),
                "fast_finalization_lag_seconds": self.config.max_finalization_lag.total_seconds(),
                "venue_lifecycle_source": "https://docs.kalshi.com/getting_started/market_lifecycle",
                "venue_settlement_source": "https://docs.kalshi.com/getting_started/market_settlement",
            },
            confidence=max(0.2, 0.45 - min(0.25, spread * 2.5)),
            uncertainty={
                "market_spread": spread,
                "settlement_timer_seconds": float(timer),
                "time_to_expected_expiration_seconds": expected_horizon.total_seconds(),
                "time_to_latest_expiration_seconds": (
                    latest_expiration - context.decision_time
                ).total_seconds(),
                "fast_finalization_lag_seconds": self.config.max_finalization_lag.total_seconds(),
            },
            evidence_event_ids=(rule.event_id, books[-1].event_id),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )


class FastPredictionSettlementV7Specialist(FastPredictionSettlementV6Specialist):
    """Preregistered successor that rejects policy-inconsistent early labels."""

    agent_id = "prediction-market-fast-settlement-baseline-v7"
    model_version = "baseline-v7"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V7_HYPOTHESIS


class FastPredictionSettlementV8Specialist(FastPredictionSettlementV7Specialist):
    """Successor that requires venue-recorded early-close corroboration."""

    agent_id = "prediction-market-fast-settlement-baseline-v8"
    model_version = "baseline-v8"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V8_HYPOTHESIS


class FastPredictionSettlementV9Specialist(FastPredictionSettlementV8Specialist):
    """Successor that requires the early close to precede finalization strictly."""

    agent_id = "prediction-market-fast-settlement-baseline-v9"
    model_version = "baseline-v9"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V9_HYPOTHESIS


class FastPredictionSettlementV10Specialist:
    """Fast lane anchored to the observed trading-close time, not an estimate."""

    agent_id = "prediction-market-fast-settlement-baseline-v10"
    model_version = "baseline-v10"
    supported_asset_classes = frozenset({AssetClass.PREDICTION})
    hypothesis = PREDICTION_FAST_SETTLEMENT_V10_HYPOTHESIS

    def __init__(self, config: FastPredictionSettlementConfig | None = None) -> None:
        self.config = config or FastPredictionSettlementConfig()

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
        rule = rules[-1]
        if not self._rule_is_current(rule, books[-1]):
            return None
        if str(rule.payload.get("status", "")).lower() != "active":
            return None
        if not isinstance(rule.payload.get("can_close_early"), bool):
            return None
        timer = _positive_int(rule.payload.get("settlement_timer_seconds"))
        if timer is None or timer > self.config.max_settlement_timer_seconds:
            return None
        event_ticker = rule.payload.get("event_ticker")
        if not isinstance(event_ticker, str) or not event_ticker:
            return None
        close_time = prediction_close_time(rule)
        expected_expiration = prediction_expected_expiration_time(rule)
        latest_expiration = prediction_latest_expiration_time(rule)
        if close_time is None or expected_expiration is None or latest_expiration is None:
            return None
        close_horizon = close_time - context.decision_time
        if (
            latest_expiration < expected_expiration
            or not self.config.min_forecast_horizon < close_horizon
            or close_horizon > self.config.forecast_horizon
        ):
            return None
        executable = prediction_book(books[-1])
        if executable is None:
            return None
        yes_bid, yes_ask, market_probability, spread = executable
        if spread > self.config.max_book_spread:
            return None
        settlement_deadline = (
            close_time
            + timedelta(seconds=timer)
            + self.config.max_finalization_lag
        )
        return Forecast(
            forecast_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.agent_id}:{primary_id}:{context.decision_time}",
                )
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=primary_id,
            kind=ForecastKind.BINARY_PROBABILITY,
            generated_at=context.decision_time,
            valid_until=close_time,
            values={
                "probability": market_probability,
                "market_probability": market_probability,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "spread": spread,
                "state": "executable_market_prior",
                "event_ticker": event_ticker,
                "outcome_cluster": event_ticker,
                "can_close_early": rule.payload["can_close_early"],
                "target_time": close_time.isoformat(),
                "registered_close_time": close_time.isoformat(),
                "expected_expiration_time": expected_expiration.isoformat(),
                "latest_expiration_time": latest_expiration.isoformat(),
                "settlement_deadline": settlement_deadline.isoformat(),
                "fast_finalization_lag_seconds": self.config.max_finalization_lag.total_seconds(),
                "venue_lifecycle_source": "https://docs.kalshi.com/getting_started/market_lifecycle",
                "venue_settlement_source": "https://docs.kalshi.com/getting_started/market_settlement",
            },
            confidence=max(0.2, 0.45 - min(0.25, spread * 2.5)),
            uncertainty={
                "market_spread": spread,
                "settlement_timer_seconds": float(timer),
                "time_to_registered_close_seconds": close_horizon.total_seconds(),
                "time_to_expected_expiration_seconds": (
                    expected_expiration - context.decision_time
                ).total_seconds(),
                "time_to_latest_expiration_seconds": (
                    latest_expiration - context.decision_time
                ).total_seconds(),
                "fast_finalization_lag_seconds": self.config.max_finalization_lag.total_seconds(),
            },
            evidence_event_ids=(rule.event_id, books[-1].event_id),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )

    def _rule_is_current(self, rule: MarketEvent, book: MarketEvent) -> bool:
        """Keep v10 behavior stable; stricter successors override this guard."""
        return True


class FastPredictionSettlementV11Specialist(FastPredictionSettlementV10Specialist):
    """Prospective fast lane requiring a current lifecycle snapshot for each book."""

    agent_id = "prediction-market-fast-settlement-baseline-v11"
    model_version = "baseline-v11"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V11_HYPOTHESIS

    def _rule_is_current(self, rule: MarketEvent, book: MarketEvent) -> bool:
        age = book.available_at - rule.available_at
        return timedelta(0) <= age <= self.config.max_book_age


class FastPredictionSettlementV12Specialist(FastPredictionSettlementV11Specialist):
    """Successor excluding API-provisional markets with unreliable public labels."""

    agent_id = "prediction-market-fast-settlement-baseline-v12"
    model_version = "baseline-v12"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V12_HYPOTHESIS

    def _rule_is_current(self, rule: MarketEvent, book: MarketEvent) -> bool:
        if not super()._rule_is_current(rule, book):
            return False
        # Kalshi documents this as an optional boolean that is present as true
        # for markets that can disappear from the public API.  An omitted flag
        # remains compatible with the documented response; any supplied value
        # other than a real boolean is malformed external data and must not
        # become prospective evidence.
        if "is_provisional" not in rule.payload:
            return True
        return rule.payload["is_provisional"] is False


class FastPredictionSettlementV13Specialist(FastPredictionSettlementV12Specialist):
    """Successor that excludes labels after a documented close-time extension."""

    agent_id = "prediction-market-fast-settlement-baseline-v13"
    model_version = "baseline-v13"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V13_HYPOTHESIS


class FastPredictionSettlementV14Specialist(FastPredictionSettlementV13Specialist):
    """Successor that retains the recorded early-close policy at settlement."""

    agent_id = "prediction-market-fast-settlement-baseline-v14"
    model_version = "baseline-v14"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V14_HYPOTHESIS


class FastPredictionSettlementV15Specialist(FastPredictionSettlementV14Specialist):
    """Successor that requires explicit non-provisional binary contracts."""

    agent_id = "prediction-market-fast-settlement-baseline-v15"
    model_version = "baseline-v15"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V15_HYPOTHESIS

    def _rule_is_current(self, rule: MarketEvent, book: MarketEvent) -> bool:
        # ``market_type`` is a documented field on the public market response.
        # The registered lane is binary-only, so an omitted, scalar, or otherwise
        # malformed value is not interchangeable with a binary contract.
        return (
            super()._rule_is_current(rule, book)
            and rule.payload.get("is_provisional") is False
            and rule.payload.get("market_type") == "binary"
        )


class FastPredictionSettlementV16Specialist(FastPredictionSettlementV15Specialist):
    """Fresh close-window successor with the same strict contract eligibility."""

    agent_id = "prediction-market-fast-settlement-baseline-v16"
    model_version = "baseline-v16"
    hypothesis = PREDICTION_FAST_SETTLEMENT_V16_HYPOTHESIS


def prediction_settlement_event_key(settlement: MarketEvent) -> str:
    occurrence = settlement.payload.get("occurrence_datetime")
    event_ticker = prediction_settlement_event_ticker(settlement)
    if not isinstance(occurrence, str) or not occurrence:
        raw_market = settlement.payload.get("raw_market")
        if isinstance(raw_market, dict):
            occurrence = raw_market.get("occurrence_datetime")
    if isinstance(event_ticker, str) and event_ticker:
        return f"event:{event_ticker}"
    if isinstance(occurrence, str) and occurrence:
        return f"occurrence:{occurrence}"
    return f"instrument:{settlement.instrument_id}"


def prediction_settlement_event_ticker(settlement: MarketEvent) -> str | None:
    """Return the venue event identity attached to a finalized settlement, if present."""
    event_ticker = settlement.payload.get("event_ticker")
    if not isinstance(event_ticker, str) or not event_ticker:
        raw_market = settlement.payload.get("raw_market")
        if isinstance(raw_market, dict):
            event_ticker = raw_market.get("event_ticker")
    return event_ticker if isinstance(event_ticker, str) and event_ticker else None


def prediction_settlement_occurrence_time(
    settlement: MarketEvent,
) -> datetime | None:
    value = settlement.payload.get("occurrence_datetime")
    if not isinstance(value, str) or not value:
        raw_market = settlement.payload.get("raw_market")
        if isinstance(raw_market, dict):
            value = raw_market.get("occurrence_datetime")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def prediction_occurrence_time(rule: MarketEvent) -> datetime | None:
    value = rule.payload.get("occurrence_datetime")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def prediction_expected_expiration_time(rule: MarketEvent) -> datetime | None:
    value = rule.payload.get("expected_expiration_time")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def prediction_latest_expiration_time(rule: MarketEvent) -> datetime | None:
    value = rule.payload.get("latest_expiration_time")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def prediction_close_time(rule: MarketEvent) -> datetime | None:
    value = rule.payload.get("close_time")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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


def fast_prediction_settlement_deadline(forecast: Forecast) -> datetime | None:
    """Return the immutable finalization deadline recorded for a fast-lane forecast."""
    value = forecast.values.get("settlement_deadline")
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None


def is_quarantined_prediction_identity_collision(forecast: Forecast) -> bool:
    """Exclude legacy generic forecasts that collided with the fast-lane v4 ID."""
    return (
        forecast.specialist_id == FastPredictionSettlementV4Specialist.agent_id
        and fast_prediction_settlement_deadline(forecast) is None
    )


TIMING_GUARDED_PREDICTION_SPECIALISTS = frozenset(
    {
        "prediction-market-calibration-baseline-v2",
        PredictionMarketCalibrationSpecialist.agent_id,
        AdjustedPredictionMarketCalibrationSpecialist.agent_id,
        FastPredictionSettlementSpecialist.agent_id,
        FastPredictionSettlementV3Specialist.agent_id,
        FastPredictionSettlementV4Specialist.agent_id,
        FastPredictionSettlementV5Specialist.agent_id,
        FastPredictionSettlementV6Specialist.agent_id,
        FastPredictionSettlementV7Specialist.agent_id,
        FastPredictionSettlementV8Specialist.agent_id,
        FastPredictionSettlementV9Specialist.agent_id,
        FastPredictionSettlementV10Specialist.agent_id,
        FastPredictionSettlementV11Specialist.agent_id,
        FastPredictionSettlementV12Specialist.agent_id,
        FastPredictionSettlementV13Specialist.agent_id,
        FastPredictionSettlementV14Specialist.agent_id,
        FastPredictionSettlementV15Specialist.agent_id,
        FastPredictionSettlementV16Specialist.agent_id,
    }
)
