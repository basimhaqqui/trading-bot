from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from trading_bot.agents.base import Specialist
from trading_bot.agents.breakout import CryptoRangeBreakoutSpecialist
from trading_bot.agents.market_math import finite_float, prediction_book
from trading_bot.agents.option_volatility import OptionVolatilitySpecialist
from trading_bot.agents.perpetual import PerpetualFundingBasisSpecialist
from trading_bot.agents.prediction import (
    PredictionMarketCalibrationSpecialist,
    TIMING_GUARDED_PREDICTION_SPECIALISTS,
    prediction_occurrence_time,
)
from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    Instrument,
    MarketEvent,
    MarketEventType,
)
from trading_bot.core.serialization import parse_datetime, require_aware
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.scoring import (
    ForecastScore,
    score_binary_forecast,
    score_funding_forecast,
    score_return_forecast,
    score_volatility_forecast,
)
from trading_bot.replay import ReplayEngine


@dataclass(frozen=True)
class ForecastGenerationSummary:
    candidates: int
    appended: int
    existing: int
    skipped: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ForecastScoringSummary:
    unscored: int
    matched: int
    appended: int
    existing: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ShadowResearchResult:
    generation: ForecastGenerationSummary
    scoring: ForecastScoringSummary


@dataclass(frozen=True)
class ShadowResearchConfig:
    max_prediction_forecasts: int = 25
    max_prediction_history: int = 250

    def __post_init__(self) -> None:
        if self.max_prediction_forecasts < 1 or self.max_prediction_history < 1:
            raise ValueError("shadow research limits must be positive")


@dataclass(frozen=True)
class _Candidate:
    specialist: Specialist
    instrument_id: str
    related_instrument_ids: tuple[str, ...]
    decision_time: datetime


class ShadowResearchRunner:
    def __init__(
        self,
        store: PointInTimeStore,
        audit: AuditLedger,
        config: ShadowResearchConfig | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.config = config or ShadowResearchConfig()

    def run(self, *, as_of: datetime) -> ShadowResearchResult:
        as_of = require_aware(as_of, "as_of")
        scoring = self.score_available(as_of=as_of)
        generation = self.generate_forecasts(as_of=as_of)
        return ShadowResearchResult(generation, scoring)

    def generate_forecasts(self, *, as_of: datetime) -> ForecastGenerationSummary:
        as_of = require_aware(as_of, "as_of")
        candidates = self._candidates(as_of)
        replay = ReplayEngine(self.store)
        appended = 0
        existing = 0
        skipped = 0
        errors: list[str] = []
        for candidate in candidates:
            try:
                result = replay.run(
                    candidate.specialist,
                    instrument_id=candidate.instrument_id,
                    related_instrument_ids=candidate.related_instrument_ids,
                    decision_times=(candidate.decision_time,),
                )
                if not result.forecasts:
                    skipped += 1
                    continue
                if self.audit.append_forecast(result.forecasts[0]):
                    appended += 1
                else:
                    existing += 1
            except Exception as exc:
                errors.append(
                    f"{candidate.instrument_id}: {type(exc).__name__}: {exc}"
                )
        return ForecastGenerationSummary(
            len(candidates), appended, existing, skipped, tuple(errors)
        )

    def score_available(self, *, as_of: datetime) -> ForecastScoringSummary:
        as_of = require_aware(as_of, "as_of")
        scored_ids = self.audit.scored_forecast_ids()
        forecasts = [
            forecast
            for forecast in self.audit.forecasts()
            if forecast.forecast_id not in scored_ids
        ]
        matched = 0
        appended = 0
        existing = 0
        errors: list[str] = []
        for forecast in forecasts:
            try:
                score = self._match_score(forecast, as_of)
                if score is None:
                    continue
                matched += 1
                if self.audit.append_forecast_score(score):
                    appended += 1
                else:
                    existing += 1
            except Exception as exc:
                errors.append(
                    f"{forecast.forecast_id}: {type(exc).__name__}: {exc}"
                )
        return ForecastScoringSummary(
            len(forecasts), matched, appended, existing, tuple(errors)
        )

    def _candidates(self, as_of: datetime) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        candidates.extend(self._breakout_candidates(as_of))
        candidates.extend(self._perpetual_candidates(as_of))
        candidates.extend(self._option_candidates(as_of))
        candidates.extend(self._prediction_candidates(as_of))
        return tuple(candidates)

    def _breakout_candidates(self, as_of: datetime) -> list[_Candidate]:
        specialist = CryptoRangeBreakoutSpecialist()
        instruments = [
            *self.store.instruments(asset_class=AssetClass.CRYPTO),
            *self.store.instruments(asset_class=AssetClass.MEMECOIN),
        ]
        candidates: list[_Candidate] = []
        for instrument in instruments:
            bars = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.BAR,
            )
            if not bars:
                continue
            latest = max(
                bars,
                key=lambda item: (item.event_time, item.available_at, item.event_id),
            )
            decision_time = latest.available_at
            if as_of - decision_time > specialist.config.max_receipt_age:
                continue
            candidates.append(
                _Candidate(specialist, instrument.instrument_id, (), decision_time)
            )
        return candidates

    def _perpetual_candidates(self, as_of: datetime) -> list[_Candidate]:
        specialist = PerpetualFundingBasisSpecialist()
        crypto = self.store.instruments(asset_class=AssetClass.CRYPTO)
        candidates: list[_Candidate] = []
        for perpetual in self.store.instruments(asset_class=AssetClass.PERPETUAL):
            spot = self._related_spot(perpetual, crypto, as_of)
            if spot is None:
                continue
            funding = self.store.events_available_at(
                as_of,
                instrument_id=perpetual.instrument_id,
                event_type=MarketEventType.FUNDING,
            )
            perpetual_books = self.store.events_available_at(
                as_of,
                instrument_id=perpetual.instrument_id,
                event_type=MarketEventType.BOOK_SNAPSHOT,
            )
            spot_books = self.store.events_available_at(
                as_of,
                instrument_id=spot.instrument_id,
                event_type=MarketEventType.BOOK_SNAPSHOT,
            )
            if not funding or not perpetual_books or not spot_books:
                continue
            if as_of - max(item.available_at for item in perpetual_books) > specialist.config.max_book_age:
                continue
            if as_of - max(item.available_at for item in spot_books) > specialist.config.max_book_age:
                continue
            if as_of - max(item.available_at for item in funding) > specialist.config.max_funding_age:
                continue
            decision_time = max(
                max(item.available_at for item in funding),
                max(item.available_at for item in perpetual_books),
                max(item.available_at for item in spot_books),
            )
            candidates.append(
                _Candidate(
                    specialist,
                    perpetual.instrument_id,
                    (spot.instrument_id,),
                    decision_time,
                )
            )
        return candidates

    def _related_spot(
        self,
        perpetual: Instrument,
        crypto: list[Instrument],
        as_of: datetime,
    ) -> Instrument | None:
        root = ""
        rules = self.store.events_available_at(
            as_of,
            instrument_id=perpetual.instrument_id,
            event_type=MarketEventType.CONTRACT_RULE,
        )
        if rules:
            product = rules[-1].payload.get("product")
            if isinstance(product, Mapping):
                details = product.get("future_product_details")
                if isinstance(details, Mapping):
                    root = str(details.get("contract_root_unit") or "").upper()
        if not root:
            root = perpetual.symbol.split("-", 1)[0].upper()
        expected = f"{root}-{perpetual.quote_currency}".upper()
        matches = [
            item
            for item in crypto
            if item.venue == perpetual.venue and item.symbol.upper() == expected
        ]
        return matches[0] if matches else None

    def _option_candidates(self, as_of: datetime) -> list[_Candidate]:
        specialist = OptionVolatilitySpecialist()
        equities = self.store.instruments(asset_class=AssetClass.EQUITY)
        by_symbol = {(item.venue, item.symbol.upper()): item for item in equities}
        options = self.store.instruments(asset_class=AssetClass.OPTION)
        option_ids = {item.instrument_id for item in options}
        quotes_by_instrument: dict[str, list[MarketEvent]] = {}
        for quote in self.store.events_available_at(
            as_of, event_type=MarketEventType.QUOTE
        ):
            if quote.instrument_id not in option_ids:
                continue
            quotes_by_instrument.setdefault(quote.instrument_id, []).append(quote)
        candidates: list[_Candidate] = []
        for option in options:
            quotes = quotes_by_instrument.get(option.instrument_id, [])
            quotes.sort(
                key=lambda item: (item.available_at, item.event_time, item.event_id)
            )
            recent = quotes[-specialist.config.lookback :]
            valid_observations = [
                quote
                for quote in recent
                if (value := finite_float(quote.payload.get("implied_volatility")))
                is not None
                and 0 < value <= 10
            ]
            if len(valid_observations) < specialist.config.min_observations:
                continue
            decision_time = quotes[-1].available_at
            if as_of - decision_time > specialist.config.max_quote_age:
                continue
            underlying = str(option.metadata.get("underlying_symbol") or "").upper()
            equity = by_symbol.get((option.venue, underlying))
            related = (equity.instrument_id,) if equity else ()
            candidates.append(
                _Candidate(specialist, option.instrument_id, related, decision_time)
            )
        return candidates

    def _prediction_candidates(self, as_of: datetime) -> list[_Candidate]:
        specialist = PredictionMarketCalibrationSpecialist()
        instruments = self.store.instruments(asset_class=AssetClass.PREDICTION)
        instrument_ids = {item.instrument_id for item in instruments}
        forecasted_event_keys = {
            str(
                forecast.values.get("outcome_cluster")
                or forecast.values.get("event_ticker")
                or forecast.instrument_id
            )
            for forecast in self.audit.forecasts()
            if forecast.kind is ForecastKind.BINARY_PROBABILITY
            and forecast.specialist_id == specialist.agent_id
        }
        settlements_by_instrument: dict[str, list[MarketEvent]] = {}
        for settlement in self.store.events_available_at(
            as_of, event_type=MarketEventType.SETTLEMENT
        ):
            if settlement.instrument_id in instrument_ids:
                settlements_by_instrument.setdefault(
                    settlement.instrument_id, []
                ).append(settlement)
        latest_books: dict[str, MarketEvent] = {}
        for book in self.store.events_available_at(
            as_of, event_type=MarketEventType.BOOK_SNAPSHOT
        ):
            if book.instrument_id not in instrument_ids:
                continue
            current = latest_books.get(book.instrument_id)
            if current is None or (
                book.available_at,
                book.event_time,
                book.event_id,
            ) > (
                current.available_at,
                current.event_time,
                current.event_id,
            ):
                latest_books[book.instrument_id] = book
        latest_rules: dict[str, MarketEvent] = {}
        for event in self.store.events_available_at(
            as_of, event_type=MarketEventType.CONTRACT_RULE
        ):
            if event.instrument_id not in instrument_ids:
                continue
            current = latest_rules.get(event.instrument_id)
            if current is None or (
                event.available_at,
                event.event_time,
                event.event_id,
            ) > (
                current.available_at,
                current.event_time,
                current.event_id,
            ):
                latest_rules[event.instrument_id] = event
        settled: list[tuple[datetime, str]] = []
        open_by_event: dict[str, tuple[datetime, Instrument, float]] = {}
        for instrument in instruments:
            settlements = settlements_by_instrument.get(instrument.instrument_id, [])
            valid_settlements = [
                item
                for item in settlements
                if str(item.payload.get("result", "")).lower() in {"yes", "no"}
            ]
            if valid_settlements:
                settled.append(
                    (
                        min(item.available_at for item in valid_settlements),
                        instrument.instrument_id,
                    )
                )
                continue
            book = latest_books.get(instrument.instrument_id)
            rule = latest_rules.get(instrument.instrument_id)
            if book is None or rule is None:
                continue
            decision_time = book.available_at
            if as_of - decision_time > specialist.config.max_book_age:
                continue
            target_time = prediction_occurrence_time(rule)
            time_to_occurrence = (
                target_time - decision_time if target_time is not None else None
            )
            if (
                target_time is None
                or time_to_occurrence is None
                or time_to_occurrence <= specialist.config.min_forecast_horizon
                or time_to_occurrence > specialist.config.forecast_horizon
            ):
                continue
            event_key = str(rule.payload["occurrence_datetime"])
            if event_key in forecasted_event_keys:
                continue
            executable = prediction_book(book)
            if executable is None or executable[3] > specialist.config.max_book_spread:
                continue
            candidate = (decision_time, instrument, executable[3])
            existing = open_by_event.get(event_key)
            if existing is None or (
                candidate[2],
                -candidate[0].timestamp(),
                candidate[1].instrument_id,
            ) < (
                existing[2],
                -existing[0].timestamp(),
                existing[1].instrument_id,
            ):
                open_by_event[event_key] = candidate

        settled.sort(key=lambda item: (item[0], item[1]), reverse=True)
        settled = settled[: self.config.max_prediction_history]
        open_candidates = [
            (decision_time, instrument)
            for decision_time, instrument, _ in open_by_event.values()
        ]
        open_candidates.sort(
            key=lambda item: (item[0], item[1].instrument_id), reverse=True
        )
        candidates: list[_Candidate] = []
        for decision_time, instrument in open_candidates[
            : self.config.max_prediction_forecasts
        ]:
            history_ids = tuple(
                instrument_id
                for available_at, instrument_id in settled
                if available_at <= decision_time
            )
            candidates.append(
                _Candidate(
                    specialist,
                    instrument.instrument_id,
                    history_ids,
                    decision_time,
                )
            )
        return candidates

    def _match_score(
        self, forecast: Forecast, as_of: datetime
    ) -> ForecastScore | None:
        if forecast.kind is ForecastKind.FUNDING_RATE:
            return self._match_funding(forecast, as_of)
        if forecast.kind is ForecastKind.VOLATILITY:
            return self._match_volatility(forecast, as_of)
        if forecast.kind is ForecastKind.BINARY_PROBABILITY:
            return self._match_binary(forecast, as_of)
        if forecast.kind is ForecastKind.RETURN_DISTRIBUTION:
            return self._match_return(forecast, as_of)
        return None

    def _match_return(
        self, forecast: Forecast, as_of: datetime
    ) -> ForecastScore | None:
        reference_close = _finite_float(forecast.values.get("reference_close"))
        if reference_close is None or reference_close <= 0:
            return None
        events = self.store.events_available_at(
            as_of,
            instrument_id=forecast.instrument_id,
            event_type=MarketEventType.BAR,
        )
        events.sort(key=lambda item: (item.available_at, item.event_time, item.event_id))
        for event in events:
            if (
                event.available_at <= forecast.generated_at
                or event.event_time < forecast.valid_until
            ):
                continue
            close = _finite_float(event.payload.get("close"))
            if close is None or close <= 0:
                continue
            return score_return_forecast(
                forecast,
                actual_return=close / reference_close - 1.0,
                target_time=event.event_time,
                scored_at=as_of,
            )
        return None

    def _match_funding(
        self, forecast: Forecast, as_of: datetime
    ) -> ForecastScore | None:
        current_period = self._funding_period_at_forecast(forecast)
        events = self.store.events_available_at(
            as_of,
            instrument_id=forecast.instrument_id,
            event_type=MarketEventType.FUNDING,
        )
        events.sort(key=lambda item: (item.available_at, item.event_time, item.event_id))
        for event in events:
            if event.available_at <= forecast.generated_at:
                continue
            period = str(event.payload.get("funding_time") or event.event_time.isoformat())
            if current_period and period == current_period:
                continue
            actual = _finite_float(event.payload.get("funding_rate"))
            target_time = _payload_time(event.payload.get("funding_time")) or event.event_time
            if actual is None or target_time < forecast.generated_at:
                continue
            return score_funding_forecast(
                forecast,
                actual_rate=actual,
                target_time=target_time,
                scored_at=as_of,
            )
        return None

    def _funding_period_at_forecast(self, forecast: Forecast) -> str:
        stored = forecast.values.get("current_funding_time")
        if isinstance(stored, str) and stored:
            return stored
        evidence: list[MarketEvent] = []
        for event_id in forecast.evidence_event_ids:
            try:
                event = self.store.event(event_id)
            except KeyError:
                continue
            if event.event_type is MarketEventType.FUNDING:
                evidence.append(event)
        if not evidence:
            return ""
        latest = max(evidence, key=lambda item: (item.available_at, item.event_id))
        return str(latest.payload.get("funding_time") or latest.event_time.isoformat())

    def _match_volatility(
        self, forecast: Forecast, as_of: datetime
    ) -> ForecastScore | None:
        events = self.store.events_available_at(
            as_of,
            instrument_id=forecast.instrument_id,
            event_type=MarketEventType.QUOTE,
        )
        events.sort(key=lambda item: (item.available_at, item.event_time, item.event_id))
        for event in events:
            if event.available_at < forecast.valid_until or event.event_time < forecast.valid_until:
                continue
            actual = _finite_float(event.payload.get("implied_volatility"))
            if actual is None or not 0 < actual <= 10:
                continue
            return score_volatility_forecast(
                forecast,
                actual_implied_volatility=actual,
                target_time=event.event_time,
                scored_at=as_of,
            )
        return None

    def _match_binary(
        self, forecast: Forecast, as_of: datetime
    ) -> ForecastScore | None:
        target_time: datetime | None = None
        if forecast.specialist_id in TIMING_GUARDED_PREDICTION_SPECIALISTS:
            target_time = _payload_time(forecast.values.get("outcome_cluster"))
            if target_time is None or target_time <= forecast.generated_at:
                return None
        events = self.store.events_available_at(
            as_of,
            instrument_id=forecast.instrument_id,
            event_type=MarketEventType.SETTLEMENT,
        )
        events.sort(key=lambda item: (item.available_at, item.event_time, item.event_id))
        for event in events:
            result = str(event.payload.get("result", "")).lower()
            if (
                event.available_at <= forecast.generated_at
                or event.event_time < forecast.generated_at
                or (target_time is not None and event.event_time < target_time)
                or result not in {"yes", "no"}
            ):
                continue
            return score_binary_forecast(
                forecast,
                outcome=result == "yes",
                target_time=event.event_time,
                scored_at=as_of,
            )
        return None


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _payload_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_datetime(value)
    except (TypeError, ValueError):
        return None
