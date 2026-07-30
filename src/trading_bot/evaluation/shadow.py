from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Mapping

from trading_bot.agents.base import Specialist
from trading_bot.agents.breakout import CryptoRangeBreakoutSpecialist
from trading_bot.agents.crypto_momentum import (
    CryptoIntradayMomentumSpecialist,
    CryptoIntradayMomentumV2Specialist,
)
from trading_bot.agents.market_math import finite_float, prediction_book
from trading_bot.agents.option_volatility import (
    OptionVolatilitySpecialist,
    option_quote_is_fresh,
)
from trading_bot.agents.perpetual import PerpetualFundingBasisSpecialist
from trading_bot.agents.prediction import (
    AdjustedPredictionMarketCalibrationSpecialist,
    FastPredictionSettlementSpecialist,
    FastPredictionSettlementV3Specialist,
    FastPredictionSettlementV4Specialist,
    FastPredictionSettlementV5Specialist,
    FastPredictionSettlementV6Specialist,
    TIMING_GUARDED_PREDICTION_SPECIALISTS,
    fast_prediction_settlement_deadline,
    is_quarantined_prediction_identity_collision,
    prediction_forecast_target_time,
    prediction_expected_expiration_time,
    prediction_latest_expiration_time,
    prediction_occurrence_time,
    prediction_settlement_event_ticker,
    prediction_settlement_event_key,
    prediction_settlement_occurrence_time,
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
from trading_bot.evaluation.outcomes import forecast_outcome_target_time
from trading_bot.evaluation.checkpoint import checkpointed_walk_forward_report
from trading_bot.evaluation.reporting import (
    DECISION_SCOPE_AGGREGATE,
    EdgeStatus,
    EvaluationDecision,
    WalkForwardReport,
    build_walk_forward_report,
)
from trading_bot.replay import ReplayEngine


@dataclass(frozen=True)
class ForecastGenerationSummary:
    candidates: int
    appended: int
    existing: int
    skipped: int
    blocked_by_rejection: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ForecastScoringSummary:
    unscored: int
    not_due: int
    due_unmatched: int
    quarantined: int
    next_due_at: datetime | None
    oldest_due_at: datetime | None
    matched: int
    appended: int
    existing: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ShadowResearchResult:
    generation: ForecastGenerationSummary
    scoring: ForecastScoringSummary
    report: WalkForwardReport
    locked_decisions: tuple[EvaluationDecision, ...]
    fast_prediction_eligibility: FastPredictionEligibilitySummary


@dataclass(frozen=True)
class FastPredictionEligibilitySummary:
    paired_markets: int
    fresh_book_markets: int
    active_markets: int
    documented_close_policy_markets: int
    early_close_enabled_markets: int
    early_close_disabled_markets: int
    missing_close_policy_markets: int
    invalid_close_policy_markets: int
    short_timer_markets: int
    horizon_markets: int
    executable_markets: int
    unforecasted_event_candidates: int
    selected_events: int


@dataclass(frozen=True)
class IntradayMomentumEligibilitySummary:
    observed_instruments: int
    fresh_instruments: int
    adequate_lookback_instruments: int
    signal_instruments: int
    v2_assigned_instruments: int
    v2_signal_instruments: int


class ShadowResearchProfile(StrEnum):
    """Bound candidate work to the instruments observed by a scheduler lane."""

    FULL = "full"
    RAPID = "rapid"


@dataclass(frozen=True)
class ShadowResearchConfig:
    max_prediction_forecasts: int = 25
    max_prediction_history: int = 250
    profile: ShadowResearchProfile = ShadowResearchProfile.FULL

    def __post_init__(self) -> None:
        if self.max_prediction_forecasts < 1 or self.max_prediction_history < 1:
            raise ValueError("shadow research limits must be positive")
        if not isinstance(self.profile, ShadowResearchProfile):
            raise ValueError("shadow research profile must be supported")


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
        scoring = self.score_available(
            as_of=as_of,
            specialist_ids=(
                self._rapid_specialist_ids()
                if self.config.profile is ShadowResearchProfile.RAPID
                else None
            ),
        )
        # Checkpoint before generation so a strategy which has just reached its
        # preregistered rejection boundary cannot issue one more forecast.  The
        # returned report is also reused by the caller, avoiding a second full
        # walk-forward pass in every scheduled evidence cycle.
        report, locked_decisions = checkpointed_walk_forward_report(self.audit, as_of=as_of)
        rejected_specialists = frozenset(
            item.specialist_id
            for item in report.groups
            if item.locked_status is EdgeStatus.REJECTED
        )
        # The rapid CLI publishes this funnel as telemetry and then generates
        # forecasts from the identical immutable observation set. Reuse the
        # selection so a bounded rapid cycle does not perform the same full
        # point-in-time scan twice.
        fast_prediction_candidates, fast_prediction_eligibility = (
            self._fast_prediction_selection(as_of)
        )
        generation = self.generate_forecasts(
            as_of=as_of,
            rejected_specialists=rejected_specialists,
            fast_prediction_candidates=fast_prediction_candidates,
        )
        return ShadowResearchResult(
            generation,
            scoring,
            report,
            locked_decisions,
            fast_prediction_eligibility,
        )

    def generate_forecasts(
        self,
        *,
        as_of: datetime,
        rejected_specialists: frozenset[str] | None = None,
        fast_prediction_candidates: list[_Candidate] | None = None,
    ) -> ForecastGenerationSummary:
        as_of = require_aware(as_of, "as_of")
        candidates = self._candidates(
            as_of, fast_prediction_candidates=fast_prediction_candidates
        )
        replay = ReplayEngine(self.store)
        appended = 0
        existing = 0
        skipped = 0
        blocked_by_rejection = 0
        errors: list[str] = []
        if rejected_specialists is None:
            rejected_specialists = self._rejected_specialist_ids()
        for candidate in candidates:
            if candidate.specialist.agent_id in rejected_specialists:
                blocked_by_rejection += 1
                continue
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
            len(candidates),
            appended,
            existing,
            skipped,
            blocked_by_rejection,
            tuple(errors),
        )

    def _rejected_specialist_ids(self) -> frozenset[str]:
        """Freeze new forecasts after a preregistered aggregate rejection.

        Scores and existing forecasts remain append-only and continue through the
        normal reconciliation path.  The gate also recognizes a mature rejection
        before the caller checkpoints it, so a cycle that reaches its fixed
        boundary cannot issue an additional forecast before recording the result.
        A locked candidate is deliberately not frozen: its later monitoring is
        reported separately and cannot revise the original decision.
        """
        decisions = self.audit.evaluation_decisions()
        aggregate_decisions = {
            (decision.specialist_id, decision.kind): decision
            for decision in decisions
            if decision.scope == DECISION_SCOPE_AGGREGATE
        }
        rejected = {
            specialist_id
            for (specialist_id, _), decision in aggregate_decisions.items()
            if decision.status is EdgeStatus.REJECTED
        }
        report = build_walk_forward_report(
            self.audit.forecasts(),
            self.audit.forecast_scores(),
            locked_decisions=decisions,
        )
        for group in report.groups:
            # A recorded candidate remains a candidate for operational monitoring,
            # even if later data would look worse.  Only an unrecorded mature
            # rejection is frozen here; checkpointing will append it after the
            # cycle completes.
            if (group.specialist_id, group.kind) in aggregate_decisions:
                continue
            if group.status is EdgeStatus.REJECTED:
                rejected.add(group.specialist_id)
        return frozenset(rejected)

    def _rapid_specialist_ids(self) -> frozenset[str]:
        """Families whose decision-time inputs are refreshed by the rapid plan."""
        return frozenset(
            (
                CryptoIntradayMomentumSpecialist().agent_id,
                CryptoIntradayMomentumV2Specialist().agent_id,
                FastPredictionSettlementV6Specialist().agent_id,
            )
        )

    def score_available(
        self,
        *,
        as_of: datetime,
        specialist_ids: frozenset[str] | None = None,
    ) -> ForecastScoringSummary:
        as_of = require_aware(as_of, "as_of")
        scored_ids = self.audit.scored_forecast_ids()
        forecasts = [
            forecast
            for forecast in self.audit.forecasts()
            if forecast.forecast_id not in scored_ids
            and (specialist_ids is None or forecast.specialist_id in specialist_ids)
        ]
        matched = 0
        appended = 0
        existing = 0
        not_due = 0
        due_unmatched = 0
        quarantined = 0
        next_due_at: datetime | None = None
        oldest_due_at: datetime | None = None
        errors: list[str] = []
        for forecast in forecasts:
            try:
                if is_quarantined_prediction_identity_collision(forecast):
                    quarantined += 1
                    continue
                score = self._match_score(forecast, as_of)
                if score is None:
                    target_time = forecast_outcome_target_time(forecast)
                    if target_time is None:
                        quarantined += 1
                    elif target_time > as_of:
                        not_due += 1
                        if next_due_at is None or target_time < next_due_at:
                            next_due_at = target_time
                    else:
                        due_unmatched += 1
                        if oldest_due_at is None or target_time < oldest_due_at:
                            oldest_due_at = target_time
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
            len(forecasts),
            not_due,
            due_unmatched,
            quarantined,
            next_due_at,
            oldest_due_at,
            matched,
            appended,
            existing,
            tuple(errors),
        )

    def _candidates(
        self,
        as_of: datetime,
        *,
        fast_prediction_candidates: list[_Candidate] | None = None,
    ) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        candidates.extend(self._intraday_momentum_candidates(as_of))
        candidates.extend(self._intraday_momentum_v2_candidates(as_of))
        if self.config.profile is ShadowResearchProfile.RAPID:
            candidates.extend(
                fast_prediction_candidates
                if fast_prediction_candidates is not None
                else self._fast_prediction_candidates(as_of)
            )
            return tuple(candidates)
        candidates.extend(self._breakout_candidates(as_of))
        candidates.extend(self._perpetual_candidates(as_of))
        candidates.extend(self._option_candidates(as_of))
        candidates.extend(self._prediction_candidates(as_of))
        candidates.extend(
            fast_prediction_candidates
            if fast_prediction_candidates is not None
            else self._fast_prediction_candidates(as_of)
        )
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

    def _intraday_momentum_candidates(self, as_of: datetime) -> list[_Candidate]:
        specialist = CryptoIntradayMomentumSpecialist()
        candidates: list[_Candidate] = []
        for instrument in self.store.instruments(asset_class=AssetClass.CRYPTO):
            bars = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.BAR,
            )
            eligible = [
                item
                for item in bars
                if _finite_float(item.payload.get("granularity_seconds"))
                == specialist.config.granularity_seconds
            ]
            if not eligible:
                continue
            latest = max(
                eligible,
                key=lambda item: (item.event_time, item.available_at, item.event_id),
            )
            decision_time = latest.available_at
            if as_of - decision_time > specialist.config.max_receipt_age:
                continue
            candidates.append(
                _Candidate(specialist, instrument.instrument_id, (), decision_time)
            )
        return candidates

    def _intraday_momentum_v2_candidates(self, as_of: datetime) -> list[_Candidate]:
        """Select one pre-assigned instrument per target, before signal evaluation."""
        specialist = CryptoIntradayMomentumV2Specialist()
        candidates: list[_Candidate] = []
        for instrument in self.store.instruments(asset_class=AssetClass.CRYPTO):
            if instrument.symbol.upper() not in specialist.assignment_universe:
                continue
            bars = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.BAR,
            )
            eligible = [
                item
                for item in bars
                if _finite_float(item.payload.get("granularity_seconds"))
                == specialist.config.granularity_seconds
            ]
            if not eligible:
                continue
            latest = max(
                eligible,
                key=lambda item: (item.event_time, item.available_at, item.event_id),
            )
            decision_time = latest.available_at
            target_time = latest.event_time + timedelta(
                seconds=specialist.config.granularity_seconds
            )
            if (
                decision_time < specialist.hypothesis.proposed_at
                or as_of - decision_time > specialist.config.max_receipt_age
                or specialist.selected_symbol(target_time) != instrument.symbol.upper()
            ):
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
        scored_ids = self.audit.scored_forecast_ids()
        active_instruments = {
            forecast.instrument_id
            for forecast in self.audit.forecasts()
            if forecast.specialist_id == specialist.agent_id
            and forecast.kind is ForecastKind.VOLATILITY
            and forecast.forecast_id not in scored_ids
            and forecast.valid_until > as_of
        }
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
            if option.instrument_id in active_instruments:
                continue
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
            if not option_quote_is_fresh(
                quotes[-1], as_of, specialist.config.max_quote_age
            ):
                continue
            underlying = str(option.metadata.get("underlying_symbol") or "").upper()
            equity = by_symbol.get((option.venue, underlying))
            related = (equity.instrument_id,) if equity else ()
            candidates.append(
                _Candidate(specialist, option.instrument_id, related, decision_time)
            )
        return candidates

    def _prediction_candidates(self, as_of: datetime) -> list[_Candidate]:
        specialist = AdjustedPredictionMarketCalibrationSpecialist()
        instruments = self.store.instruments(asset_class=AssetClass.PREDICTION)
        instrument_ids = {item.instrument_id for item in instruments}
        forecasted_event_keys = {
            str(
                forecast.values.get("event_ticker")
                or forecast.values.get("outcome_cluster")
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
        books_by_instrument: dict[str, list[MarketEvent]] = {}
        latest_books: dict[str, MarketEvent] = {}
        for book in self.store.events_available_at(
            as_of, event_type=MarketEventType.BOOK_SNAPSHOT
        ):
            if book.instrument_id not in instrument_ids:
                continue
            books_by_instrument.setdefault(book.instrument_id, []).append(book)
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
        rules_by_instrument: dict[str, list[MarketEvent]] = {}
        for event in self.store.events_available_at(
            as_of, event_type=MarketEventType.CONTRACT_RULE
        ):
            if event.instrument_id not in instrument_ids:
                continue
            rules_by_instrument.setdefault(event.instrument_id, []).append(event)
        settled_by_event: dict[str, list[MarketEvent]] = {}
        open_by_event: dict[str, tuple[datetime, Instrument, float, float]] = {}
        for instrument in instruments:
            settlements = settlements_by_instrument.get(instrument.instrument_id, [])
            valid_settlements = [
                item
                for item in settlements
                if str(item.payload.get("result", "")).lower() in {"yes", "no"}
            ]
            if valid_settlements:
                settlement = min(
                    valid_settlements,
                    key=lambda item: (item.available_at, item.event_id),
                )
                event_key = prediction_settlement_event_key(settlement)
                settled_by_event.setdefault(event_key, []).append(settlement)
                continue
            book = latest_books.get(instrument.instrument_id)
            if book is None:
                continue
            eligible_rules = [
                rule
                for rule in rules_by_instrument.get(instrument.instrument_id, ())
                if rule.available_at <= book.available_at
            ]
            rule = (
                max(
                    eligible_rules,
                    key=lambda item: (item.available_at, item.event_time, item.event_id),
                )
                if eligible_rules
                else None
            )
            if rule is None:
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
            event_key = rule.payload.get("event_ticker")
            if not isinstance(event_key, str) or not event_key:
                continue
            if event_key in forecasted_event_keys:
                continue
            executable = prediction_book(book)
            if executable is None or executable[3] > specialist.config.max_book_spread:
                continue
            candidate = (
                decision_time,
                instrument,
                executable[3],
                executable[2],
            )
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

        open_candidates = [
            (decision_time, instrument, probability)
            for decision_time, instrument, _, probability in open_by_event.values()
        ]
        open_candidates.sort(
            key=lambda item: (item[0], item[1].instrument_id), reverse=True
        )
        candidates: list[_Candidate] = []
        for decision_time, instrument, probability in open_candidates[
            : self.config.max_prediction_forecasts
        ]:
            history_ids = self._prediction_history_ids(
                settled_by_event,
                books_by_instrument,
                decision_time=decision_time,
                target_probability=probability,
                specialist=specialist,
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

    def _fast_prediction_candidates(self, as_of: datetime) -> list[_Candidate]:
        candidates, _ = self._fast_prediction_selection(as_of)
        return candidates

    def fast_prediction_eligibility(self, *, as_of: datetime) -> FastPredictionEligibilitySummary:
        """Return the current preregistered eligibility funnel without generating forecasts."""
        as_of = require_aware(as_of, "as_of")
        _, summary = self._fast_prediction_selection(as_of)
        return summary

    def intraday_momentum_eligibility(
        self, *, as_of: datetime
    ) -> IntradayMomentumEligibilitySummary:
        """Return the current fixed-parameter crypto funnel without recording forecasts."""
        as_of = require_aware(as_of, "as_of")
        specialist = CryptoIntradayMomentumSpecialist()
        observed = 0
        fresh = 0
        adequate = 0
        candidates: list[_Candidate] = []
        for instrument in self.store.instruments(asset_class=AssetClass.CRYPTO):
            bars = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.BAR,
            )
            latest_by_bar = {}
            for event in bars:
                if _finite_float(event.payload.get("granularity_seconds")) != (
                    specialist.config.granularity_seconds
                ):
                    continue
                existing = latest_by_bar.get(event.event_time)
                if existing is None or (event.available_at, event.event_id) > (
                    existing.available_at,
                    existing.event_id,
                ):
                    latest_by_bar[event.event_time] = event
            if not latest_by_bar:
                continue
            observed += 1
            latest = max(
                latest_by_bar.values(),
                key=lambda item: (item.event_time, item.available_at, item.event_id),
            )
            if as_of - latest.available_at > specialist.config.max_receipt_age:
                continue
            fresh += 1
            if len(latest_by_bar) < specialist.config.lookback_bars:
                continue
            adequate += 1
            candidates.append(
                _Candidate(
                    specialist,
                    instrument.instrument_id,
                    (),
                    latest.available_at,
                )
            )
        replay = ReplayEngine(self.store)
        signals = sum(
            bool(
                replay.run(
                    candidate.specialist,
                    instrument_id=candidate.instrument_id,
                    related_instrument_ids=candidate.related_instrument_ids,
                    decision_times=(candidate.decision_time,),
                ).forecasts
            )
            for candidate in candidates
        )
        v2_candidates = self._intraday_momentum_v2_candidates(as_of)
        v2_signals = sum(
            bool(
                replay.run(
                    candidate.specialist,
                    instrument_id=candidate.instrument_id,
                    related_instrument_ids=candidate.related_instrument_ids,
                    decision_times=(candidate.decision_time,),
                ).forecasts
            )
            for candidate in v2_candidates
        )
        return IntradayMomentumEligibilitySummary(
            observed,
            fresh,
            adequate,
            signals,
            len(v2_candidates),
            v2_signals,
        )

    def _fast_prediction_selection(
        self, as_of: datetime
    ) -> tuple[list[_Candidate], FastPredictionEligibilitySummary]:
        specialist = FastPredictionSettlementV6Specialist()
        instruments = self.store.instruments(asset_class=AssetClass.PREDICTION)
        instrument_ids = {item.instrument_id for item in instruments}
        forecasted_events = {
            str(forecast.values.get("event_ticker") or forecast.instrument_id)
            for forecast in self.audit.forecasts()
            if forecast.specialist_id == specialist.agent_id
            and not is_quarantined_prediction_identity_collision(forecast)
        }
        latest_books: dict[str, MarketEvent] = {}
        rules_by_instrument: dict[str, list[MarketEvent]] = {}
        # The rapid lane only needs contract rules and executable books.  Do
        # not deserialize unrelated historical candles, trades, or settlements
        # on every fifteen-minute cycle: they cannot affect this preregistered
        # selection and eventually push the read-only job past its cadence.
        for event in self.store.events_available_at(
            as_of, event_type=MarketEventType.CONTRACT_RULE
        ):
            if event.instrument_id in instrument_ids:
                rules_by_instrument.setdefault(event.instrument_id, []).append(event)
        for event in self.store.events_available_at(
            as_of, event_type=MarketEventType.BOOK_SNAPSHOT
        ):
            if event.instrument_id not in instrument_ids:
                continue
            existing = latest_books.get(event.instrument_id)
            if existing is None or (event.available_at, event.event_id) > (
                existing.available_at,
                existing.event_id,
            ):
                latest_books[event.instrument_id] = event

        best_by_event: dict[str, tuple[datetime, Instrument, float]] = {}
        paired_markets = 0
        fresh_book_markets = 0
        active_markets = 0
        documented_close_policy_markets = 0
        early_close_enabled_markets = 0
        early_close_disabled_markets = 0
        missing_close_policy_markets = 0
        invalid_close_policy_markets = 0
        short_timer_markets = 0
        horizon_markets = 0
        executable_markets = 0
        for instrument in instruments:
            book = latest_books.get(instrument.instrument_id)
            if book is None:
                continue
            eligible_rules = [
                rule
                for rule in rules_by_instrument.get(instrument.instrument_id, ())
                if rule.available_at <= book.available_at
            ]
            rule = (
                max(
                    eligible_rules,
                    key=lambda item: (item.available_at, item.event_time, item.event_id),
                )
                if eligible_rules
                else None
            )
            if rule is None:
                continue
            paired_markets += 1
            decision_time = book.available_at
            if as_of - decision_time > specialist.config.max_book_age:
                continue
            fresh_book_markets += 1
            if str(rule.payload.get("status", "")).lower() != "active":
                continue
            active_markets += 1
            close_constraint = rule.payload.get("can_close_early")
            if not isinstance(close_constraint, bool):
                if "can_close_early" not in rule.payload:
                    missing_close_policy_markets += 1
                else:
                    invalid_close_policy_markets += 1
                continue
            documented_close_policy_markets += 1
            if close_constraint:
                early_close_enabled_markets += 1
            else:
                early_close_disabled_markets += 1
            timer = rule.payload.get("settlement_timer_seconds")
            if isinstance(timer, bool) or (
                isinstance(timer, float)
                and (not math.isfinite(timer) or not timer.is_integer())
            ):
                continue
            try:
                timer_value = int(timer)
            except (TypeError, ValueError):
                continue
            if not 0 < timer_value <= specialist.config.max_settlement_timer_seconds:
                continue
            short_timer_markets += 1
            event_ticker = rule.payload.get("event_ticker")
            expected_expiration = prediction_expected_expiration_time(rule)
            latest_expiration = prediction_latest_expiration_time(rule)
            if (
                not isinstance(event_ticker, str)
                or not event_ticker
                or expected_expiration is None
                or latest_expiration is None
                or latest_expiration < expected_expiration
            ):
                continue
            expected_horizon = expected_expiration - decision_time
            if not (
                specialist.config.min_forecast_horizon < expected_horizon
                <= specialist.config.forecast_horizon
            ):
                continue
            horizon_markets += 1
            executable = prediction_book(book)
            if executable is None or executable[3] > specialist.config.max_book_spread:
                continue
            executable_markets += 1
            if event_ticker in forecasted_events:
                continue
            candidate = (decision_time, instrument, executable[3])
            existing = best_by_event.get(event_ticker)
            if existing is None or (
                candidate[2],
                -candidate[0].timestamp(),
                candidate[1].instrument_id,
            ) < (
                existing[2],
                -existing[0].timestamp(),
                existing[1].instrument_id,
            ):
                best_by_event[event_ticker] = candidate
        selected = sorted(
            best_by_event.values(),
            key=lambda item: (item[0], item[1].instrument_id),
            reverse=True,
        )[: self.config.max_prediction_forecasts]
        candidates = [
            _Candidate(specialist, instrument.instrument_id, (), decision_time)
            for decision_time, instrument, _ in selected
        ]
        return candidates, FastPredictionEligibilitySummary(
            paired_markets,
            fresh_book_markets,
            active_markets,
            documented_close_policy_markets,
            early_close_enabled_markets,
            early_close_disabled_markets,
            missing_close_policy_markets,
            invalid_close_policy_markets,
            short_timer_markets,
            horizon_markets,
            executable_markets,
            len(best_by_event),
            len(candidates),
        )

    def _prediction_history_ids(
        self,
        settled_by_event: Mapping[str, list[MarketEvent]],
        books_by_instrument: Mapping[str, list[MarketEvent]],
        *,
        decision_time: datetime,
        target_probability: float,
        specialist: AdjustedPredictionMarketCalibrationSpecialist,
    ) -> tuple[str, ...]:
        selected: list[tuple[datetime, str, str]] = []
        for event_key, settlements in settled_by_event.items():
            best: tuple[
                tuple[float, datetime, str, str],
                MarketEvent,
            ] | None = None
            for settlement in settlements:
                if settlement.available_at > decision_time:
                    continue
                occurrence_time = prediction_settlement_occurrence_time(settlement)
                if occurrence_time is None:
                    continue
                eligible_books: list[
                    tuple[MarketEvent, tuple[float, float, float, float]]
                ] = []
                for book in books_by_instrument.get(settlement.instrument_id, []):
                    time_to_occurrence = occurrence_time - book.available_at
                    executable = prediction_book(book)
                    if (
                        book.available_at <= decision_time
                        and book.event_time <= occurrence_time
                        and specialist.config.min_forecast_horizon
                        < time_to_occurrence
                        <= specialist.config.forecast_horizon
                        and executable is not None
                        and executable[3] <= specialist.config.max_book_spread
                    ):
                        eligible_books.append((book, executable))
                if not eligible_books:
                    continue
                eligible_books.sort(
                    key=lambda item: (item[0].available_at, item[0].event_id)
                )
                historical_book, executable = eligible_books[-1]
                distance = abs(executable[2] - target_probability)
                if distance > specialist.config.probability_bucket_radius:
                    continue
                rank = (
                    distance,
                    historical_book.available_at,
                    historical_book.event_id,
                    settlement.instrument_id,
                )
                if best is None or rank < best[0]:
                    best = (rank, settlement)
            if best is not None:
                selected.append(
                    (
                        best[1].available_at,
                        event_key,
                        best[1].instrument_id,
                    )
                )
        selected.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return tuple(
            instrument_id
            for _, _, instrument_id in selected[
                : self.config.max_prediction_history
            ]
        )

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
        settlement_deadline: datetime | None = None
        if forecast.specialist_id in TIMING_GUARDED_PREDICTION_SPECIALISTS:
            target_time = prediction_forecast_target_time(forecast)
            if target_time is None or target_time <= forecast.generated_at:
                return None
        if forecast.specialist_id in {
            FastPredictionSettlementSpecialist.agent_id,
            FastPredictionSettlementV3Specialist.agent_id,
            FastPredictionSettlementV4Specialist.agent_id,
            FastPredictionSettlementV5Specialist.agent_id,
            FastPredictionSettlementV6Specialist.agent_id,
        }:
            settlement_deadline = fast_prediction_settlement_deadline(forecast)
            if settlement_deadline is None or settlement_deadline < target_time:
                return None
        expected_event_ticker = forecast.values.get("event_ticker")
        if (
            forecast.specialist_id
            in {
                FastPredictionSettlementV3Specialist.agent_id,
                FastPredictionSettlementV4Specialist.agent_id,
                FastPredictionSettlementV5Specialist.agent_id,
                FastPredictionSettlementV6Specialist.agent_id,
            }
            and (not isinstance(expected_event_ticker, str) or not expected_event_ticker)
        ):
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
                or result not in {"yes", "no"}
            ):
                continue
            if settlement_deadline is not None and event.event_time > settlement_deadline:
                continue
            if (
                forecast.specialist_id
                in {
                    FastPredictionSettlementV3Specialist.agent_id,
                    FastPredictionSettlementV4Specialist.agent_id,
                }
                and (
                    event.event_time < target_time
                    or prediction_settlement_event_ticker(event) != expected_event_ticker
                )
            ):
                continue
            if (
                forecast.specialist_id
                in {
                    FastPredictionSettlementV5Specialist.agent_id,
                    FastPredictionSettlementV6Specialist.agent_id,
                }
                and prediction_settlement_event_ticker(event) != expected_event_ticker
            ):
                continue
            return score_binary_forecast(
                forecast,
                outcome=result == "yes",
                target_time=target_time or event.event_time,
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
