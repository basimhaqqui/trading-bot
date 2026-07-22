from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from datetime import timedelta

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.hypotheses import PERPETUAL_FUNDING_HYPOTHESIS
from trading_bot.agents.market_math import executable_quote, finite_float, recent_events
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    MarketEvent,
    MarketEventType,
)


@dataclass(frozen=True)
class PerpetualFundingConfig:
    funding_lookback: int = 8
    min_funding_observations: int = 2
    max_funding_age: timedelta = timedelta(hours=12)
    max_book_age: timedelta = timedelta(minutes=5)
    assumed_round_trip_cost_bps: float = 8.0
    forecast_horizon: timedelta = timedelta(hours=8)

    def __post_init__(self) -> None:
        if self.funding_lookback < self.min_funding_observations or self.min_funding_observations < 1:
            raise ValueError("funding lookback must cover the minimum observations")
        if self.assumed_round_trip_cost_bps < 0:
            raise ValueError("assumed costs cannot be negative")


class PerpetualFundingBasisSpecialist:
    agent_id = "perpetual-funding-basis-baseline"
    model_version = "baseline-v1"
    supported_asset_classes = frozenset({AssetClass.PERPETUAL})
    hypothesis = PERPETUAL_FUNDING_HYPOTHESIS

    def __init__(self, config: PerpetualFundingConfig | None = None) -> None:
        self.config = config or PerpetualFundingConfig()

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        spot_instruments = [
            item for item in context.related_instruments if item.asset_class is AssetClass.CRYPTO
        ]
        if len(spot_instruments) != 1:
            return None
        spot = spot_instruments[0]
        funding_events = recent_events(
            context.events,
            instrument_id=context.instrument.instrument_id,
            event_type=MarketEventType.FUNDING,
            decision_time=context.decision_time,
            max_age=self.config.max_funding_age,
        )[-self.config.funding_lookback :]
        rates_by_period: dict[str, tuple[MarketEvent, float]] = {}
        for event in funding_events:
            rate = finite_float(event.payload.get("funding_rate"))
            if rate is not None and abs(rate) <= 0.1:
                period = str(event.payload.get("funding_time") or event.event_time.isoformat())
                rates_by_period[period] = (event, rate)
        rates = sorted(
            rates_by_period.values(),
            key=lambda item: (item[0].available_at, item[0].event_id),
        )[-self.config.funding_lookback :]
        if len(rates) < self.config.min_funding_observations:
            return None

        perpetual_books = recent_events(
            context.events,
            instrument_id=context.instrument.instrument_id,
            event_type=MarketEventType.BOOK_SNAPSHOT,
            decision_time=context.decision_time,
            max_age=self.config.max_book_age,
        )
        spot_books = recent_events(
            context.events,
            instrument_id=spot.instrument_id,
            event_type=MarketEventType.BOOK_SNAPSHOT,
            decision_time=context.decision_time,
            max_age=self.config.max_book_age,
        )
        if not perpetual_books or not spot_books:
            return None
        perpetual_quote = executable_quote(perpetual_books[-1])
        spot_quote = executable_quote(spot_books[-1])
        if perpetual_quote is None or spot_quote is None:
            return None

        _, _, perpetual_mid, perpetual_spread_bps = perpetual_quote
        _, _, spot_mid, spot_spread_bps = spot_quote
        basis_bps = (perpetual_mid / spot_mid - 1.0) * 10_000
        predicted_rate = statistics.median(rate for _, rate in rates)
        current_rate = rates[-1][1]
        funding_bps = predicted_rate * 10_000
        execution_bound_bps = (
            self.config.assumed_round_trip_cost_bps
            + perpetual_spread_bps / 2
            + spot_spread_bps / 2
        )
        carry_margin_bps = abs(funding_bps) - execution_bound_bps
        same_signed = funding_bps * basis_bps > 0
        if carry_margin_bps > 0 and same_signed and funding_bps > 0:
            state = "positive_carry_long_spot_short_perpetual"
        elif carry_margin_bps > 0 and same_signed and funding_bps < 0:
            state = "negative_carry_long_perpetual_short_spot"
        else:
            state = "inside_cost_or_basis_bound"

        confidence = min(0.7, 0.25 + len(rates) * 0.05)
        if not same_signed:
            confidence *= 0.6
        confidence *= max(0.25, 1.0 - min(0.75, execution_bound_bps / 200))
        evidence = [event.event_id for event, _ in rates]
        evidence.extend((perpetual_books[-1].event_id, spot_books[-1].event_id))
        open_interest = recent_events(
            context.events,
            instrument_id=context.instrument.instrument_id,
            event_type=MarketEventType.OPEN_INTEREST,
            decision_time=context.decision_time,
            max_age=self.config.max_funding_age,
        )
        values: dict[str, float | str | bool] = {
            "current_funding_rate": current_rate,
            "current_funding_time": str(
                rates[-1][0].payload.get("funding_time")
                or rates[-1][0].event_time.isoformat()
            ),
            "predicted_funding_rate": predicted_rate,
            "perpetual_spot_basis_bps": basis_bps,
            "execution_bound_bps": execution_bound_bps,
            "carry_margin_bps": carry_margin_bps,
            "funding_and_basis_same_signed": same_signed,
            "state": state,
        }
        if open_interest:
            open_interest_value = finite_float(open_interest[-1].payload.get("open_interest"))
            if open_interest_value is not None:
                values["open_interest"] = open_interest_value
                evidence.append(open_interest[-1].event_id)
        return Forecast(
            forecast_id=str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent_id}:{context.instrument.instrument_id}:{context.decision_time}")
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=context.instrument.instrument_id,
            kind=ForecastKind.FUNDING_RATE,
            generated_at=context.decision_time,
            valid_until=context.decision_time + self.config.forecast_horizon,
            values=values,
            confidence=confidence,
            uncertainty={
                "funding_observations": float(len(rates)),
                "combined_spread_bps": perpetual_spread_bps + spot_spread_bps,
            },
            evidence_event_ids=tuple(dict.fromkeys(evidence)),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )
