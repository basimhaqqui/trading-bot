from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log, sqrt

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.hypotheses import OPTIONS_VOLATILITY_HYPOTHESIS
from trading_bot.agents.market_math import executable_quote, finite_float, recent_events
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    MarketEvent,
    MarketEventType,
)


@dataclass(frozen=True)
class OptionVolatilityConfig:
    lookback: int = 20
    min_observations: int = 3
    max_quote_age: timedelta = timedelta(hours=1)
    state_threshold: float = 0.10
    underlying_bar_lookback: int = 30
    min_underlying_returns: int = 5
    forecast_horizon: timedelta = timedelta(days=1)

    def __post_init__(self) -> None:
        if self.lookback < self.min_observations or self.min_observations < 2:
            raise ValueError("lookback must cover at least two observations")
        if self.state_threshold <= 0:
            raise ValueError("state_threshold must be positive")
        if self.underlying_bar_lookback <= self.min_underlying_returns:
            raise ValueError("underlying bar lookback must exceed minimum returns")


class OptionVolatilitySpecialist:
    agent_id = "options-implied-volatility-state-baseline"
    model_version = "baseline-v1"
    supported_asset_classes = frozenset({AssetClass.OPTION})
    hypothesis = OPTIONS_VOLATILITY_HYPOTHESIS

    def __init__(self, config: OptionVolatilityConfig | None = None) -> None:
        self.config = config or OptionVolatilityConfig()

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        quotes = recent_events(
            context.events,
            instrument_id=context.instrument.instrument_id,
            event_type=MarketEventType.QUOTE,
            decision_time=context.decision_time,
        )[-self.config.lookback :]
        observations: list[tuple[object, float]] = []
        for event in quotes:
            implied = finite_float(event.payload.get("implied_volatility"))
            if implied is not None and 0 < implied <= 10:
                observations.append((event, implied))
        if len(observations) < self.config.min_observations:
            return None
        latest_event = observations[-1][0]
        if not option_quote_is_fresh(
            latest_event, context.decision_time, self.config.max_quote_age
        ):
            return None
        quote = executable_quote(latest_event)
        if quote is None:
            return None
        _, _, option_midpoint, spread_bps = quote
        current_iv = observations[-1][1]
        trailing_median = statistics.median(value for _, value in observations[:-1])
        relative_deviation = current_iv / trailing_median - 1.0
        expected_iv = 0.6 * current_iv + 0.4 * trailing_median
        if relative_deviation >= self.config.state_threshold:
            state = "implied_volatility_elevated"
        elif relative_deviation <= -self.config.state_threshold:
            state = "implied_volatility_discounted"
        else:
            state = "implied_volatility_near_trailing_state"
        feed = str(latest_event.payload.get("feed", "unknown"))
        confidence_cap = 0.25 if feed == "indicative" else 0.7
        confidence = min(confidence_cap, 0.2 + len(observations) * 0.035)
        confidence *= max(0.2, 1.0 - min(0.8, spread_bps / 2_000))
        target_time = context.decision_time + self.config.forecast_horizon
        values: dict[str, float | str | bool] = {
            "current_implied_volatility": current_iv,
            "expected_implied_volatility": expected_iv,
            "trailing_median_implied_volatility": trailing_median,
            "relative_deviation": relative_deviation,
            "option_midpoint": option_midpoint,
            "spread_bps": spread_bps,
            "feed": feed,
            "state": state,
            "outcome_cluster": f"option-session:{target_time.date().isoformat()}",
        }
        evidence = [event.event_id for event, _ in observations]
        underlying_realized = self._underlying_realized_volatility(context)
        if underlying_realized is not None:
            realized_volatility, bar_events = underlying_realized
            values["underlying_realized_volatility"] = realized_volatility
            values["implied_minus_realized_volatility"] = current_iv - realized_volatility
            evidence.extend(event.event_id for event in bar_events)
        return Forecast(
            forecast_id=str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent_id}:{context.instrument.instrument_id}:{context.decision_time}")
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=context.instrument.instrument_id,
            kind=ForecastKind.VOLATILITY,
            generated_at=context.decision_time,
            valid_until=target_time,
            values=values,
            confidence=confidence,
            uncertainty={
                "observations": float(len(observations)),
                "indicative_feed": 1.0 if feed == "indicative" else 0.0,
            },
            evidence_event_ids=tuple(dict.fromkeys(evidence)),
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )

    def _underlying_realized_volatility(
        self, context: ReplayContext
    ) -> tuple[float, list[MarketEvent]] | None:
        equities = [
            item for item in context.related_instruments if item.asset_class is AssetClass.EQUITY
        ]
        if len(equities) != 1:
            return None
        bars = recent_events(
            context.events,
            instrument_id=equities[0].instrument_id,
            event_type=MarketEventType.BAR,
            decision_time=context.decision_time,
        )
        latest_by_time = {}
        for event in bars:
            close = finite_float(event.payload.get("close"))
            if close is not None and close > 0:
                latest_by_time[event.event_time] = (event, close)
        ordered = [latest_by_time[key] for key in sorted(latest_by_time)][
            -self.config.underlying_bar_lookback :
        ]
        returns = [log(current[1] / previous[1]) for previous, current in zip(ordered, ordered[1:])]
        if len(returns) < self.config.min_underlying_returns:
            return None
        return statistics.stdev(returns) * sqrt(252), [event for event, _ in ordered]


def option_quote_is_fresh(
    event: MarketEvent, decision_time: datetime, max_age: timedelta
) -> bool:
    source_age = decision_time - event.event_time
    receipt_age = decision_time - event.available_at
    return all(
        timedelta(0) <= age <= max_age for age in (source_age, receipt_age)
    )
