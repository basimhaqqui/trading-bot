from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from datetime import timedelta

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.hypotheses import CRYPTO_BREAKOUT_HYPOTHESIS
from trading_bot.agents.market_math import bar_values
from trading_bot.core.schemas import AssetClass, Forecast, ForecastKind, MarketEventType


@dataclass(frozen=True)
class CryptoBreakoutConfig:
    lookback_bars: int = 20
    minimum_volume_ratio: float = 0.75
    forecast_shrinkage: float = 0.25
    maximum_absolute_forecast: float = 0.05
    granularity_seconds: int = 3600
    max_receipt_age: timedelta = timedelta(minutes=90)

    def __post_init__(self) -> None:
        if self.lookback_bars < 5:
            raise ValueError("breakout lookback must be at least five bars")
        if self.minimum_volume_ratio < 0:
            raise ValueError("minimum volume ratio cannot be negative")
        if not 0 < self.forecast_shrinkage <= 1:
            raise ValueError("forecast shrinkage must be between zero and one")
        if self.maximum_absolute_forecast <= 0:
            raise ValueError("maximum forecast must be positive")
        if self.granularity_seconds <= 0:
            raise ValueError("breakout granularity must be positive")


class CryptoRangeBreakoutSpecialist:
    agent_id = "crypto-range-breakout-continuation-baseline"
    model_version = "baseline-v1"
    supported_asset_classes = frozenset({AssetClass.CRYPTO, AssetClass.MEMECOIN})
    hypothesis = CRYPTO_BREAKOUT_HYPOTHESIS

    def __init__(self, config: CryptoBreakoutConfig | None = None) -> None:
        self.config = config or CryptoBreakoutConfig()

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        latest_by_bar = {}
        for event in context.events:
            if (
                event.instrument_id != context.instrument.instrument_id
                or event.event_type is not MarketEventType.BAR
            ):
                continue
            values = bar_values(event.payload)
            if values is None or values[5] != self.config.granularity_seconds:
                continue
            granularity_seconds = values[5]
            key = (event.event_time, granularity_seconds)
            existing = latest_by_bar.get(key)
            if existing is None or (event.available_at, event.event_id) > (
                existing[0].available_at,
                existing[0].event_id,
            ):
                latest_by_bar[key] = (event, values)
        if not latest_by_bar:
            return None
        latest_event, latest_values = max(
            latest_by_bar.values(),
            key=lambda item: (item[0].event_time, item[0].available_at, item[0].event_id),
        )
        granularity_seconds = latest_values[5]
        ordered = sorted(
            (
                item
                for (_, seconds), item in latest_by_bar.items()
                if seconds == granularity_seconds
            ),
            key=lambda item: (item[0].event_time, item[0].available_at, item[0].event_id),
        )
        required = self.config.lookback_bars + 1
        if len(ordered) < required:
            return None
        ordered = ordered[-required:]
        latest_event, latest_values = ordered[-1]
        if context.decision_time - latest_event.available_at > self.config.max_receipt_age:
            return None

        prior = ordered[:-1]
        prior_high = max(values[1] for _, values in prior)
        prior_low = min(values[2] for _, values in prior)
        prior_volume = statistics.median(values[4] for _, values in prior)
        latest_close = latest_values[3]
        latest_volume = latest_values[4]
        volume_ratio = latest_volume / prior_volume if prior_volume > 0 else 0.0
        if volume_ratio < self.config.minimum_volume_ratio:
            return None

        if latest_close > prior_high:
            direction = "up"
            breakout_distance = latest_close / prior_high - 1.0
        elif latest_close < prior_low:
            direction = "down"
            breakout_distance = latest_close / prior_low - 1.0
        else:
            return None
        predicted_return = max(
            -self.config.maximum_absolute_forecast,
            min(
                self.config.maximum_absolute_forecast,
                breakout_distance * self.config.forecast_shrinkage,
            ),
        )
        target_time = latest_event.event_time + timedelta(seconds=granularity_seconds)
        if target_time <= context.decision_time:
            return None
        confidence = min(0.45, 0.20 + min(volume_ratio, 3.0) * 0.05)
        evidence = tuple(event.event_id for event, _ in ordered)
        return Forecast(
            forecast_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.agent_id}:{context.instrument.instrument_id}:{context.decision_time}",
                )
            ),
            specialist_id=self.agent_id,
            model_version=self.model_version,
            instrument_id=context.instrument.instrument_id,
            kind=ForecastKind.RETURN_DISTRIBUTION,
            generated_at=context.decision_time,
            valid_until=target_time,
            values={
                "predicted_return": predicted_return,
                "benchmark_return": 0.0,
                "reference_close": latest_close,
                "direction": direction,
                "breakout_distance": breakout_distance,
                "prior_range_high": prior_high,
                "prior_range_low": prior_low,
                "volume_ratio": volume_ratio,
                "granularity_seconds": float(granularity_seconds),
                "outcome_cluster": target_time.isoformat(),
                "state": f"confirmed_{direction}_range_breakout",
            },
            confidence=confidence,
            uncertainty={
                "lookback_bars": float(self.config.lookback_bars),
                "volume_ratio": volume_ratio,
                "receipt_lag_seconds": (
                    latest_event.available_at - latest_event.event_time
                ).total_seconds(),
            },
            evidence_event_ids=evidence,
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )
