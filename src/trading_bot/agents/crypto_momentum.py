from __future__ import annotations

import math
import statistics
import uuid
from dataclasses import dataclass
from datetime import timedelta

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.hypotheses import CRYPTO_INTRADAY_MOMENTUM_HYPOTHESIS
from trading_bot.agents.market_math import bar_values
from trading_bot.core.schemas import AssetClass, Forecast, ForecastKind, MarketEventType


@dataclass(frozen=True)
class CryptoIntradayMomentumConfig:
    lookback_bars: int = 8
    granularity_seconds: int = 15 * 60
    forecast_shrinkage: float = 0.25
    maximum_absolute_forecast: float = 0.01
    minimum_absolute_lookback_return: float = 0.001
    max_receipt_age: timedelta = timedelta(minutes=45)

    def __post_init__(self) -> None:
        if self.lookback_bars < 3:
            raise ValueError("momentum lookback must be at least three bars")
        if self.granularity_seconds <= 0:
            raise ValueError("momentum granularity must be positive")
        if not 0 < self.forecast_shrinkage <= 1:
            raise ValueError("momentum shrinkage must be between zero and one")
        if self.maximum_absolute_forecast <= 0:
            raise ValueError("maximum forecast must be positive")
        if self.minimum_absolute_lookback_return < 0:
            raise ValueError("minimum lookback return cannot be negative")


class CryptoIntradayMomentumSpecialist:
    agent_id = "crypto-intraday-momentum-baseline"
    model_version = "baseline-v1"
    supported_asset_classes = frozenset({AssetClass.CRYPTO})
    hypothesis = CRYPTO_INTRADAY_MOMENTUM_HYPOTHESIS

    def __init__(self, config: CryptoIntradayMomentumConfig | None = None) -> None:
        self.config = config or CryptoIntradayMomentumConfig()

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
            key = event.event_time
            existing = latest_by_bar.get(key)
            if existing is None or (event.available_at, event.event_id) > (
                existing[0].available_at,
                existing[0].event_id,
            ):
                latest_by_bar[key] = (event, values)
        ordered = sorted(
            latest_by_bar.values(),
            key=lambda item: (item[0].event_time, item[0].available_at, item[0].event_id),
        )
        if len(ordered) < self.config.lookback_bars:
            return None
        ordered = ordered[-self.config.lookback_bars :]
        latest_event = ordered[-1][0]
        if context.decision_time - latest_event.available_at > self.config.max_receipt_age:
            return None

        closes = [values[3] for _, values in ordered]
        lookback_return = closes[-1] / closes[0] - 1.0
        if abs(lookback_return) < self.config.minimum_absolute_lookback_return:
            return None
        average_log_return = math.log(closes[-1] / closes[0]) / (len(closes) - 1)
        predicted_return = math.expm1(
            average_log_return * self.config.forecast_shrinkage
        )
        predicted_return = max(
            -self.config.maximum_absolute_forecast,
            min(self.config.maximum_absolute_forecast, predicted_return),
        )
        if predicted_return == 0:
            return None

        target_time = latest_event.event_time + timedelta(
            seconds=self.config.granularity_seconds
        )
        if target_time <= context.decision_time:
            return None
        volumes = [values[4] for _, values in ordered]
        latest_volume = volumes[-1]
        median_volume = statistics.median(volumes[:-1])
        volume_ratio = latest_volume / median_volume if median_volume > 0 else 0.0
        direction = "up" if predicted_return > 0 else "down"
        confidence = min(
            0.45,
            0.20
            + min(abs(lookback_return), 0.05) * 3
            + min(volume_ratio, 2.0) * 0.025,
        )
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
                "reference_close": closes[-1],
                "direction": direction,
                "lookback_return": lookback_return,
                "average_log_return": average_log_return,
                "volume_ratio": volume_ratio,
                "granularity_seconds": float(self.config.granularity_seconds),
                "outcome_cluster": f"crypto-intraday:{target_time.isoformat()}",
                "state": f"intraday_momentum_{direction}",
            },
            confidence=confidence,
            uncertainty={
                "lookback_bars": float(self.config.lookback_bars),
                "lookback_return": abs(lookback_return),
                "receipt_lag_seconds": (
                    latest_event.available_at - latest_event.event_time
                ).total_seconds(),
            },
            evidence_event_ids=evidence,
            invalidation_conditions=self.hypothesis.invalidation_conditions,
        )
