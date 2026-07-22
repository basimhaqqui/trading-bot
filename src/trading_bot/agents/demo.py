from __future__ import annotations

import uuid
from datetime import timedelta

from trading_bot.agents.base import ReplayContext
from trading_bot.core.schemas import AssetClass, Forecast, ForecastKind, MarketEventType


class DemoRegimeSpecialist:
    """Small deterministic specialist used only to verify replay integrity."""

    agent_id = "demo-equity-regime"
    supported_asset_classes = frozenset({AssetClass.EQUITY})

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        bars = [event for event in context.events if event.event_type is MarketEventType.BAR]
        if len(bars) < 2:
            return None
        previous = float(bars[-2].payload["close"])
        latest = float(bars[-1].payload["close"])
        regime = "up" if latest > previous else "down" if latest < previous else "flat"
        return Forecast(
            forecast_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent_id}:{context.decision_time}")),
            specialist_id=self.agent_id,
            model_version="demo-v1",
            instrument_id=context.instrument.instrument_id,
            kind=ForecastKind.REGIME,
            generated_at=context.decision_time,
            valid_until=context.decision_time + timedelta(days=1),
            values={"regime": regime, "last_available_close": latest},
            confidence=0.5,
            uncertainty={"demo_only": 1.0},
            evidence_event_ids=(bars[-2].event_id, bars[-1].event_id),
            invalidation_conditions=("new bar becomes available",),
        )
