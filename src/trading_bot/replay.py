from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from trading_bot.agents.base import ReplayContext, Specialist
from trading_bot.core.schemas import Forecast
from trading_bot.core.serialization import require_aware
from trading_bot.core.store import PointInTimeStore


class ReplayIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayResult:
    decision_times: int
    forecasts: tuple[Forecast, ...]


class ReplayEngine:
    def __init__(self, store: PointInTimeStore) -> None:
        self.store = store

    def run(
        self,
        specialist: Specialist,
        *,
        instrument_id: str,
        decision_times: Iterable[datetime],
        related_instrument_ids: Iterable[str] = (),
    ) -> ReplayResult:
        instrument = self.store.instrument(instrument_id)
        if instrument.asset_class not in specialist.supported_asset_classes:
            raise ValueError(
                f"{specialist.agent_id} does not support {instrument.asset_class.value}"
            )

        related_ids = tuple(related_instrument_ids)
        if instrument_id in related_ids or len(set(related_ids)) != len(related_ids):
            raise ValueError("related_instrument_ids must be unique and exclude the primary")
        related_instruments = tuple(self.store.instrument(item) for item in related_ids)

        forecasts: list[Forecast] = []
        count = 0
        previous_time: datetime | None = None
        for raw_time in decision_times:
            decision_time = require_aware(raw_time, "decision_time")
            if previous_time is not None and decision_time <= previous_time:
                raise ValueError("decision_times must be strictly increasing")
            previous_time = decision_time
            count += 1

            visible_events = self.store.events_available_at(
                decision_time,
                instrument_ids=(instrument_id, *related_ids),
            )
            visible_events.sort(
                key=lambda event: (
                    event.event_time,
                    event.available_at,
                    event.instrument_id,
                    event.sequence if event.sequence is not None else -1,
                    event.event_id,
                )
            )
            context = ReplayContext(
                decision_time, instrument, tuple(visible_events), related_instruments
            )
            forecast = specialist.evaluate(context)
            if forecast is None:
                continue
            self._validate_forecast(forecast, context, specialist)
            forecasts.append(forecast)

        return ReplayResult(count, tuple(forecasts))

    @staticmethod
    def _validate_forecast(
        forecast: Forecast, context: ReplayContext, specialist: Specialist
    ) -> None:
        if forecast.specialist_id != specialist.agent_id:
            raise ReplayIntegrityError("forecast specialist_id does not match running agent")
        if forecast.instrument_id != context.instrument.instrument_id:
            raise ReplayIntegrityError("forecast instrument does not match replay instrument")
        if forecast.generated_at != context.decision_time:
            raise ReplayIntegrityError("forecast generated_at must equal the replay decision time")
        available_ids = {event.event_id for event in context.events}
        unknown_evidence = set(forecast.evidence_event_ids) - available_ids
        if unknown_evidence:
            raise ReplayIntegrityError(
                f"forecast cites unavailable events: {sorted(unknown_evidence)}"
            )
