from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from trading_bot.core.schemas import AssetClass, Forecast, Instrument, MarketEvent


@dataclass(frozen=True)
class ReplayContext:
    decision_time: datetime
    instrument: Instrument
    events: tuple[MarketEvent, ...]
    related_instruments: tuple[Instrument, ...] = ()


class Specialist(Protocol):
    agent_id: str
    supported_asset_classes: frozenset[AssetClass]

    def evaluate(self, context: ReplayContext) -> Forecast | None:
        """Return a structured forecast using only context events."""
        ...
