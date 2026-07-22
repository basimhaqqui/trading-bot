from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from trading_bot.core.schemas import Instrument, MarketEvent


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DiagnosticCode(StrEnum):
    EMPTY_BOOK = "empty_book"
    CROSSED_BOOK = "crossed_book"
    INVALID_VALUE = "invalid_value"
    STALE_EVENT = "stale_event"
    SOURCE_CLOCK_AHEAD = "source_clock_ahead"
    SEQUENCE_GAP = "sequence_gap"
    INDICATIVE_FEED = "indicative_feed"


@dataclass(frozen=True)
class DataQualityDiagnostic:
    code: DiagnosticCode
    severity: DiagnosticSeverity
    message: str
    instrument_id: str | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class CollectionBatch:
    venue: str
    instruments: tuple[Instrument, ...] = ()
    events: tuple[MarketEvent, ...] = ()
    diagnostics: tuple[DataQualityDiagnostic, ...] = ()
    cursor: str | None = None
    metadata: Mapping[str, str | int | float | bool] | None = None

    def __post_init__(self) -> None:
        if not self.venue:
            raise ValueError("venue is required")
        if any(item.venue != self.venue for item in self.instruments):
            raise ValueError("instrument venue does not match batch")
        if any(item.venue != self.venue for item in self.events):
            raise ValueError("event venue does not match batch")
