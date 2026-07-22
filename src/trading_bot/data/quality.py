from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from math import isfinite

from trading_bot.core.schemas import MarketEvent, MarketEventType
from trading_bot.data.schemas import (
    DataQualityDiagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
)


FRESHNESS_EVENT_TYPES = {
    MarketEventType.TRADE,
    MarketEventType.QUOTE,
    MarketEventType.BOOK_SNAPSHOT,
    MarketEventType.BOOK_DELTA,
    MarketEventType.BAR,
    MarketEventType.FUNDING,
    MarketEventType.OPEN_INTEREST,
    MarketEventType.LIQUIDATION,
    MarketEventType.ONCHAIN_STATE,
}


def inspect_events(
    events: tuple[MarketEvent, ...] | list[MarketEvent],
    *,
    stale_after: timedelta = timedelta(minutes=5),
) -> tuple[DataQualityDiagnostic, ...]:
    diagnostics: list[DataQualityDiagnostic] = []
    sequenced: dict[tuple[str, str, MarketEventType], list[MarketEvent]] = defaultdict(list)
    for event in events:
        if (
            event.event_type in FRESHNESS_EVENT_TYPES
            and event.available_at - event.event_time > stale_after
        ):
            diagnostics.append(
                DataQualityDiagnostic(
                    DiagnosticCode.STALE_EVENT,
                    DiagnosticSeverity.WARNING,
                    f"event arrived {event.available_at - event.event_time} after source time",
                    event.instrument_id,
                    event.event_id,
                )
            )
        if event.sequence is not None:
            sequenced[(event.venue, event.instrument_id, event.event_type)].append(event)
        if event.event_type in (MarketEventType.QUOTE, MarketEventType.BOOK_SNAPSHOT):
            diagnostics.extend(_inspect_market_depth(event))

    for items in sequenced.values():
        ordered = sorted(items, key=lambda item: (item.sequence or 0, item.event_time))
        for previous, current in zip(ordered, ordered[1:]):
            if current.sequence is not None and previous.sequence is not None:
                if current.sequence > previous.sequence + 1:
                    diagnostics.append(
                        DataQualityDiagnostic(
                            DiagnosticCode.SEQUENCE_GAP,
                            DiagnosticSeverity.WARNING,
                            f"sequence jumped from {previous.sequence} to {current.sequence}",
                            current.instrument_id,
                            current.event_id,
                        )
                    )
    return tuple(diagnostics)


def _number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _inspect_market_depth(event: MarketEvent) -> list[DataQualityDiagnostic]:
    result: list[DataQualityDiagnostic] = []
    payload = event.payload
    if "yes_bids" in payload or "no_bids" in payload:
        yes_bids = payload.get("yes_bids")
        no_bids = payload.get("no_bids")
        if not yes_bids and not no_bids:
            result.append(_diagnostic(event, DiagnosticCode.EMPTY_BOOK, "prediction book is empty"))
            return result
        try:
            yes_best = max(float(level[0]) for level in yes_bids or [])
            no_best = max(float(level[0]) for level in no_bids or [])
        except (TypeError, ValueError, IndexError):
            result.append(
                _diagnostic(
                    event,
                    DiagnosticCode.INVALID_VALUE,
                    "prediction book contains an invalid price level",
                    DiagnosticSeverity.ERROR,
                )
            )
            return result
        if yes_bids and no_bids and yes_best + no_best > 1.0 + 1e-9:
            result.append(
                _diagnostic(
                    event,
                    DiagnosticCode.CROSSED_BOOK,
                    "yes and no bids imply a crossed binary book",
                    DiagnosticSeverity.ERROR,
                )
            )
        return result

    bid = _number(payload.get("bid_price"))
    ask = _number(payload.get("ask_price"))
    if bid is None and ask is None:
        result.append(_diagnostic(event, DiagnosticCode.EMPTY_BOOK, "quote has no bid or ask"))
    elif bid is not None and ask is not None and bid >= ask:
        result.append(
            _diagnostic(
                event,
                DiagnosticCode.CROSSED_BOOK,
                f"bid {bid} is not below ask {ask}",
                DiagnosticSeverity.ERROR,
            )
        )
    if (bid is not None and bid < 0) or (ask is not None and ask < 0):
        result.append(
            _diagnostic(
                event,
                DiagnosticCode.INVALID_VALUE,
                "quote contains a negative price",
                DiagnosticSeverity.ERROR,
            )
        )
    return result


def _diagnostic(
    event: MarketEvent,
    code: DiagnosticCode,
    message: str,
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
) -> DataQualityDiagnostic:
    return DataQualityDiagnostic(code, severity, message, event.instrument_id, event.event_id)
