from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from typing import Iterable, Mapping

from trading_bot.core.schemas import MarketEvent, MarketEventType


def finite_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def bar_values(
    payload: object,
) -> tuple[float, float, float, float, float, int] | None:
    if not isinstance(payload, Mapping):
        return None
    values = tuple(
        finite_float(payload.get(name))
        for name in ("open", "high", "low", "close", "volume", "granularity_seconds")
    )
    if any(value is None for value in values):
        return None
    open_price, high, low, close, volume, raw_granularity = (
        float(value) for value in values
    )
    granularity = int(raw_granularity)
    if (
        min(open_price, high, low, close) <= 0
        or volume < 0
        or granularity <= 0
        or granularity != raw_granularity
        or low > min(open_price, close)
        or high < max(open_price, close)
    ):
        return None
    return open_price, high, low, close, volume, granularity


def recent_events(
    events: Iterable[MarketEvent],
    *,
    instrument_id: str,
    event_type: MarketEventType,
    decision_time: datetime,
    max_age: timedelta | None = None,
) -> list[MarketEvent]:
    result = [
        event
        for event in events
        if event.instrument_id == instrument_id and event.event_type is event_type
    ]
    result.sort(key=lambda event: (event.available_at, event.event_time, event.event_id))
    if max_age is not None:
        result = [event for event in result if decision_time - event.available_at <= max_age]
    return result


def executable_quote(event: MarketEvent) -> tuple[float, float, float, float] | None:
    bid = finite_float(event.payload.get("bid_price"))
    ask = finite_float(event.payload.get("ask_price"))
    if bid is None or ask is None or bid < 0 or ask <= bid:
        return None
    midpoint = (bid + ask) / 2
    if midpoint <= 0:
        return None
    spread_bps = (ask - bid) / midpoint * 10_000
    return bid, ask, midpoint, spread_bps


def prediction_book(event: MarketEvent) -> tuple[float, float, float, float] | None:
    return prediction_book_payload(event.payload)


def prediction_book_payload(
    payload: Mapping[str, object],
) -> tuple[float, float, float, float] | None:
    yes_levels = payload.get("yes_bids")
    no_levels = payload.get("no_bids")
    if not isinstance(yes_levels, list) or not isinstance(no_levels, list):
        return None
    try:
        yes_bid = max(float(level[0]) for level in yes_levels)
        no_bid = max(float(level[0]) for level in no_levels)
    except (TypeError, ValueError, IndexError):
        return None
    yes_ask = 1.0 - no_bid
    if not 0 <= yes_bid < yes_ask <= 1:
        return None
    midpoint = (yes_bid + yes_ask) / 2
    return yes_bid, yes_ask, midpoint, yes_ask - yes_bid
