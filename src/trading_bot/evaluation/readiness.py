from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from trading_bot.core.schemas import AssetClass, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware
from trading_bot.core.store import PointInTimeStore


@dataclass(frozen=True)
class SpecialistReadiness:
    specialist: str
    ready: bool
    observations: int
    requirement: str


def data_readiness(
    store: PointInTimeStore,
    *,
    as_of: datetime,
    perpetual_symbol: str = "BIP-20DEC30-CDE",
    spot_symbol: str = "BTC-USD",
) -> tuple[SpecialistReadiness, ...]:
    as_of = require_aware(as_of, "as_of")
    events = store.events_available_at(as_of)
    with store.connect() as connection:
        instruments = connection.execute(
            "SELECT instrument_id, venue, symbol, asset_class FROM instruments"
        ).fetchall()
    instrument_classes = {
        row["instrument_id"]: AssetClass(row["asset_class"]) for row in instruments
    }
    symbol_ids = {
        (row["venue"], row["symbol"]): row["instrument_id"] for row in instruments
    }

    perpetual_id = symbol_ids.get(("coinbase", perpetual_symbol))
    spot_id = symbol_ids.get(("coinbase", spot_symbol))
    funding_periods = {
        str(event.payload.get("funding_time") or event.event_time.isoformat())
        for event in events
        if event.instrument_id == perpetual_id
        and event.event_type is MarketEventType.FUNDING
    }
    has_perpetual_book = _has_event(
        events, perpetual_id, MarketEventType.BOOK_SNAPSHOT
    )
    has_spot_book = _has_event(events, spot_id, MarketEventType.BOOK_SNAPSHOT)
    perpetual_ready = (
        perpetual_id is not None
        and spot_id is not None
        and len(funding_periods) >= 2
        and has_perpetual_book
        and has_spot_book
    )
    missing_perpetual: list[str] = []
    if len(funding_periods) < 2:
        missing_perpetual.append(f"{2 - len(funding_periods)} new funding period(s)")
    if not has_perpetual_book:
        missing_perpetual.append("perpetual book")
    if not has_spot_book:
        missing_perpetual.append("spot book")

    option_quotes = sum(
        1
        for event in events
        if instrument_classes.get(event.instrument_id) is AssetClass.OPTION
        and event.event_type is MarketEventType.QUOTE
    )
    equity_bars = sum(
        1
        for event in events
        if instrument_classes.get(event.instrument_id) is AssetClass.EQUITY
        and event.event_type is MarketEventType.BAR
    )
    options_ready = option_quotes >= 3 and equity_bars >= 6
    missing_options: list[str] = []
    if option_quotes < 3:
        missing_options.append(f"{3 - option_quotes} option quote snapshot(s)")
    if equity_bars < 6:
        missing_options.append(f"{6 - equity_bars} underlying daily bar(s)")

    settlements_by_instrument: dict[str, MarketEvent] = {}
    for event in events:
        if (
            event.event_type is MarketEventType.SETTLEMENT
            and instrument_classes.get(event.instrument_id) is AssetClass.PREDICTION
        ):
            existing = settlements_by_instrument.get(event.instrument_id)
            if existing is None or (event.available_at, event.event_id) < (
                existing.available_at,
                existing.event_id,
            ):
                settlements_by_instrument[event.instrument_id] = event
    books_by_instrument: dict[str, list[MarketEvent]] = defaultdict(list)
    for event in events:
        if (
            event.event_type is MarketEventType.BOOK_SNAPSHOT
            and instrument_classes.get(event.instrument_id) is AssetClass.PREDICTION
        ):
            books_by_instrument[event.instrument_id].append(event)
    labeled_books = sum(
        1
        for settlement in settlements_by_instrument.values()
        if any(
            book.available_at <= settlement.event_time
            for book in books_by_instrument[settlement.instrument_id]
        )
    )
    prediction_ready = labeled_books >= 5

    hourly_crypto_bars: dict[str, set[datetime]] = defaultdict(set)
    intraday_crypto_bars: dict[str, set[datetime]] = defaultdict(set)
    for event in events:
        if (
            event.event_type is MarketEventType.BAR
            and instrument_classes.get(event.instrument_id)
            in {AssetClass.CRYPTO, AssetClass.MEMECOIN}
        ):
            if event.payload.get("granularity_seconds") == 3600:
                hourly_crypto_bars[event.instrument_id].add(event.event_time)
            if (
                instrument_classes.get(event.instrument_id) is AssetClass.CRYPTO
                and event.payload.get("granularity_seconds") == 900
            ):
                intraday_crypto_bars[event.instrument_id].add(event.event_time)
    maximum_hourly_bars = max(
        (len(items) for items in hourly_crypto_bars.values()), default=0
    )
    maximum_intraday_bars = max(
        (len(items) for items in intraday_crypto_bars.values()), default=0
    )
    breakout_ready = maximum_hourly_bars >= 21
    intraday_ready = maximum_intraday_bars >= 8

    return (
        SpecialistReadiness(
            "perpetual-funding-basis",
            perpetual_ready,
            len(funding_periods),
            "ready" if perpetual_ready else ", ".join(missing_perpetual),
        ),
        SpecialistReadiness(
            "options-volatility",
            options_ready,
            option_quotes,
            "ready" if options_ready else ", ".join(missing_options),
        ),
        SpecialistReadiness(
            "prediction-calibration",
            prediction_ready,
            labeled_books,
            "ready"
            if prediction_ready
                else f"{5 - labeled_books} resolved market(s) with pre-settlement books",
        ),
        SpecialistReadiness(
            "crypto-range-breakout",
            breakout_ready,
            maximum_hourly_bars,
            "ready"
            if breakout_ready
            else f"{21 - maximum_hourly_bars} completed hourly bar(s)",
        ),
        SpecialistReadiness(
            "crypto-intraday-momentum",
            intraday_ready,
            maximum_intraday_bars,
            "ready"
            if intraday_ready
            else f"{8 - maximum_intraday_bars} completed fifteen-minute bar(s)",
        ),
    )


def _has_event(
    events: list[MarketEvent], instrument_id: str | None, event_type: MarketEventType
) -> bool:
    return instrument_id is not None and any(
        event.instrument_id == instrument_id and event.event_type is event_type
        for event in events
    )
