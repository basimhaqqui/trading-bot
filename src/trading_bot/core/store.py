from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from trading_bot.core.database import (
    DatabaseLocation,
    PostgresConnection,
    connect_database,
    initialize_schema,
)
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import canonical_json, parse_datetime, require_aware


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instruments (
    instrument_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    multiplier REAL NOT NULL CHECK (multiplier > 0),
    active_from TEXT,
    active_until TEXT,
    expiry TEXT,
    settlement TEXT,
    metadata_json TEXT NOT NULL,
    digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    venue TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    event_time TEXT NOT NULL,
    available_at TEXT NOT NULL,
    source TEXT NOT NULL,
    sequence INTEGER,
    ingested_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_events_replay
ON market_events(available_at, event_time, instrument_id, event_type);

CREATE TRIGGER IF NOT EXISTS market_events_no_update
BEFORE UPDATE ON market_events BEGIN
    SELECT RAISE(ABORT, 'market_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS market_events_no_delete
BEFORE DELETE ON market_events BEGIN
    SELECT RAISE(ABORT, 'market_events is append-only');
END;
"""


class EventConflictError(RuntimeError):
    pass


class PointInTimeStore:
    def __init__(self, path: str | Path) -> None:
        self.path: DatabaseLocation = path

    @contextmanager
    def connect(self) -> Iterator[Any]:
        with connect_database(self.path) as connection:
            yield connection

    def initialize(self) -> None:
        with self.connect() as connection:
            initialize_schema(
                connection,
                SCHEMA,
                append_only_tables=("market_events",),
            )

    def register_instrument(self, instrument: Instrument) -> None:
        from trading_bot.core.serialization import sha256_digest

        digest = sha256_digest(instrument)
        values = (
            instrument.instrument_id,
            instrument.venue,
            instrument.symbol,
            instrument.asset_class.value,
            instrument.quote_currency,
            instrument.multiplier,
            instrument.active_from.isoformat() if instrument.active_from else None,
            instrument.active_until.isoformat() if instrument.active_until else None,
            instrument.expiry.isoformat() if instrument.expiry else None,
            instrument.settlement,
            canonical_json(instrument.metadata),
            digest,
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT digest FROM instruments WHERE instrument_id = ?", (instrument.instrument_id,)
            ).fetchone()
            if existing and existing["digest"] != digest:
                raise EventConflictError(
                    f"instrument {instrument.instrument_id} already exists with different contents"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO instruments (
                    instrument_id, venue, symbol, asset_class, quote_currency, multiplier,
                    active_from, active_until, expiry, settlement, metadata_json, digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def append_event(self, event: MarketEvent) -> bool:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM market_events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            if existing:
                if self._same_market_observation(existing, event):
                    return False
                if existing["digest"] != event.digest:
                    raise EventConflictError(
                        f"event {event.event_id} already exists with different contents"
                    )
                return False
            connection.execute(
                """
                INSERT INTO market_events (
                    event_id, event_type, venue, instrument_id, event_time, available_at,
                    source, sequence, ingested_at, payload_json, digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type.value,
                    event.venue,
                    event.instrument_id,
                    event.event_time.isoformat(),
                    event.available_at.isoformat(),
                    event.source,
                    event.sequence,
                    event.ingested_at.isoformat(),
                    canonical_json(event.payload),
                    event.digest,
                ),
            )
            return True

    def append_batch(self, batch: object) -> tuple[int, int]:
        from trading_bot.data.schemas import CollectionBatch
        from trading_bot.core.serialization import sha256_digest

        if not isinstance(batch, CollectionBatch):
            raise TypeError("batch must be a CollectionBatch")
        inserted = 0
        with self.connect() as connection:
            for instrument in batch.instruments:
                digest = sha256_digest(instrument)
                existing = connection.execute(
                    "SELECT digest FROM instruments WHERE instrument_id = ?",
                    (instrument.instrument_id,),
                ).fetchone()
                if existing and existing["digest"] != digest:
                    raise EventConflictError(
                        f"instrument {instrument.instrument_id} already exists with different contents"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO instruments (
                        instrument_id, venue, symbol, asset_class, quote_currency, multiplier,
                        active_from, active_until, expiry, settlement, metadata_json, digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        instrument.instrument_id,
                        instrument.venue,
                        instrument.symbol,
                        instrument.asset_class.value,
                        instrument.quote_currency,
                        instrument.multiplier,
                        instrument.active_from.isoformat() if instrument.active_from else None,
                        instrument.active_until.isoformat() if instrument.active_until else None,
                        instrument.expiry.isoformat() if instrument.expiry else None,
                        instrument.settlement,
                        canonical_json(instrument.metadata),
                        digest,
                    ),
                )
            for event in batch.events:
                existing = connection.execute(
                    "SELECT * FROM market_events WHERE event_id = ?", (event.event_id,)
                ).fetchone()
                if existing:
                    if self._same_market_observation(existing, event):
                        continue
                    if existing["digest"] != event.digest:
                        raise EventConflictError(
                            f"event {event.event_id} already exists with different contents"
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO market_events (
                        event_id, event_type, venue, instrument_id, event_time, available_at,
                        source, sequence, ingested_at, payload_json, digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.event_type.value,
                        event.venue,
                        event.instrument_id,
                        event.event_time.isoformat(),
                        event.available_at.isoformat(),
                        event.source,
                        event.sequence,
                        event.ingested_at.isoformat(),
                        canonical_json(event.payload),
                        event.digest,
                    ),
                )
                inserted += 1
        return len(batch.instruments), inserted

    def events_available_at(
        self,
        as_of: datetime,
        *,
        instrument_id: str | None = None,
        instrument_ids: Iterable[str] | None = None,
        event_type: MarketEventType | None = None,
    ) -> list[MarketEvent]:
        as_of = require_aware(as_of, "as_of")
        selected_ids = tuple(dict.fromkeys(instrument_ids or ()))
        if instrument_id is not None and instrument_ids is not None:
            raise ValueError("instrument_id and instrument_ids are mutually exclusive")
        if instrument_ids is not None and not selected_ids:
            return []
        clauses = ["available_at <= ?"]
        parameters: list[str] = [as_of.isoformat()]
        if instrument_id:
            clauses.append("instrument_id = ?")
            parameters.append(instrument_id)
        elif selected_ids:
            placeholders = ", ".join("?" for _ in selected_ids)
            clauses.append(f"instrument_id IN ({placeholders})")
            parameters.extend(selected_ids)
        if event_type:
            clauses.append("event_type = ?")
            parameters.append(event_type.value)
        query = f"""
            SELECT * FROM market_events
            WHERE {' AND '.join(clauses)}
            ORDER BY event_time, COALESCE(sequence, -1), event_id
        """
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._event_from_row(row) for row in rows]

    def latest_events(
        self,
        as_of: datetime,
        *,
        instrument_id: str,
        event_type: MarketEventType,
        limit: int,
    ) -> list[MarketEvent]:
        if limit < 1:
            raise ValueError("limit must be positive")
        events = self.events_available_at(
            as_of, instrument_id=instrument_id, event_type=event_type
        )
        return events[-limit:]

    @staticmethod
    def _event_from_row(row: Any) -> MarketEvent:
        return MarketEvent(
            event_id=row["event_id"],
            event_type=MarketEventType(row["event_type"]),
            venue=row["venue"],
            instrument_id=row["instrument_id"],
            event_time=parse_datetime(row["event_time"]),
            available_at=parse_datetime(row["available_at"]),
            source=row["source"],
            sequence=row["sequence"],
            ingested_at=parse_datetime(row["ingested_at"]),
            payload=json.loads(row["payload_json"]),
        )

    @staticmethod
    def _same_market_observation(row: Any, event: MarketEvent) -> bool:
        return (
            row["event_type"] == event.event_type.value
            and row["venue"] == event.venue
            and row["instrument_id"] == event.instrument_id
            and row["event_time"] == event.event_time.isoformat()
            and row["source"] == event.source
            and row["sequence"] == event.sequence
            and row["payload_json"] == canonical_json(event.payload)
        )

    def instrument(self, instrument_id: str) -> Instrument:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM instruments WHERE instrument_id = ?", (instrument_id,)
            ).fetchone()
        if row is None:
            raise KeyError(instrument_id)
        return self._instrument_from_row(row)

    def instruments(self, *, asset_class: AssetClass | None = None) -> list[Instrument]:
        query = "SELECT * FROM instruments"
        parameters: tuple[str, ...] = ()
        if asset_class is not None:
            query += " WHERE asset_class = ?"
            parameters = (asset_class.value,)
        query += " ORDER BY instrument_id"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._instrument_from_row(row) for row in rows]

    def event(self, event_id: str) -> MarketEvent:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM market_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._event_from_row(row)

    @staticmethod
    def _instrument_from_row(row: Any) -> Instrument:
        return Instrument(
            instrument_id=row["instrument_id"],
            venue=row["venue"],
            symbol=row["symbol"],
            asset_class=AssetClass(row["asset_class"]),
            quote_currency=row["quote_currency"],
            multiplier=row["multiplier"],
            active_from=parse_datetime(row["active_from"]) if row["active_from"] else None,
            active_until=parse_datetime(row["active_until"]) if row["active_until"] else None,
            expiry=parse_datetime(row["expiry"]) if row["expiry"] else None,
            settlement=row["settlement"],
            metadata=json.loads(row["metadata_json"]),
        )
