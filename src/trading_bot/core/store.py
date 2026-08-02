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

CREATE INDEX IF NOT EXISTS idx_events_instrument_type_available
ON market_events(instrument_id, event_type, available_at, event_time, event_id);

CREATE TRIGGER IF NOT EXISTS market_events_no_update
BEFORE UPDATE ON market_events BEGIN
    SELECT RAISE(ABORT, 'market_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS market_events_no_delete
BEFORE DELETE ON market_events BEGIN
    SELECT RAISE(ABORT, 'market_events is append-only');
END;
"""

BATCH_LOOKUP_SIZE = 900


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
        with self.connect() as connection:
            existing_instruments = self._rows_by_identifier(
                connection,
                "instruments",
                "instrument_id",
                (item.instrument_id for item in batch.instruments),
            )
            instrument_rows: list[tuple[object, ...]] = []
            batch_instruments: dict[str, str] = {}
            for instrument in batch.instruments:
                digest = sha256_digest(instrument)
                prior_digest = batch_instruments.get(instrument.instrument_id)
                if prior_digest is not None:
                    if prior_digest != digest:
                        raise EventConflictError(
                            f"instrument {instrument.instrument_id} already exists with different contents"
                        )
                    continue
                batch_instruments[instrument.instrument_id] = digest
                existing = existing_instruments.get(instrument.instrument_id)
                if existing and existing["digest"] != digest:
                    raise EventConflictError(
                        f"instrument {instrument.instrument_id} already exists with different contents"
                    )
                if existing is None:
                    instrument_rows.append(
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
                        )
                    )
            if instrument_rows:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO instruments (
                        instrument_id, venue, symbol, asset_class, quote_currency, multiplier,
                        active_from, active_until, expiry, settlement, metadata_json, digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    instrument_rows,
                )

            existing_events = self._rows_by_identifier(
                connection,
                "market_events",
                "event_id",
                (item.event_id for item in batch.events),
            )
            event_rows: list[tuple[object, ...]] = []
            batch_events: dict[str, MarketEvent] = {}
            for event in batch.events:
                prior = batch_events.get(event.event_id)
                if prior is not None:
                    if not self._same_market_events(prior, event):
                        raise EventConflictError(
                            f"event {event.event_id} already exists with different contents"
                        )
                    continue
                batch_events[event.event_id] = event
                existing = existing_events.get(event.event_id)
                if existing:
                    if self._same_market_observation(existing, event):
                        continue
                    if existing["digest"] != event.digest:
                        raise EventConflictError(
                            f"event {event.event_id} already exists with different contents"
                        )
                    continue
                event_rows.append(
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
                    )
                )
            if event_rows:
                connection.executemany(
                    """
                    INSERT INTO market_events (
                        event_id, event_type, venue, instrument_id, event_time, available_at,
                        source, sequence, ingested_at, payload_json, digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_rows,
                )
        return len(batch.instruments), len(event_rows)

    def events_available_at(
        self,
        as_of: datetime,
        *,
        instrument_id: str | None = None,
        instrument_ids: Iterable[str] | None = None,
        event_type: MarketEventType | None = None,
        available_since: datetime | None = None,
    ) -> list[MarketEvent]:
        as_of = require_aware(as_of, "as_of")
        if available_since is not None:
            available_since = require_aware(available_since, "available_since")
            if available_since > as_of:
                raise ValueError("available_since cannot be after as_of")
        selected_ids = tuple(dict.fromkeys(instrument_ids or ()))
        if instrument_id is not None and instrument_ids is not None:
            raise ValueError("instrument_id and instrument_ids are mutually exclusive")
        if instrument_ids is not None and not selected_ids:
            return []
        clauses = ["available_at <= ?"]
        parameters: list[str] = [as_of.isoformat()]
        if available_since is not None:
            clauses.append("available_at >= ?")
            parameters.append(available_since.isoformat())
        if instrument_id:
            clauses.append("instrument_id = ?")
            parameters.append(instrument_id)
        if event_type:
            clauses.append("event_type = ?")
            parameters.append(event_type.value)
        chunks = (
            tuple(
                selected_ids[offset : offset + BATCH_LOOKUP_SIZE]
                for offset in range(0, len(selected_ids), BATCH_LOOKUP_SIZE)
            )
            if selected_ids
            else ((),)
        )
        rows: list[Any] = []
        with self.connect() as connection:
            for chunk in chunks:
                chunk_clauses = list(clauses)
                chunk_parameters = list(parameters)
                if chunk:
                    placeholders = ", ".join("?" for _ in chunk)
                    chunk_clauses.append(f"instrument_id IN ({placeholders})")
                    chunk_parameters.extend(chunk)
                query = f"""
                    SELECT * FROM market_events
                    WHERE {' AND '.join(chunk_clauses)}
                    ORDER BY event_time, COALESCE(sequence, -1), event_id
                """
                rows.extend(connection.execute(query, chunk_parameters).fetchall())
        events = [self._event_from_row(row) for row in rows]
        events.sort(
            key=lambda item: (
                item.event_time,
                item.sequence if item.sequence is not None else -1,
                item.event_id,
            )
        )
        return events

    def has_events(self, event_ids: Iterable[str]) -> bool:
        """Check canonical evidence IDs without transferring their payloads."""
        selected = tuple(dict.fromkeys(event_ids))
        if not selected:
            return True
        found: set[str] = set()
        with self.connect() as connection:
            for offset in range(0, len(selected), BATCH_LOOKUP_SIZE):
                chunk = selected[offset : offset + BATCH_LOOKUP_SIZE]
                placeholders = ", ".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT event_id FROM market_events WHERE event_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                found.update(str(row["event_id"]) for row in rows)
        return len(found) == len(selected)

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

    @staticmethod
    def _same_market_events(left: MarketEvent, right: MarketEvent) -> bool:
        return (
            left.event_type is right.event_type
            and left.venue == right.venue
            and left.instrument_id == right.instrument_id
            and left.event_time == right.event_time
            and left.source == right.source
            and left.sequence == right.sequence
            and canonical_json(left.payload) == canonical_json(right.payload)
        )

    @staticmethod
    def _rows_by_identifier(
        connection: Any,
        table: str,
        identifier: str,
        values: Iterable[str],
    ) -> dict[str, Any]:
        selected = tuple(dict.fromkeys(values))
        rows: dict[str, Any] = {}
        for offset in range(0, len(selected), BATCH_LOOKUP_SIZE):
            chunk = selected[offset : offset + BATCH_LOOKUP_SIZE]
            if not chunk:
                continue
            placeholders = ", ".join("?" for _ in chunk)
            query = f"SELECT * FROM {table} WHERE {identifier} IN ({placeholders})"
            for row in connection.execute(query, chunk).fetchall():
                rows[row[identifier]] = row
        return rows

    def instrument(self, instrument_id: str) -> Instrument:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM instruments WHERE instrument_id = ?", (instrument_id,)
            ).fetchone()
        if row is None:
            raise KeyError(instrument_id)
        return self._instrument_from_row(row)

    def instruments(
        self,
        *,
        asset_class: AssetClass | None = None,
        venue: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> list[Instrument]:
        selected_symbols = tuple(dict.fromkeys(symbols or ()))
        if symbols is not None and not selected_symbols:
            return []
        query = "SELECT * FROM instruments"
        clauses: list[str] = []
        parameters: list[str] = []
        if asset_class is not None:
            clauses.append("asset_class = ?")
            parameters.append(asset_class.value)
        if venue is not None:
            clauses.append("venue = ?")
            parameters.append(venue)
        if selected_symbols:
            placeholders = ", ".join("?" for _ in selected_symbols)
            clauses.append(f"symbol IN ({placeholders})")
            parameters.extend(selected_symbols)
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
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
