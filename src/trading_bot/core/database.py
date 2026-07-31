from __future__ import annotations

import os
import re
import sqlite3
from contextvars import ContextVar
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DatabaseLocation = str | Path
DEFAULT_POSTGRES_SCHEMA = "public"
_POSTGRES_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_POSTGRES_EGRESS_METER: ContextVar["PostgresEgressMeter | None"] = ContextVar(
    "postgres_egress_meter", default=None
)


class DatabaseConfigurationError(ValueError):
    pass


@dataclass
class PostgresEgressMeter:
    """Conservative client-side estimate of PostgreSQL result-set egress."""

    bytes_received: int = 0
    rows_received: int = 0

    def record_row(self, names: Sequence[str], values: Sequence[Any]) -> None:
        # Add fixed framing overhead per value and row so the report never
        # presents decoded payload bytes as exact wire accounting.
        self.rows_received += 1
        self.bytes_received += 48
        for name, value in zip(names, values):
            self.bytes_received += len(name.encode("utf-8")) + 16
            if value is None:
                self.bytes_received += 4
            elif isinstance(value, bytes):
                self.bytes_received += len(value)
            else:
                self.bytes_received += len(str(value).encode("utf-8"))


@contextmanager
def measure_postgres_egress() -> Iterator[PostgresEgressMeter]:
    """Measure PostgreSQL rows decoded during one command without altering storage."""
    meter = PostgresEgressMeter()
    token = _POSTGRES_EGRESS_METER.set(meter)
    try:
        yield meter
    finally:
        _POSTGRES_EGRESS_METER.reset(token)


class _PostgresRow(dict[str, Any]):
    """Rows compatible with the SQLite access patterns used by the research store."""

    def __init__(self, names: Sequence[str], values: Sequence[Any]) -> None:
        # Keep the full positional result separately. A mapping necessarily collapses
        # duplicate column names, but SQLite-style tuple unpacking must retain every
        # selected value (for example two different ``available_at`` columns).
        super().__init__(zip(names, values))
        self._values = tuple(values)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)

    def __iter__(self):
        return iter(self._values)


def is_postgres_location(location: DatabaseLocation) -> bool:
    return isinstance(location, str) and location.startswith(("postgres://", "postgresql://"))


def database_display_name(location: DatabaseLocation) -> str:
    return "PostgreSQL" if is_postgres_location(location) else str(location)


def postgres_schema() -> str:
    schema = os.getenv("TRADING_DB_SCHEMA", DEFAULT_POSTGRES_SCHEMA)
    if not _POSTGRES_IDENTIFIER.fullmatch(schema):
        raise DatabaseConfigurationError("shadow database schema name is invalid")
    return schema


def _validate_postgres_location(location: str) -> None:
    parsed = urlsplit(location)
    host = parsed.hostname or ""
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise DatabaseConfigurationError("shadow database must use a PostgreSQL URL")
    if not parsed.username or not parsed.password or not parsed.path or parsed.path == "/":
        raise DatabaseConfigurationError("shadow database URL is incomplete")
    if not host.endswith(".neon.tech") or "-pooler." not in host:
        raise DatabaseConfigurationError("shadow database must use the configured pooled Neon endpoint")
    if "sslmode=" not in parsed.query:
        raise DatabaseConfigurationError("shadow database URL must require TLS")


def _postgres_row_factory(cursor: Any, meter: PostgresEgressMeter | None = None):
    if cursor.description is None:
        return lambda values: values
    names = [column.name for column in cursor.description]

    def make_row(values: Sequence[Any]) -> _PostgresRow:
        if meter is not None:
            meter.record_row(names, values)
        return _PostgresRow(names, values)

    return make_row


_INSERT_OR_IGNORE = re.compile(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)
_JSON_EXTRACT = re.compile(
    r"json_extract\((?P<column>[A-Za-z_][A-Za-z0-9_.]*),\s*'\$\.(?P<path>[A-Za-z0-9_.]+)'\)",
    re.IGNORECASE,
)


def _postgres_sql(query: str) -> str:
    """Translate the narrow SQLite SQL subset used by this project to PostgreSQL."""
    translated = _INSERT_OR_IGNORE.sub("INSERT INTO", query)
    if translated != query:
        translated = translated.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    translated = _JSON_EXTRACT.sub(
        lambda match: (
            f"({match.group('column')}::jsonb #>> "
            f"'{{{match.group('path').replace('.', ',')}}}')"
        ),
        translated,
    )
    return translated.replace("?", "%s")


class PostgresConnection:
    def __init__(self, connection: Any, schema: str = DEFAULT_POSTGRES_SCHEMA) -> None:
        self._connection = connection
        self._schema = schema
        self._schema_selected = schema == DEFAULT_POSTGRES_SCHEMA

    def _ensure_schema(self) -> None:
        if self._schema_selected:
            return
        self._connection.execute(f"SET LOCAL search_path TO {self._schema}")
        self._schema_selected = True

    def execute(self, query: str, parameters: Sequence[Any] | None = None):
        self._ensure_schema()
        translated = _postgres_sql(query)
        if parameters is None:
            return self._connection.execute(translated)
        return self._connection.execute(translated, parameters)

    def executemany(self, query: str, parameters: Sequence[Sequence[Any]]):
        self._ensure_schema()
        with self._connection.cursor() as cursor:
            cursor.executemany(_postgres_sql(query), parameters)

    def commit(self) -> None:
        self._connection.commit()
        self._schema_selected = self._schema == DEFAULT_POSTGRES_SCHEMA

    def rollback(self) -> None:
        self._connection.rollback()
        self._schema_selected = self._schema == DEFAULT_POSTGRES_SCHEMA

    def close(self) -> None:
        self._connection.close()


@contextmanager
def connect_database(location: DatabaseLocation) -> Iterator[sqlite3.Connection | PostgresConnection]:
    if not is_postgres_location(location):
        path = Path(location)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return

    dsn = str(location)
    _validate_postgres_location(dsn)
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError("PostgreSQL persistence requires psycopg") from exc
    schema = postgres_schema()
    meter = _POSTGRES_EGRESS_METER.get()
    raw_connection = psycopg.connect(
        dsn,
        row_factory=lambda cursor: _postgres_row_factory(cursor, meter),
        prepare_threshold=None,
    )
    connection = PostgresConnection(raw_connection, schema)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def bootstrap_postgres_schema(location: DatabaseLocation) -> None:
    if not is_postgres_location(location):
        raise ValueError("schema bootstrap requires PostgreSQL")
    dsn = str(location)
    _validate_postgres_location(dsn)
    schema = postgres_schema()
    if schema == DEFAULT_POSTGRES_SCHEMA:
        return
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError("PostgreSQL persistence requires psycopg") from exc
    with psycopg.connect(dsn, prepare_threshold=None) as connection:
        connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


def initialize_schema(
    connection: sqlite3.Connection | PostgresConnection,
    sqlite_schema: str,
    *,
    append_only_tables: Sequence[str] = (),
) -> None:
    if not isinstance(connection, PostgresConnection):
        connection.executescript(sqlite_schema)
        return
    filtered = re.sub(r"^\s*PRAGMA[^;]*;", "", sqlite_schema, flags=re.MULTILINE)
    filtered = re.sub(
        r"CREATE TRIGGER IF NOT EXISTS .*?END;",
        "",
        filtered,
        flags=re.DOTALL,
    )
    for statement in filtered.split(";"):
        if statement.strip():
            connection.execute(statement)
    if not append_only_tables:
        return
    connection.execute(
        """
        CREATE OR REPLACE FUNCTION trading_bot_reject_append_only_change()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'append-only table % cannot be modified', TG_TABLE_NAME;
        END;
        $$
        """
    )
    for table in append_only_tables:
        trigger = f"{table}_append_only_guard"
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        connection.execute(
            f"""
            CREATE TRIGGER {trigger}
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION trading_bot_reject_append_only_change()
            """
        )


def postgres_integrity_ok(location: DatabaseLocation) -> bool:
    if not is_postgres_location(location):
        with connect_database(location) as connection:
            return connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with connect_database(location) as connection:
        connection.execute("SELECT 1").fetchone()
    return True
