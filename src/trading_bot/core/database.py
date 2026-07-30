from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DatabaseLocation = str | Path
DEFAULT_POSTGRES_SCHEMA = "public"
_POSTGRES_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class DatabaseConfigurationError(ValueError):
    pass


class _PostgresRow(dict[str, Any]):
    """Rows compatible with the SQLite access patterns used by the research store."""

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)

    def __iter__(self):
        return iter(self.values())


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


def _postgres_row_factory(cursor: Any):
    if cursor.description is None:
        return lambda values: values
    names = [column.name for column in cursor.description]

    def make_row(values: Sequence[Any]) -> _PostgresRow:
        return _PostgresRow(zip(names, values))

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
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, query: str, parameters: Sequence[Any] | None = None):
        translated = _postgres_sql(query)
        if parameters is None:
            return self._connection.execute(translated)
        return self._connection.execute(translated, parameters)

    def executemany(self, query: str, parameters: Sequence[Sequence[Any]]):
        with self._connection.cursor() as cursor:
            cursor.executemany(_postgres_sql(query), parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

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
    connection_options: dict[str, Any] = {
        "row_factory": _postgres_row_factory,
        "prepare_threshold": None,
    }
    schema = postgres_schema()
    raw_connection = psycopg.connect(dsn, **connection_options)
    if schema != DEFAULT_POSTGRES_SCHEMA:
        # Neon transaction poolers reject libpq startup ``options``. Scope the
        # validated schema selection to this connection's transaction instead,
        # so a pooled backend cannot retain it after the research operation.
        raw_connection.execute(f"SET LOCAL search_path TO {schema}")
    connection = PostgresConnection(raw_connection)
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
