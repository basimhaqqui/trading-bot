from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DatabaseLocation = str | Path


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
        return self._connection.execute(_postgres_sql(query), parameters or ())

    def executemany(self, query: str, parameters: Sequence[Sequence[Any]]):
        return self._connection.executemany(_postgres_sql(query), parameters)

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
    raw_connection = psycopg.connect(
        dsn,
        row_factory=_postgres_row_factory,
        prepare_threshold=None,
    )
    connection = PostgresConnection(raw_connection)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


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
