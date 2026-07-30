from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from trading_bot.core.database import DatabaseLocation, connect_database, is_postgres_location


# Copy only immutable research and reconciliation history. The mutable paper-control
# row is recreated in its default locked state by initialization.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "instruments": ("instrument_id", "venue", "symbol", "asset_class", "quote_currency", "multiplier", "active_from", "active_until", "expiry", "settlement", "metadata_json", "digest"),
    "market_events": ("event_id", "event_type", "venue", "instrument_id", "event_time", "available_at", "source", "sequence", "ingested_at", "payload_json", "digest"),
    "audit_records": ("record_type", "record_id", "occurred_at", "recorded_at", "payload_json", "digest"),
    "hypotheses": ("hypothesis_id", "family", "market", "proposed_at", "record_json"),
    "experiments": ("experiment_id", "hypothesis_id", "family", "status", "config_json", "code_version", "data_cutoff", "created_at", "completed_at", "metrics_json", "notes"),
    "ingestion_runs": ("run_id", "plan_name", "job_id", "venue", "dataset", "status", "started_at", "finished_at", "record_json", "digest"),
    "paper_control_events": ("event_id", "action", "reason", "occurred_at", "payload_json", "digest"),
    "paper_account_snapshots": ("snapshot_id", "observed_at", "payload_json", "digest"),
    "paper_order_events": ("event_id", "order_id", "client_order_id", "status", "observed_at", "remote_updated_at", "payload_json", "digest"),
}

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "instruments": ("instrument_id",),
    "market_events": ("event_id",),
    "audit_records": ("record_type", "record_id"),
    "hypotheses": ("hypothesis_id",),
    "experiments": ("experiment_id",),
    "ingestion_runs": ("run_id",),
    "paper_control_events": ("event_id",),
    "paper_account_snapshots": ("snapshot_id",),
    "paper_order_events": ("event_id",),
}


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError("legacy SQLite source does not exist")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _require_locked_paper_control(source: sqlite3.Connection) -> None:
    row = source.execute(
        "SELECT enabled, kill_switch_active FROM paper_execution_control "
        "WHERE environment = 'paper'"
    ).fetchone()
    if row is not None and (bool(row["enabled"]) or not bool(row["kill_switch_active"])):
        raise RuntimeError("refusing migration from an unlocked paper-control source")


def _verified_overlap(
    table: str,
    source_rows: list[tuple[object, ...]],
    target_rows: list[tuple[object, ...]],
) -> int:
    columns = TABLE_COLUMNS[table]
    key_indexes = tuple(columns.index(column) for column in PRIMARY_KEYS[table])
    source_by_key = {
        tuple(row[index] for index in key_indexes): row for row in source_rows
    }
    if len(source_by_key) != len(source_rows):
        raise RuntimeError(f"legacy SQLite source contains duplicate keys in {table}")
    overlap = 0
    for target_row in target_rows:
        key = tuple(target_row[index] for index in key_indexes)
        source_row = source_by_key.get(key)
        if source_row is None:
            continue
        if source_row != target_row:
            raise RuntimeError(
                f"refusing migration with conflicting rows already present in {table}"
            )
        overlap += 1
    return overlap


def migrate_sqlite_to_postgres(source_path: Path, destination: DatabaseLocation) -> tuple[str, dict[str, int]]:
    """Merge a locked legacy SQLite store into PostgreSQL without rewriting history."""
    if not is_postgres_location(destination):
        raise ValueError("SQLite migration target must be PostgreSQL")
    source = _legacy_connection(source_path)
    try:
        tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = {
            "instruments",
            "market_events",
            "audit_records",
            "ingestion_runs",
            "paper_execution_control",
        } - tables
        if missing:
            raise RuntimeError(f"legacy SQLite source is missing required tables: {', '.join(sorted(missing))}")
        _require_locked_paper_control(source)
        with connect_database(destination) as target:
            control = target.execute(
                "SELECT enabled, kill_switch_active FROM paper_execution_control "
                "WHERE environment = 'paper'"
            ).fetchone()
            if control is None or bool(control["enabled"]) or not bool(control["kill_switch_active"]):
                raise RuntimeError("refusing migration into an unlocked PostgreSQL target")
            expected_counts: dict[str, int] = {}
            for table, columns in TABLE_COLUMNS.items():
                source_rows = (
                    source.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
                    if table in tables
                    else []
                )
                source_values = [
                    tuple(row[column] for column in columns) for row in source_rows
                ]
                target_rows = target.execute(
                    f"SELECT {', '.join(columns)} FROM {table}"
                ).fetchall()
                target_values = [
                    tuple(row[column] for column in columns) for row in target_rows
                ]
                overlap = _verified_overlap(table, source_values, target_values)
                expected_counts[table] = (
                    len(target_values) + len(source_values) - overlap
                )
                if not source_values:
                    continue
                placeholders = ", ".join("?" for _ in columns)
                target.executemany(
                    f"INSERT OR IGNORE INTO {table} "
                    f"({', '.join(columns)}) VALUES ({placeholders})",
                    source_values,
                )
            copied = {
                table: int(target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in TABLE_COLUMNS
            }
        if copied != expected_counts:
            raise RuntimeError("PostgreSQL migration row-count verification failed")
        return _source_digest(source_path), copied
    finally:
        source.close()
