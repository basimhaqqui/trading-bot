from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trading_bot.core.audit import AuditLedger
from trading_bot.core.database import DatabaseLocation, is_postgres_location
from trading_bot.core.serialization import sha256_digest
from trading_bot.ingestion.runner import IngestionRunLedger


@dataclass(frozen=True)
class SnapshotSummary:
    output_path: Path
    bytes_written: int
    sha256: str
    events: int
    audit_records: int
    ingestion_runs: int
    paper_records: int
    paper_control_ready: bool


def snapshot_manifest(summary: SnapshotSummary) -> dict[str, object]:
    return {
        "format": "trading-bot-sqlite-snapshot",
        "version": 1,
        "file": summary.output_path.name,
        "bytes": summary.bytes_written,
        "sha256": summary.sha256,
        "counts": {
            "market_events": summary.events,
            "audit_records": summary.audit_records,
            "ingestion_runs": summary.ingestion_runs,
            "paper_records": summary.paper_records,
        },
        "paper_control_ready": summary.paper_control_ready,
    }


def create_verified_snapshot(
    source_path: DatabaseLocation, output_path: str | Path
) -> SnapshotSummary:
    if is_postgres_location(source_path):
        raise ValueError("SQLite snapshots are unavailable for PostgreSQL persistence")
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"database does not exist: {source}")
    if source == output:
        raise ValueError("snapshot output must differ from the source database")
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_uri = f"{source.as_uri()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source_connection:
            with sqlite3.connect(temporary) as snapshot_connection:
                source_connection.backup(snapshot_connection)
        with sqlite3.connect(temporary) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"snapshot integrity check failed: {integrity}")
            events = int(connection.execute("SELECT COUNT(*) FROM market_events").fetchone()[0])
        audit_records = AuditLedger(temporary).verify_integrity()
        ingestion_runs = IngestionRunLedger(temporary).verify_integrity()
        paper_records, paper_control_ready = _verify_paper_state(temporary)
        digest = _file_digest(temporary)
        bytes_written = temporary.stat().st_size
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return SnapshotSummary(
        output,
        bytes_written,
        digest,
        events,
        audit_records,
        ingestion_runs,
        paper_records,
        paper_control_ready,
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_paper_state(path: Path) -> tuple[int, bool]:
    """Verify optional paper tables without mutating a snapshot under review."""
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        total = 0
        for table in ("paper_account_snapshots", "paper_order_events"):
            if table not in table_names:
                continue
            rows = connection.execute(
                f"SELECT payload_json, digest FROM {table}"
            ).fetchall()
            total += len(rows)
            for payload_json, digest in rows:
                if sha256_digest(json.loads(payload_json)) != digest:
                    raise RuntimeError(f"digest mismatch in {table}")

        ready = False
        if "paper_execution_control" in table_names:
            row = connection.execute(
                """
                SELECT enabled, kill_switch_active
                FROM paper_execution_control
                WHERE environment = 'paper'
                """
            ).fetchone()
            if row is not None:
                ready = bool(row[0]) and not bool(row[1])
    return total, ready
