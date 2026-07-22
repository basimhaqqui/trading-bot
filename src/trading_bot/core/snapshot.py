from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trading_bot.core.audit import AuditLedger
from trading_bot.ingestion.runner import IngestionRunLedger


@dataclass(frozen=True)
class SnapshotSummary:
    output_path: Path
    bytes_written: int
    sha256: str
    events: int
    audit_records: int
    ingestion_runs: int


def create_verified_snapshot(
    source_path: str | Path, output_path: str | Path
) -> SnapshotSummary:
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
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
