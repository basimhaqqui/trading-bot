from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Protocol

from trading_bot.core.serialization import canonical_json, require_aware, sha256_digest, utc_now
from trading_bot.core.store import PointInTimeStore
from trading_bot.data.collectors import (
    AlpacaOptionsCollector,
    AlpacaStockCollector,
    CoinbaseCollector,
    KalshiCollector,
)
from trading_bot.data.schemas import CollectionBatch, DataQualityDiagnostic, DiagnosticSeverity
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan


class CollectorFactory(Protocol):
    def __call__(self, venue: str, dataset: str) -> object:
        ...


class IngestionRunStatus(StrEnum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


MAX_CURSOR_LENGTH = 4096


@dataclass(frozen=True)
class IngestionRunRecord:
    run_id: str
    plan_name: str
    job_id: str
    venue: str
    dataset: str
    status: IngestionRunStatus
    started_at: datetime
    finished_at: datetime
    instruments_seen: int
    events_inserted: int
    diagnostics: tuple[DataQualityDiagnostic, ...] = ()
    request_cursor: str | None = None
    next_cursor: str | None = None
    batch_digest: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "started_at", require_aware(self.started_at, "started_at"))
        object.__setattr__(self, "finished_at", require_aware(self.finished_at, "finished_at"))
        if self.finished_at < self.started_at:
            raise ValueError("ingestion run cannot finish before it starts")
        if min(self.instruments_seen, self.events_inserted) < 0:
            raise ValueError("ingestion counts cannot be negative")
        if self.status is IngestionRunStatus.FAILED and not self.error_type:
            raise ValueError("failed ingestion runs require an error type")
        for field_name, cursor in (
            ("request_cursor", self.request_cursor),
            ("next_cursor", self.next_cursor),
        ):
            if cursor is not None and (
                not isinstance(cursor, str) or len(cursor) > MAX_CURSOR_LENGTH
            ):
                raise ValueError(
                    f"{field_name} must be a string no longer than {MAX_CURSOR_LENGTH} characters"
                )


RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    plan_name TEXT NOT NULL,
    job_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    dataset TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    digest TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_job
ON ingestion_runs(plan_name, job_id, started_at);

CREATE TRIGGER IF NOT EXISTS ingestion_runs_no_update
BEFORE UPDATE ON ingestion_runs BEGIN
    SELECT RAISE(ABORT, 'ingestion_runs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS ingestion_runs_no_delete
BEFORE DELETE ON ingestion_runs BEGIN
    SELECT RAISE(ABORT, 'ingestion_runs is append-only');
END;
"""


class IngestionRunLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(RUN_SCHEMA)

    def append(self, record: IngestionRunRecord) -> None:
        record_json = canonical_json(record)
        digest = sha256_digest(record)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, plan_name, job_id, venue, dataset, status,
                    started_at, finished_at, record_json, digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.plan_name,
                    record.job_id,
                    record.venue,
                    record.dataset,
                    record.status.value,
                    record.started_at.isoformat(),
                    record.finished_at.isoformat(),
                    record_json,
                    digest,
                ),
            )

    def count(self) -> int:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()
        return int(row[0])

    def resume_cursor(self, plan_name: str, job_id: str) -> str | None:
        """Recover the next page from the latest completed page for one job."""
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM ingestion_runs
                WHERE plan_name = ? AND job_id = ? AND status IN (?, ?)
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """,
                (
                    plan_name,
                    job_id,
                    IngestionRunStatus.SUCCESS.value,
                    IngestionRunStatus.DEGRADED.value,
                ),
            ).fetchone()
        if row is None:
            return None
        cursor = json.loads(row[0]).get("next_cursor")
        if cursor is None:
            return None
        if not isinstance(cursor, str) or len(cursor) > MAX_CURSOR_LENGTH:
            raise RuntimeError(f"invalid stored cursor for ingestion job: {job_id}")
        return cursor

    def verify_integrity(self) -> int:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT run_id, record_json, digest FROM ingestion_runs"
            ).fetchall()
        for run_id, record_json, digest in rows:
            if sha256_digest(json.loads(record_json)) != digest:
                raise RuntimeError(f"ingestion run digest mismatch: {run_id}")
        return len(rows)


class ShadowIngestionRunner:
    def __init__(
        self,
        store: PointInTimeStore,
        ledger: IngestionRunLedger,
        collector_factory: CollectorFactory | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.collector_factory = collector_factory or default_collector_factory

    def run_plan(
        self,
        plan: ShadowIngestionPlan,
        *,
        collected_at: datetime | None = None,
    ) -> tuple[IngestionRunRecord, ...]:
        collection_override = (
            require_aware(collected_at, "collected_at") if collected_at is not None else None
        )
        lock_path = self.store.path.with_suffix(self.store.path.suffix + ".shadow.lock")
        with exclusive_run_lock(lock_path):
            return tuple(
                self._run_job(plan.name, job, collection_override)
                for job in plan.jobs
                if job.is_active()
            )

    def _run_job(
        self, plan_name: str, job: ObservationJob, collected_at: datetime | None
    ) -> IngestionRunRecord:
        run_id = str(uuid.uuid4())
        started_at = utc_now()
        request_cursor: str | None = None
        try:
            request_cursor = self.ledger.resume_cursor(plan_name, job.job_id)
            collector = self.collector_factory(job.venue, job.dataset)
            batch = collect_job(collector, job, collected_at, request_cursor)
            if batch.cursor is not None and (
                not isinstance(batch.cursor, str) or len(batch.cursor) > MAX_CURSOR_LENGTH
            ):
                raise ValueError(
                    f"next_cursor must be a string no longer than {MAX_CURSOR_LENGTH} characters"
                )
            instruments_seen, events_inserted = self.store.append_batch(batch)
            status = (
                IngestionRunStatus.DEGRADED
                if any(item.severity is not DiagnosticSeverity.INFO for item in batch.diagnostics)
                else IngestionRunStatus.SUCCESS
            )
            record = IngestionRunRecord(
                run_id=run_id,
                plan_name=plan_name,
                job_id=job.job_id,
                venue=job.venue,
                dataset=job.dataset,
                status=status,
                started_at=started_at,
                finished_at=utc_now(),
                instruments_seen=instruments_seen,
                events_inserted=events_inserted,
                diagnostics=batch.diagnostics,
                request_cursor=request_cursor,
                next_cursor=batch.cursor,
                batch_digest=sha256_digest(batch),
            )
        except Exception as exc:
            record = IngestionRunRecord(
                run_id=run_id,
                plan_name=plan_name,
                job_id=job.job_id,
                venue=job.venue,
                dataset=job.dataset,
                status=IngestionRunStatus.FAILED,
                started_at=started_at,
                finished_at=utc_now(),
                instruments_seen=0,
                events_inserted=0,
                request_cursor=request_cursor,
                error_type=type(exc).__name__,
                error_message=str(exc)[:2000],
            )
        self.ledger.append(record)
        return record


def default_collector_factory(venue: str, dataset: str) -> object:
    if venue == "kalshi":
        return KalshiCollector()
    if venue == "coinbase":
        return CoinbaseCollector()
    if venue == "alpaca":
        key_id = os.getenv("ALPACA_MARKET_DATA_KEY_ID", "")
        secret_key = os.getenv("ALPACA_MARKET_DATA_SECRET_KEY", "")
        if dataset == "bars":
            return AlpacaStockCollector(key_id, secret_key)
        return AlpacaOptionsCollector(key_id, secret_key)
    raise ValueError(f"unsupported venue: {venue}")


def collect_job(
    collector: object,
    job: ObservationJob,
    collected_at: datetime | None,
    cursor: str | None = None,
) -> CollectionBatch:
    if job.venue == "kalshi":
        if job.dataset == "markets":
            return collector.collect_markets(  # type: ignore[attr-defined,no-any-return]
                collected_at=collected_at,
                status=job.status,
                limit=job.limit,
                cursor=cursor,
            )
        if job.dataset == "trades":
            return collector.collect_trades(  # type: ignore[attr-defined,no-any-return]
                collected_at=collected_at,
                ticker=job.symbol,
                limit=job.limit,
                cursor=cursor,
            )
        if job.dataset == "book" and job.symbol:
            return collector.collect_orderbook(  # type: ignore[attr-defined,no-any-return]
                job.symbol, collected_at=collected_at, depth=min(job.limit, 100)
            )
    if job.venue == "coinbase":
        if job.dataset == "products":
            return collector.collect_products(  # type: ignore[attr-defined,no-any-return]
                collected_at=collected_at,
                product_type=job.product_type,
                limit=job.limit,
                cursor=cursor,
            )
        if job.dataset == "book" and job.symbol:
            return collector.collect_product_book(  # type: ignore[attr-defined,no-any-return]
                job.symbol, collected_at=collected_at, limit=job.limit
            )
        if job.dataset == "candles" and job.symbol:
            return collector.collect_candles(  # type: ignore[attr-defined,no-any-return]
                job.symbol,
                collected_at=collected_at,
                granularity=job.granularity,
                limit=job.limit,
            )
    if job.venue == "alpaca":
        if job.dataset == "chain" and job.symbol:
            return collector.collect_chain(  # type: ignore[attr-defined,no-any-return]
                job.symbol,
                collected_at=collected_at,
                feed=job.feed,
                limit=job.limit,
                page_token=cursor,
            )
        if job.dataset == "bars" and job.symbol:
            return collector.collect_daily_bars(  # type: ignore[attr-defined,no-any-return]
                job.symbol,
                collected_at=collected_at,
                feed=job.stock_feed,
                lookback_days=job.lookback_days,
                limit=job.limit,
                page_token=cursor,
            )
    raise TypeError(f"collector does not implement {job.venue}/{job.dataset}")


@contextmanager
def exclusive_run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another shadow ingestion cycle is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
