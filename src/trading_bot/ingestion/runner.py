from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Protocol

from trading_bot.agents.prediction import (
    FastPredictionSettlementV7Specialist,
    FastPredictionSettlementV8Specialist,
    FastPredictionSettlementV9Specialist,
    FastPredictionSettlementV10Specialist,
)
from trading_bot.core.audit import AuditLedger
from trading_bot.core.database import (
    DatabaseLocation,
    connect_database,
    initialize_schema,
    is_postgres_location,
)
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    MarketEvent,
    MarketEventType,
)
from trading_bot.core.serialization import (
    canonical_json,
    require_aware,
    sha256_digest,
    utc_now,
)
from trading_bot.core.store import PointInTimeStore
from trading_bot.data.collectors import (
    AlpacaOptionsCollector,
    AlpacaStockCollector,
    CoinbaseCollector,
    DexscreenerCollector,
    KalshiCollector,
    SolanaMintAuthorityCollector,
)
from trading_bot.data.schemas import CollectionBatch, DataQualityDiagnostic, DiagnosticSeverity
from trading_bot.evaluation.outcomes import (
    forecast_label_deadline,
    forecast_outcome_target_time,
)
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan


class CollectorFactory(Protocol):
    def __call__(self, venue: str, dataset: str) -> object:
        ...


class IngestionRunStatus(StrEnum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


class IngestionObservationOrigin(StrEnum):
    """Attestation supplied by the process that ran an observation cycle."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"


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
    observation_origin: IngestionObservationOrigin = IngestionObservationOrigin.MANUAL
    requested_instruments: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "started_at", require_aware(self.started_at, "started_at"))
        object.__setattr__(self, "finished_at", require_aware(self.finished_at, "finished_at"))
        if self.finished_at < self.started_at:
            raise ValueError("ingestion run cannot finish before it starts")
        if min(self.instruments_seen, self.events_inserted) < 0:
            raise ValueError("ingestion counts cannot be negative")
        if self.requested_instruments is not None and self.requested_instruments < 0:
            raise ValueError("requested instrument count cannot be negative")
        if self.status is IngestionRunStatus.FAILED and not self.error_type:
            raise ValueError("failed ingestion runs require an error type")
        if not isinstance(self.observation_origin, IngestionObservationOrigin):
            raise ValueError("observation_origin must be a supported origin")
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
        self.path: DatabaseLocation = path

    def initialize(self) -> None:
        with connect_database(self.path) as connection:
            initialize_schema(
                connection,
                RUN_SCHEMA,
                append_only_tables=("ingestion_runs",),
            )

    def append(self, record: IngestionRunRecord) -> None:
        record_json = canonical_json(record)
        digest = sha256_digest(record)
        with connect_database(self.path) as connection:
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
        with connect_database(self.path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()
        return int(row[0])

    def resume_cursor(self, plan_name: str, job_id: str) -> str | None:
        """Recover the next page from the latest completed page for one job."""
        with connect_database(self.path) as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM ingestion_runs
                WHERE plan_name = ? AND job_id = ? AND status IN (?, ?)
                ORDER BY started_at DESC, run_id DESC
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
        with connect_database(self.path) as connection:
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
        audit: AuditLedger | None = None,
        working_set: PointInTimeStore | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.collector_factory = collector_factory or default_collector_factory
        self.audit = audit
        self.working_set = working_set

    def run_plan(
        self,
        plan: ShadowIngestionPlan,
        *,
        collected_at: datetime | None = None,
        observation_origin: IngestionObservationOrigin = IngestionObservationOrigin.MANUAL,
    ) -> tuple[IngestionRunRecord, ...]:
        if not isinstance(observation_origin, IngestionObservationOrigin):
            raise ValueError("observation_origin must be a supported origin")
        collection_override = (
            require_aware(collected_at, "collected_at") if collected_at is not None else None
        )
        if is_postgres_location(self.store.path):
            # GitHub Actions serializes production cycles. Neon uses transaction-mode
            # pooling, where session advisory locks are intentionally unsupported.
            return tuple(
                self._run_job(plan.name, job, collection_override, observation_origin)
                for job in plan.jobs
                if job.is_active()
            )
        lock_path = Path(self.store.path).with_suffix(Path(self.store.path).suffix + ".shadow.lock")
        with exclusive_run_lock(lock_path):
            return tuple(
                self._run_job(plan.name, job, collection_override, observation_origin)
                for job in plan.jobs
                if job.is_active()
            )

    def _run_job(
        self,
        plan_name: str,
        job: ObservationJob,
        collected_at: datetime | None,
        observation_origin: IngestionObservationOrigin,
    ) -> IngestionRunRecord:
        run_id = str(uuid.uuid4())
        started_at = utc_now()
        request_cursor: str | None = None
        requested_instruments: int | None = None
        try:
            request_cursor = (
                None
                if job.cursor_mode == "restart"
                else self.ledger.resume_cursor(plan_name, job.job_id)
            )
            collector = self.collector_factory(job.venue, job.dataset)
            tickers: tuple[str, ...] = ()
            option_reference_price: float | None = None
            if (
                job.venue == "alpaca"
                and job.dataset == "chain"
                and job.symbol
                and job.strike_band_pct is not None
            ):
                option_reference_price = self._latest_equity_close(
                    job.symbol, collected_at or started_at
                )
                if option_reference_price is None:
                    raise RuntimeError(
                        f"no point-in-time underlying close for filtered option cohort: {job.symbol}"
                    )
            if job.venue == "kalshi" and job.dataset == "forecast_outcomes":
                tickers = self._pending_prediction_tickers(
                    collected_at or started_at, job.limit
                )
                requested_instruments = len(tickers)
                if not tickers:
                    batch = CollectionBatch(
                        "kalshi", metadata={"pending_forecast_tickers": 0}
                    )
                else:
                    batch = collect_job(
                        collector,
                        job,
                        collected_at,
                        request_cursor,
                        tickers=tickers,
                    )
            elif job.venue == "solana" and job.dataset == "mint_authorities":
                token_addresses = self._pending_solana_mint_addresses(
                    collected_at or started_at, job.limit
                )
                if not token_addresses:
                    batch = CollectionBatch(
                        "solana", metadata={"pending_mint_authority_addresses": 0}
                    )
                else:
                    batch = collect_job(
                        collector,
                        job,
                        collected_at,
                        request_cursor,
                        token_addresses=token_addresses,
                    )
            elif job.venue == "solana" and job.dataset == "holder_concentrations":
                token_addresses = self._pending_solana_holder_concentration_addresses(
                    collected_at or started_at, job.limit
                )
                if not token_addresses:
                    batch = CollectionBatch(
                        "solana", metadata={"pending_holder_concentration_addresses": 0}
                    )
                else:
                    batch = collect_job(
                        collector,
                        job,
                        collected_at,
                        request_cursor,
                        token_addresses=token_addresses,
                    )
            elif job.venue == "solana" and job.dataset == "holder_activity":
                token_addresses = self._pending_solana_holder_activity_addresses(
                    collected_at or started_at, job.limit
                )
                if not token_addresses:
                    batch = CollectionBatch(
                        "solana", metadata={"pending_holder_activity_addresses": 0}
                    )
                else:
                    batch = collect_job(
                        collector,
                        job,
                        collected_at,
                        request_cursor,
                        token_addresses=token_addresses,
                    )
            else:
                batch = collect_job(
                    collector,
                    job,
                    collected_at,
                    request_cursor,
                    option_reference_price=option_reference_price,
                )
            if batch.cursor is not None and (
                not isinstance(batch.cursor, str) or len(batch.cursor) > MAX_CURSOR_LENGTH
            ):
                raise ValueError(
                    f"next_cursor must be a string no longer than {MAX_CURSOR_LENGTH} characters"
                )
            instruments_seen, events_inserted = self.store.append_batch(batch)
            # This is a disposable local read cache. Canonical persistence always
            # precedes it, so an eviction can delay research but cannot replace or
            # fabricate durable evidence.
            if self.working_set is not None:
                self.working_set.append_batch(batch)
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
                requested_instruments=requested_instruments,
                diagnostics=batch.diagnostics,
                request_cursor=request_cursor,
                next_cursor=batch.cursor,
                batch_digest=sha256_digest(batch),
                observation_origin=observation_origin,
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
                requested_instruments=requested_instruments,
                request_cursor=request_cursor,
                error_type=type(exc).__name__,
                error_message=str(exc)[:2000],
                observation_origin=observation_origin,
            )
        self.ledger.append(record)
        return record

    def _pending_prediction_tickers(
        self, as_of: datetime, limit: int
    ) -> tuple[str, ...]:
        if self.audit is None:
            raise RuntimeError("forecast outcome jobs require an audit ledger")
        scored_ids = self.audit.scored_forecast_ids()
        forecasts_with_targets = []
        for forecast in self.audit.forecasts():
            if (
                forecast.kind is not ForecastKind.BINARY_PROBABILITY
                or forecast.forecast_id in scored_ids
            ):
                continue
            target_time = forecast_outcome_target_time(forecast)
            if target_time is not None:
                forecasts_with_targets.append((forecast, target_time))
        forecast_instrument_ids = tuple(
            dict.fromkeys(forecast.instrument_id for forecast, _ in forecasts_with_targets)
        )
        latest_rules: dict[str, MarketEvent] = {}
        for event in self.store.events_available_at(
            as_of,
            instrument_ids=forecast_instrument_ids,
            event_type=MarketEventType.CONTRACT_RULE,
            # A receipt after an overdue target changes poll priority. Older
            # receipts cannot make a current request more urgent, so this
            # bounded window avoids re-reading the full Kalshi rule history.
            available_since=as_of - timedelta(days=1),
        ):
            current = latest_rules.get(event.instrument_id)
            if current is None or (
                event.available_at,
                event.event_time,
                event.event_id,
            ) > (
                current.available_at,
                current.event_time,
                current.event_id,
            ):
                latest_rules[event.instrument_id] = event
        def priority(item: tuple[Forecast, datetime]) -> tuple[object, ...]:
            forecast, target_time = item
            label_deadline = forecast_label_deadline(forecast)
            # Kalshi documents that markets with can_close_early=true may close
            # before their registered target time. Poll those v7-v10 forecasts for the
            # full pre-registered window so the bounded request cap cannot
            # delay receipt of an otherwise eligible early finalization.
            if (
                forecast.specialist_id
                in {
                    FastPredictionSettlementV7Specialist.agent_id,
                    FastPredictionSettlementV8Specialist.agent_id,
                    FastPredictionSettlementV9Specialist.agent_id,
                    FastPredictionSettlementV10Specialist.agent_id,
                }
                and forecast.values.get("can_close_early") is True
                and label_deadline is not None
                and label_deadline > forecast.generated_at
                and forecast.generated_at <= as_of <= label_deadline
            ):
                return (
                    0,
                    label_deadline,
                    target_time,
                    forecast.generated_at,
                    forecast.forecast_id,
                )
            # Fast-lane labels are only useful while their immutable
            # finalization window remains open. Prioritize them over an
            # unbounded legacy overdue queue so the fixed 100-ticker request
            # cap cannot erase prospective rapid evidence.
            if (
                target_time <= as_of
                and label_deadline is not None
                and label_deadline > target_time
                and as_of <= label_deadline
            ):
                return (
                    1,
                    label_deadline,
                    target_time,
                    forecast.generated_at,
                    forecast.forecast_id,
                )
            rule = latest_rules.get(forecast.instrument_id)
            checked_at = (
                rule.available_at
                if rule is not None and rule.available_at >= target_time
                else None
            )
            if target_time <= as_of:
                # A Kalshi request records a fresh contract rule.  Until that
                # receipt exists after the forecast target, prefer the overdue
                # ticker so a 100-ticker API cap cannot permanently starve it.
                return (
                    2,
                    checked_at is not None,
                    checked_at or target_time,
                    target_time,
                    forecast.generated_at,
                    forecast.forecast_id,
                )
            return (
                3,
                target_time,
                forecast.generated_at,
                forecast.forecast_id,
            )

        forecasts_with_targets.sort(key=priority)

        selected: list[str] = []
        selected_instruments: set[str] = set()
        for forecast, _ in forecasts_with_targets:
            if forecast.instrument_id in selected_instruments:
                continue
            rule = latest_rules.get(forecast.instrument_id)
            prefix = "kalshi:prediction:"
            if not forecast.instrument_id.startswith(prefix):
                continue
            ticker = forecast.instrument_id.removeprefix(prefix)
            if (
                (rule is not None and rule.payload.get("mve_collection_ticker"))
                or ticker.startswith("KXMVE")
            ):
                continue
            selected.append(ticker)
            selected_instruments.add(forecast.instrument_id)
            if len(selected) >= min(limit, 100):
                break
        return tuple(selected)

    def _latest_equity_close(self, symbol: str, as_of: datetime) -> float | None:
        equities = [
            instrument
            for instrument in self.store.instruments(asset_class=AssetClass.EQUITY)
            if instrument.venue == "alpaca" and instrument.symbol.upper() == symbol.upper()
        ]
        if len(equities) != 1:
            return None
        bars = self.store.events_available_at(
            as_of,
            instrument_id=equities[0].instrument_id,
            event_type=MarketEventType.BAR,
        )
        bars.sort(key=lambda item: (item.event_time, item.available_at, item.event_id))
        for event in reversed(bars):
            try:
                close = float(event.payload.get("close"))
            except (TypeError, ValueError):
                continue
            if close > 0 and close < float("inf"):
                return close
        return None

    def _pending_solana_mint_addresses(self, as_of: datetime, limit: int) -> tuple[str, ...]:
        """Select mints missing v2 transfer-control reads, then oldest-observed.

        This is a read-only fairness queue.  It never turns a discovery into a
        forecast or intent and uses only the existing public instrument symbols.
        """
        candidates: list[tuple[bool, datetime | None, str]] = []
        for instrument in self.store.instruments(asset_class=AssetClass.MEMECOIN):
            if (
                instrument.venue != "dexscreener"
                or not SolanaMintAuthorityCollector.is_valid_mint_address(instrument.symbol)
            ):
                continue
            events = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.ONCHAIN_STATE,
            )
            authority_observations = [
                event
                for event in events
                if event.source
                in {
                    "solana-rpc-get-multiple-accounts-finalized-v1",
                    "solana-rpc-get-multiple-accounts-finalized-v2",
                }
            ]
            has_transfer_control_read = any(
                event.source == "solana-rpc-get-multiple-accounts-finalized-v2"
                for event in authority_observations
            )
            latest = max((event.available_at for event in authority_observations), default=None)
            candidates.append((has_transfer_control_read, latest, instrument.symbol))
        candidates.sort(
            key=lambda item: (item[0], item[1] is not None, item[1] or as_of, item[2])
        )
        return tuple(address for _, _, address in candidates[:limit])

    def _pending_solana_holder_concentration_addresses(
        self, as_of: datetime, limit: int
    ) -> tuple[str, ...]:
        """Select mints missing a bounded finalized concentration observation."""
        candidates: list[tuple[bool, datetime | None, str]] = []
        for instrument in self.store.instruments(asset_class=AssetClass.MEMECOIN):
            if (
                instrument.venue != "dexscreener"
                or not SolanaMintAuthorityCollector.is_valid_mint_address(instrument.symbol)
            ):
                continue
            events = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.ONCHAIN_STATE,
            )
            observations = [
                event
                for event in events
                if event.source == "solana-rpc-token-holder-concentration-finalized-v1"
            ]
            latest = max((event.available_at for event in observations), default=None)
            candidates.append((bool(observations), latest, instrument.symbol))
        candidates.sort(
            key=lambda item: (item[0], item[1] is not None, item[1] or as_of, item[2])
        )
        return tuple(address for _, _, address in candidates[:limit])

    def _pending_solana_holder_activity_addresses(
        self, as_of: datetime, limit: int
    ) -> tuple[str, ...]:
        """Select discovered mints missing aggregate finalized account-activity reads."""
        candidates: list[tuple[bool, datetime | None, str]] = []
        for instrument in self.store.instruments(asset_class=AssetClass.MEMECOIN):
            if (
                instrument.venue != "dexscreener"
                or not SolanaMintAuthorityCollector.is_valid_mint_address(instrument.symbol)
            ):
                continue
            events = self.store.events_available_at(
                as_of,
                instrument_id=instrument.instrument_id,
                event_type=MarketEventType.ONCHAIN_STATE,
            )
            observations = [
                event
                for event in events
                if event.source == "solana-rpc-finalized-holder-activity-v1"
            ]
            latest = max((event.available_at for event in observations), default=None)
            candidates.append((bool(observations), latest, instrument.symbol))
        candidates.sort(
            key=lambda item: (item[0], item[1] is not None, item[1] or as_of, item[2])
        )
        return tuple(address for _, _, address in candidates[:limit])


def default_collector_factory(venue: str, dataset: str) -> object:
    if venue == "kalshi":
        return KalshiCollector()
    if venue == "coinbase":
        return CoinbaseCollector()
    if venue == "dexscreener":
        return DexscreenerCollector()
    if venue == "solana" and dataset in {
        "mint_authorities",
        "holder_concentrations",
        "holder_activity",
    }:
        return SolanaMintAuthorityCollector()
    if venue == "alpaca":
        key_id = os.getenv("ALPACA_MARKET_DATA_KEY_ID", "")
        secret_key = os.getenv("ALPACA_MARKET_DATA_SECRET_KEY", "")
        if dataset in {"bars", "quotes"}:
            return AlpacaStockCollector(key_id, secret_key)
        return AlpacaOptionsCollector(key_id, secret_key)
    raise ValueError(f"unsupported venue: {venue}")


def collect_job(
    collector: object,
    job: ObservationJob,
    collected_at: datetime | None,
    cursor: str | None = None,
    *,
    tickers: tuple[str, ...] = (),
    token_addresses: tuple[str, ...] = (),
    option_reference_price: float | None = None,
) -> CollectionBatch:
    if job.venue == "kalshi":
        if job.dataset == "markets":
            window_start = collected_at or utc_now()
            min_close_ts = None
            max_close_ts = None
            status: str | None = job.status
            active_only = False
            if job.close_lookahead_hours is not None:
                min_close_ts = int(window_start.timestamp())
                max_close_ts = int(
                    (window_start + timedelta(hours=job.close_lookahead_hours)).timestamp()
                )
                # Kalshi documents close-time filters as incompatible with the
                # public `status=open` filter. Fetch the legal unfiltered
                # window, then keep only the REST `active` lifecycle state in
                # the read-only collector.
                status = None
                active_only = True
            return collector.collect_markets(  # type: ignore[attr-defined,no-any-return]
                collected_at=collected_at,
                status=status,
                limit=job.limit,
                cursor=cursor,
                mve_filter=job.mve_filter,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
                active_only=active_only,
            )
        if job.dataset == "forecast_outcomes":
            return collector.collect_markets(  # type: ignore[attr-defined,no-any-return]
                collected_at=collected_at,
                status=None,
                limit=min(job.limit, 100),
                cursor=cursor,
                tickers=tickers,
                mve_filter="exclude",
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
    if job.venue == "dexscreener" and job.dataset == "token_profiles":
        return collector.collect_token_profiles(  # type: ignore[attr-defined,no-any-return]
            collected_at=collected_at,
            limit=job.limit,
            include_pool_observations=job.include_pool_observations,
        )
    if job.venue == "solana" and job.dataset == "mint_authorities":
        return collector.collect_mint_authorities(  # type: ignore[attr-defined,no-any-return]
            token_addresses, collected_at=collected_at
        )
    if job.venue == "solana" and job.dataset == "holder_concentrations":
        return collector.collect_holder_concentrations(  # type: ignore[attr-defined,no-any-return]
            token_addresses, collected_at=collected_at
        )
    if job.venue == "solana" and job.dataset == "holder_activity":
        return collector.collect_holder_activity(  # type: ignore[attr-defined,no-any-return]
            token_addresses, collected_at=collected_at
        )
    if job.venue == "alpaca":
        if job.dataset == "chain" and job.symbol:
            window_start = collected_at or utc_now()
            strike_price_gte = None
            strike_price_lte = None
            if job.strike_band_pct is not None:
                if option_reference_price is None or option_reference_price <= 0:
                    raise ValueError("filtered option cohort requires a positive reference price")
                strike_price_gte = option_reference_price * (1 - job.strike_band_pct)
                strike_price_lte = option_reference_price * (1 + job.strike_band_pct)
            expiration_date_gte = None
            expiration_date_lte = None
            if job.expiration_lookahead_days is not None:
                expiration_date_gte = window_start.date().isoformat()
                expiration_date_lte = (
                    window_start.date() + timedelta(days=job.expiration_lookahead_days)
                ).isoformat()
            updated_since = None
            if job.updated_since_minutes is not None:
                updated_since = window_start - timedelta(minutes=job.updated_since_minutes)
            return collector.collect_chain(  # type: ignore[attr-defined,no-any-return]
                job.symbol,
                collected_at=collected_at,
                feed=job.feed,
                limit=job.limit,
                page_token=cursor,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                strike_price_gte=strike_price_gte,
                strike_price_lte=strike_price_lte,
                updated_since=updated_since,
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
        if job.dataset == "quotes" and job.symbol:
            return collector.collect_latest_quote(  # type: ignore[attr-defined,no-any-return]
                job.symbol,
                collected_at=collected_at,
                feed=job.stock_feed,
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
