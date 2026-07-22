import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.store import PointInTimeStore
from trading_bot.data.schemas import (
    CollectionBatch,
    DataQualityDiagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
)
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan, load_plan
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
    IngestionRunStatus,
    ShadowIngestionRunner,
    collect_job,
)


class FakeCoinbaseCollector:
    def __init__(self, batch):
        self.batch = batch

    def collect_products(self, **kwargs):
        return self.batch


class FailingCollector:
    def collect_products(self, **kwargs):
        raise TimeoutError("fixture timeout")


class PaginatedCoinbaseCollector:
    def __init__(self):
        self.seen_cursors = []
        self.fail_on = None

    def collect_products(self, **kwargs):
        cursor = kwargs.get("cursor")
        self.seen_cursors.append(cursor)
        if self.fail_on is not None and cursor == self.fail_on:
            raise TimeoutError("page timeout")
        next_cursor = "page-2" if cursor is None else None
        return CollectionBatch("coinbase", cursor=next_cursor)


class FakeAlpacaCollector:
    def __init__(self):
        self.page_token = None
        self.bar_page_token = None

    def collect_chain(self, symbol, **kwargs):
        self.page_token = kwargs.get("page_token")
        return CollectionBatch("alpaca")

    def collect_daily_bars(self, symbol, **kwargs):
        self.bar_page_token = kwargs.get("page_token")
        return CollectionBatch("alpaca")


class FakeKalshiCollector:
    def __init__(self):
        self.market_cursor = None
        self.trade_cursor = None

    def collect_markets(self, **kwargs):
        self.market_cursor = kwargs.get("cursor")
        return CollectionBatch("kalshi")

    def collect_trades(self, **kwargs):
        self.trade_cursor = kwargs.get("cursor")
        return CollectionBatch("kalshi")


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.db_path = self.path / "shadow.db"
        self.store = PointInTimeStore(self.db_path)
        self.store.initialize()
        self.ledger = IngestionRunLedger(self.db_path)
        self.ledger.initialize()
        self.now = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_plan_rejects_embedded_credentials(self):
        plan_path = self.path / "bad-plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "name": "bad",
                    "api_key": "must-not-be-here",
                    "jobs": [
                        {
                            "job_id": "products",
                            "venue": "coinbase",
                            "dataset": "products",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "credentials are forbidden"):
            load_plan(plan_path)

    def test_coinbase_candle_jobs_are_bounded_and_typed(self):
        job = ObservationJob(
            "btc-candles",
            "coinbase",
            "candles",
            symbol="BTC-USD",
            limit=30,
            granularity="ONE_HOUR",
        )
        self.assertEqual(job.granularity, "ONE_HOUR")
        with self.assertRaisesRegex(ValueError, "cannot exceed 350"):
            ObservationJob(
                "too-many-candles",
                "coinbase",
                "candles",
                symbol="BTC-USD",
                limit=351,
            )

    def test_cycle_records_degraded_data_without_execution_access(self):
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        event = MarketEvent(
            "fixture-book",
            MarketEventType.BOOK_SNAPSHOT,
            "coinbase",
            instrument.instrument_id,
            self.now,
            self.now,
            "fixture",
            {"bid_price": 100, "ask_price": 101},
            ingested_at=self.now,
        )
        diagnostic = DataQualityDiagnostic(
            DiagnosticCode.STALE_EVENT,
            DiagnosticSeverity.WARNING,
            "fixture warning",
            instrument.instrument_id,
            event.event_id,
        )
        batch = CollectionBatch(
            "coinbase", (instrument,), (event,), (diagnostic,), cursor="next"
        )
        collector = FakeCoinbaseCollector(batch)
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (ObservationJob("products", "coinbase", "products"),),
        )
        records = runner.run_plan(plan, collected_at=self.now)
        self.assertEqual(records[0].status, IngestionRunStatus.DEGRADED)
        self.assertEqual(records[0].events_inserted, 1)
        self.assertEqual(records[0].next_cursor, "next")
        self.assertEqual(self.ledger.resume_cursor("fixture-plan", "products"), "next")
        self.assertEqual(self.ledger.count(), 1)
        self.assertEqual(self.ledger.verify_integrity(), 1)
        self.assertEqual(
            self.store.events_available_at(self.now)[0].event_id, "fixture-book"
        )
        with self.assertRaises(sqlite3.DatabaseError):
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    "UPDATE ingestion_runs SET status = 'success' WHERE job_id = 'products'"
                )

    def test_failed_job_is_recorded_and_does_not_abort_cycle(self):
        runner = ShadowIngestionRunner(
            self.store,
            self.ledger,
            collector_factory=lambda venue, dataset: FailingCollector(),
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (ObservationJob("products", "coinbase", "products"),),
        )
        record = runner.run_plan(plan, collected_at=self.now)[0]
        self.assertEqual(record.status, IngestionRunStatus.FAILED)
        self.assertEqual(record.error_type, "TimeoutError")
        self.assertEqual(record.events_inserted, 0)
        self.assertEqual(self.ledger.count(), 1)

    def test_successful_pages_advance_and_terminal_page_resets_cursor(self):
        collector = PaginatedCoinbaseCollector()
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (ObservationJob("products", "coinbase", "products"),),
        )

        first = runner.run_plan(plan, collected_at=self.now)[0]
        second = runner.run_plan(plan, collected_at=self.now)[0]
        third = runner.run_plan(plan, collected_at=self.now)[0]

        self.assertEqual(collector.seen_cursors, [None, "page-2", None])
        self.assertEqual(first.request_cursor, None)
        self.assertEqual(first.next_cursor, "page-2")
        self.assertEqual(second.request_cursor, "page-2")
        self.assertEqual(second.next_cursor, None)
        self.assertEqual(third.request_cursor, None)
        self.assertEqual(third.next_cursor, "page-2")
        self.assertEqual(self.ledger.verify_integrity(), 3)

    def test_failed_page_retries_last_successful_cursor(self):
        collector = PaginatedCoinbaseCollector()
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (ObservationJob("products", "coinbase", "products"),),
        )

        runner.run_plan(plan, collected_at=self.now)
        collector.fail_on = "page-2"
        first_failure = runner.run_plan(plan, collected_at=self.now)[0]
        second_failure = runner.run_plan(plan, collected_at=self.now)[0]

        self.assertEqual(collector.seen_cursors, [None, "page-2", "page-2"])
        self.assertEqual(first_failure.status, IngestionRunStatus.FAILED)
        self.assertEqual(first_failure.request_cursor, "page-2")
        self.assertEqual(second_failure.request_cursor, "page-2")
        self.assertEqual(self.ledger.resume_cursor("fixture-plan", "products"), "page-2")

    def test_alpaca_jobs_receive_resumed_page_token(self):
        collector = FakeAlpacaCollector()
        chain_job = ObservationJob(
            "options", "alpaca", "chain", symbol="AAPL", feed="indicative"
        )
        bars_job = ObservationJob("bars", "alpaca", "bars", symbol="AAPL")
        collect_job(collector, chain_job, self.now, "alpaca-page-2")
        collect_job(collector, bars_job, self.now, "alpaca-bars-page-2")
        self.assertEqual(collector.page_token, "alpaca-page-2")
        self.assertEqual(collector.bar_page_token, "alpaca-bars-page-2")

    def test_kalshi_jobs_receive_resumed_cursor(self):
        collector = FakeKalshiCollector()
        markets_job = ObservationJob("markets", "kalshi", "markets")
        trades_job = ObservationJob("trades", "kalshi", "trades")
        collect_job(collector, markets_job, self.now, "kalshi-markets-page-2")
        collect_job(collector, trades_job, self.now, "kalshi-trades-page-2")
        self.assertEqual(collector.market_cursor, "kalshi-markets-page-2")
        self.assertEqual(collector.trade_cursor, "kalshi-trades-page-2")

    def test_oversized_cursor_fails_before_events_are_stored(self):
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        event = MarketEvent(
            "oversized-cursor-event",
            MarketEventType.BOOK_SNAPSHOT,
            "coinbase",
            instrument.instrument_id,
            self.now,
            self.now,
            "fixture",
            {"bid_price": 100, "ask_price": 101},
            ingested_at=self.now,
        )
        collector = FakeCoinbaseCollector(
            CollectionBatch("coinbase", (instrument,), (event,), cursor="x" * 4097)
        )
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (ObservationJob("products", "coinbase", "products"),),
        )
        record = runner.run_plan(plan, collected_at=self.now)[0]
        self.assertEqual(record.status, IngestionRunStatus.FAILED)
        self.assertEqual(record.error_type, "ValueError")
        self.assertEqual(self.store.events_available_at(self.now), [])


if __name__ == "__main__":
    unittest.main()
