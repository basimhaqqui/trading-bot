import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    Instrument,
    MarketEvent,
    MarketEventType,
)
from trading_bot.core.store import PointInTimeStore
from trading_bot.data.schemas import (
    CollectionBatch,
    DataQualityDiagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
)
from trading_bot.evaluation.shadow import ShadowResearchRunner
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan, load_plan
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
    IngestionObservationOrigin,
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
        self.chain_kwargs = {}
        self.quote_kwargs = None

    def collect_chain(self, symbol, **kwargs):
        self.page_token = kwargs.get("page_token")
        self.chain_kwargs = kwargs
        return CollectionBatch("alpaca")

    def collect_daily_bars(self, symbol, **kwargs):
        self.bar_page_token = kwargs.get("page_token")
        return CollectionBatch("alpaca")

    def collect_latest_quote(self, symbol, **kwargs):
        self.quote_kwargs = {"symbol": symbol, **kwargs}
        return CollectionBatch("alpaca")


class PaginatedAlpacaCollector:
    def __init__(self):
        self.page_tokens = []

    def collect_chain(self, symbol, **kwargs):
        self.page_tokens.append(kwargs.get("page_token"))
        return CollectionBatch("alpaca", cursor="alpaca-page-2")


class FakeKalshiCollector:
    def __init__(self, market_batch=None):
        self.market_cursor = None
        self.trade_cursor = None
        self.market_kwargs = {}
        self.market_batch = market_batch or CollectionBatch("kalshi")

    def collect_markets(self, **kwargs):
        self.market_cursor = kwargs.get("cursor")
        self.market_kwargs = kwargs
        return self.market_batch

    def collect_trades(self, **kwargs):
        self.trade_cursor = kwargs.get("cursor")
        return CollectionBatch("kalshi")


class FakeSolanaCollector:
    def __init__(self):
        self.addresses = ()
        self.holder_addresses = ()

    def collect_mint_authorities(self, addresses, **kwargs):
        self.addresses = addresses
        return CollectionBatch("solana")

    def collect_holder_concentrations(self, addresses, **kwargs):
        self.holder_addresses = addresses
        return CollectionBatch("solana")


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

    def test_dexscreener_profiles_are_bounded_and_need_no_symbol(self):
        job = ObservationJob(
            "solana-profiles",
            "dexscreener",
            "token_profiles",
            limit=25,
            include_pool_observations=True,
        )
        self.assertEqual(job.limit, 25)
        self.assertTrue(job.include_pool_observations)
        with self.assertRaisesRegex(ValueError, "only valid for Dexscreener"):
            ObservationJob(
                "bad-pool-observation", "coinbase", "products", include_pool_observations=True
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            ObservationJob(
                "bad-pool-observation", "dexscreener", "token_profiles", include_pool_observations=1
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed 100"):
            ObservationJob("too-many-profiles", "dexscreener", "token_profiles", limit=101)
        with self.assertRaisesRegex(ValueError, "do not accept a symbol"):
            ObservationJob(
                "symbol-profile", "dexscreener", "token_profiles", symbol="SOL"
            )

    def test_solana_authority_jobs_are_bounded_and_select_discovered_mints(self):
        job = ObservationJob("mint-authorities", "solana", "mint_authorities", limit=25)
        self.assertEqual(job.limit, 25)
        with self.assertRaisesRegex(ValueError, "cannot exceed 25"):
            ObservationJob("too-many-authorities", "solana", "mint_authorities", limit=26)
        with self.assertRaisesRegex(ValueError, "do not accept a symbol"):
            ObservationJob(
                "symbol-authorities", "solana", "mint_authorities", symbol="SOL", limit=25
            )
        mint = "11111111111111111111111111111111"
        instrument = Instrument(
            f"dexscreener:memecoin:solana:{mint}",
            "dexscreener",
            mint,
            AssetClass.MEMECOIN,
            "USD",
        )
        self.store.register_instrument(instrument)
        collector = FakeSolanaCollector()
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        records = runner.run_plan(ShadowIngestionPlan("fixture", (job,)), collected_at=self.now)
        self.assertEqual(records[0].status, IngestionRunStatus.SUCCESS)
        self.assertEqual(collector.addresses, (mint,))

    def test_solana_holder_concentration_jobs_are_bounded_and_select_discovered_mints(self):
        job = ObservationJob("holder-concentrations", "solana", "holder_concentrations", limit=25)
        self.assertEqual(job.limit, 25)
        with self.assertRaisesRegex(ValueError, "cannot exceed 25"):
            ObservationJob("too-many-holders", "solana", "holder_concentrations", limit=26)
        mint = "11111111111111111111111111111111"
        instrument = Instrument(
            f"dexscreener:memecoin:solana:{mint}",
            "dexscreener",
            mint,
            AssetClass.MEMECOIN,
            "USD",
        )
        self.store.register_instrument(instrument)
        collector = FakeSolanaCollector()
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        records = runner.run_plan(ShadowIngestionPlan("fixture", (job,)), collected_at=self.now)
        self.assertEqual(records[0].status, IngestionRunStatus.SUCCESS)
        self.assertEqual(collector.holder_addresses, (mint,))

    def test_solana_transfer_control_reads_upgrade_v1_before_refreshing_v2(self):
        v1_mint = "11111111111111111111111111111111"
        v2_mint = "22222222222222222222222222222222"
        for mint, source in (
            (v1_mint, "solana-rpc-get-multiple-accounts-finalized-v1"),
            (v2_mint, "solana-rpc-get-multiple-accounts-finalized-v2"),
        ):
            instrument = Instrument(
                f"dexscreener:memecoin:solana:{mint}",
                "dexscreener",
                mint,
                AssetClass.MEMECOIN,
                "USD",
            )
            self.store.register_instrument(instrument)
            self.store.append_event(
                MarketEvent(
                    f"authority-{mint[0]}",
                    MarketEventType.ONCHAIN_STATE,
                    "solana",
                    instrument.instrument_id,
                    self.now - timedelta(minutes=5),
                    self.now - timedelta(minutes=5),
                    source,
                    {"safety_status": "blocked_unverified"},
                    ingested_at=self.now - timedelta(minutes=5),
                )
            )
        runner = ShadowIngestionRunner(self.store, self.ledger)

        self.assertEqual(runner._pending_solana_mint_addresses(self.now, 1), (v1_mint,))

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

    def test_cycle_persists_observation_origin_and_defaults_to_manual(self):
        runner = ShadowIngestionRunner(
            self.store,
            self.ledger,
            collector_factory=lambda venue, dataset: FakeCoinbaseCollector(
                CollectionBatch("coinbase")
            ),
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (ObservationJob("products", "coinbase", "products"),),
        )

        scheduled = runner.run_plan(
            plan,
            collected_at=self.now,
            observation_origin=IngestionObservationOrigin.SCHEDULED,
        )[0]
        manual = runner.run_plan(plan, collected_at=self.now)[0]

        self.assertIs(scheduled.observation_origin, IngestionObservationOrigin.SCHEDULED)
        self.assertIs(manual.observation_origin, IngestionObservationOrigin.MANUAL)
        with sqlite3.connect(self.db_path) as connection:
            origins = [
                json.loads(row[0])["observation_origin"]
                for row in connection.execute(
                    "SELECT record_json FROM ingestion_runs ORDER BY rowid ASC"
                )
            ]
        self.assertEqual(origins, ["scheduled", "manual"])

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

    def test_alpaca_quote_jobs_dispatch_to_the_stock_quote_collector(self):
        collector = FakeAlpacaCollector()
        quote_job = ObservationJob(
            "spy-quote", "alpaca", "quotes", symbol="SPY", stock_feed="iex"
        )
        collect_job(collector, quote_job, self.now, None)
        self.assertEqual(
            collector.quote_kwargs,
            {"symbol": "SPY", "collected_at": self.now, "feed": "iex"},
        )
        with self.assertRaises(ValueError):
            ObservationJob("missing-symbol", "alpaca", "quotes")

    def test_restart_cursor_mode_repeats_first_chain_page_and_keeps_audit_cursor(self):
        collector = PaginatedAlpacaCollector()
        runner = ShadowIngestionRunner(
            self.store, self.ledger, collector_factory=lambda venue, dataset: collector
        )
        plan = ShadowIngestionPlan(
            "fixture-plan",
            (
                ObservationJob(
                    "options-cohort",
                    "alpaca",
                    "chain",
                    symbol="AAPL",
                    cursor_mode="restart",
                ),
            ),
        )

        first = runner.run_plan(plan, collected_at=self.now)[0]
        second = runner.run_plan(plan, collected_at=self.now)[0]

        self.assertEqual(collector.page_tokens, [None, None])
        self.assertIsNone(first.request_cursor)
        self.assertIsNone(second.request_cursor)
        self.assertEqual(first.next_cursor, "alpaca-page-2")
        self.assertEqual(second.next_cursor, "alpaca-page-2")
        self.assertEqual(
            self.ledger.resume_cursor("fixture-plan", "options-cohort"),
            "alpaca-page-2",
        )

    def test_restart_cursor_mode_is_restricted_to_allowlisted_paginated_jobs(self):
        with self.assertRaisesRegex(ValueError, "not valid for this observation job"):
            ObservationJob(
                "products",
                "coinbase",
                "products",
                cursor_mode="restart",
            )

    def test_filtered_option_cohort_is_bounded_and_uses_underlying_close(self):
        equity = Instrument(
            "alpaca:equity:AAPL",
            "alpaca",
            "AAPL",
            AssetClass.EQUITY,
            "USD",
        )
        self.store.register_instrument(equity)
        self.store.append_event(
            MarketEvent(
                "aapl-close",
                MarketEventType.BAR,
                "alpaca",
                equity.instrument_id,
                self.now - timedelta(days=1),
                self.now,
                "fixture",
                {"close": 200.0},
                ingested_at=self.now,
            )
        )
        collector = FakeAlpacaCollector()
        runner = ShadowIngestionRunner(
            self.store,
            self.ledger,
            collector_factory=lambda venue, dataset: collector,
        )
        job = ObservationJob(
            "aapl-liquid-options",
            "alpaca",
            "chain",
            symbol="AAPL",
            cursor_mode="restart",
            expiration_lookahead_days=14,
            strike_band_pct=0.1,
            updated_since_minutes=120,
        )

        record = runner.run_plan(
            ShadowIngestionPlan("liquid-options", (job,)),
            collected_at=self.now,
        )[0]

        self.assertEqual(record.status, IngestionRunStatus.SUCCESS)
        self.assertEqual(collector.chain_kwargs["expiration_date_gte"], "2026-07-21")
        self.assertEqual(collector.chain_kwargs["expiration_date_lte"], "2026-08-04")
        self.assertAlmostEqual(collector.chain_kwargs["strike_price_gte"], 180.0)
        self.assertAlmostEqual(collector.chain_kwargs["strike_price_lte"], 220.0)
        self.assertEqual(
            collector.chain_kwargs["updated_since"],
            self.now - timedelta(minutes=120),
        )
        self.assertIsNone(collector.chain_kwargs["page_token"])

    def test_option_cohort_filters_are_strictly_scoped_and_bounded(self):
        valid = ObservationJob(
            "liquid-options",
            "alpaca",
            "chain",
            symbol="AAPL",
            cursor_mode="restart",
            expiration_lookahead_days=14,
            strike_band_pct=0.1,
            updated_since_minutes=120,
        )
        self.assertEqual(valid.expiration_lookahead_days, 14)
        with self.assertRaisesRegex(ValueError, "must restart pagination"):
            ObservationJob(
                "resume-filtered-options",
                "alpaca",
                "chain",
                symbol="AAPL",
                expiration_lookahead_days=14,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 60"):
            ObservationJob(
                "wide-options",
                "alpaca",
                "chain",
                symbol="AAPL",
                cursor_mode="restart",
                expiration_lookahead_days=61,
            )
        with self.assertRaisesRegex(ValueError, "between zero and 0.5"):
            ObservationJob(
                "wide-strikes",
                "alpaca",
                "chain",
                symbol="AAPL",
                cursor_mode="restart",
                strike_band_pct=0.6,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 1440"):
            ObservationJob(
                "stale-options",
                "alpaca",
                "chain",
                symbol="AAPL",
                cursor_mode="restart",
                updated_since_minutes=1441,
            )
        with self.assertRaisesRegex(ValueError, "only valid for Alpaca chain"):
            ObservationJob(
                "stock-bars",
                "alpaca",
                "bars",
                symbol="AAPL",
                cursor_mode="resume",
                strike_band_pct=0.1,
            )

    def test_mve_filter_is_restricted_to_kalshi_market_jobs(self):
        self.assertEqual(
            ObservationJob(
                "binary-markets",
                "kalshi",
                "markets",
                mve_filter="exclude",
            ).mve_filter,
            "exclude",
        )
        with self.assertRaisesRegex(ValueError, "only valid for Kalshi market"):
            ObservationJob(
                "trades",
                "kalshi",
                "trades",
                mve_filter="exclude",
            )

    def test_close_lookahead_is_bounded_and_restricted_to_open_kalshi_markets(self):
        job = ObservationJob(
            "closing-markets",
            "kalshi",
            "markets",
            close_lookahead_hours=48,
        )
        self.assertEqual(job.close_lookahead_hours, 48)
        with self.assertRaisesRegex(ValueError, "between 1 and 168"):
            ObservationJob(
                "too-wide",
                "kalshi",
                "markets",
                close_lookahead_hours=169,
            )
        with self.assertRaisesRegex(ValueError, "must target open"):
            ObservationJob(
                "settled",
                "kalshi",
                "markets",
                status="settled",
                close_lookahead_hours=48,
            )
        with self.assertRaisesRegex(ValueError, "must be only or exclude"):
            ObservationJob(
                "markets",
                "kalshi",
                "markets",
                mve_filter="invalid",
            )

    def test_forecast_outcome_job_polls_due_then_future_unscored_binary_markets(self):
        audit = AuditLedger(self.db_path)
        audit.initialize()
        closed = Instrument(
            "kalshi:prediction:KXCLOSED-YES",
            "kalshi",
            "KXCLOSED-YES",
            AssetClass.PREDICTION,
            "USD",
        )
        active = Instrument(
            "kalshi:prediction:KXACTIVE-YES",
            "kalshi",
            "KXACTIVE-YES",
            AssetClass.PREDICTION,
            "USD",
        )
        mve = Instrument(
            "kalshi:prediction:KXMVE-COMBO",
            "kalshi",
            "KXMVE-COMBO",
            AssetClass.PREDICTION,
            "USD",
        )
        for instrument, close_time, mve_ticker in (
            (closed, self.now - timedelta(hours=1), None),
            (active, self.now + timedelta(hours=1), None),
            (mve, self.now - timedelta(hours=1), "KXMVE-COLLECTION"),
        ):
            self.store.register_instrument(instrument)
            rule = MarketEvent(
                f"rule-{instrument.symbol}",
                MarketEventType.CONTRACT_RULE,
                "kalshi",
                instrument.instrument_id,
                self.now - timedelta(hours=2),
                self.now - timedelta(hours=2),
                "fixture",
                {
                    "close_time": close_time.isoformat(),
                    "mve_collection_ticker": mve_ticker,
                },
                ingested_at=self.now - timedelta(hours=2),
            )
            self.store.append_event(rule)
            audit.append_forecast(
                Forecast(
                    f"forecast-{instrument.symbol}",
                    "prediction-market-calibration-baseline-v3",
                    "baseline-v3",
                    instrument.instrument_id,
                    ForecastKind.BINARY_PROBABILITY,
                    self.now - timedelta(hours=2),
                    close_time,
                    {
                        "probability": 0.5,
                        "market_probability": 0.5,
                        "event_ticker": instrument.symbol,
                        "target_time": close_time.isoformat(),
                    },
                    0.25,
                    {"market_spread": 0.1},
                    (rule.event_id,),
                    ("no edge",),
                )
            )

        unsafe = Instrument(
            "kalshi:prediction:KXUNSAFE-YES",
            "kalshi",
            "KXUNSAFE-YES",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(unsafe)
        unsafe_rule = MarketEvent(
            "rule-KXUNSAFE-YES",
            MarketEventType.CONTRACT_RULE,
            "kalshi",
            unsafe.instrument_id,
            self.now - timedelta(hours=2),
            self.now - timedelta(hours=2),
            "fixture",
            {
                "close_time": (self.now - timedelta(hours=1)).isoformat(),
                "occurrence_datetime": (self.now - timedelta(hours=3)).isoformat(),
            },
            ingested_at=self.now - timedelta(hours=2),
        )
        self.store.append_event(unsafe_rule)
        audit.append_forecast(
            Forecast(
                "unsafe-v2-forecast",
                "prediction-market-calibration-baseline-v2",
                "baseline-v2",
                unsafe.instrument_id,
                ForecastKind.BINARY_PROBABILITY,
                self.now - timedelta(hours=2),
                self.now - timedelta(hours=1),
                {
                    "probability": 0.9,
                    "market_probability": 0.9,
                    "outcome_cluster": (self.now - timedelta(hours=3)).isoformat(),
                },
                0.25,
                {"market_spread": 0.02},
                (unsafe_rule.event_id,),
                ("post-occurrence information invalidates the forecast",),
            )
        )

        settlement = MarketEvent(
            "closed-settlement",
            MarketEventType.SETTLEMENT,
            "kalshi",
            closed.instrument_id,
            self.now - timedelta(minutes=30),
            self.now,
            "fixture",
            {
                "result": "yes",
                "event_ticker": "KXCLOSED",
                "occurrence_datetime": (self.now - timedelta(minutes=30)).isoformat(),
            },
            ingested_at=self.now,
        )
        collector = FakeKalshiCollector(
            CollectionBatch("kalshi", (closed,), (settlement,))
        )
        runner = ShadowIngestionRunner(
            self.store,
            self.ledger,
            collector_factory=lambda venue, dataset: collector,
            audit=audit,
        )
        plan = ShadowIngestionPlan(
            "forecast-outcomes",
            (
                ObservationJob(
                    "forecast-outcomes",
                    "kalshi",
                    "forecast_outcomes",
                    limit=100,
                    cursor_mode="restart",
                ),
            ),
        )

        record = runner.run_plan(plan, collected_at=self.now)[0]

        self.assertEqual(record.status, IngestionRunStatus.SUCCESS)
        self.assertEqual(
            collector.market_kwargs["tickers"],
            ("KXCLOSED-YES", "KXACTIVE-YES"),
        )
        self.assertIsNone(collector.market_kwargs["status"])
        self.assertEqual(collector.market_kwargs["mve_filter"], "exclude")
        self.assertIsNone(collector.market_kwargs["cursor"])
        scored = ShadowResearchRunner(self.store, audit).score_available(as_of=self.now)
        self.assertEqual(scored.appended, 1)
        with self.assertRaisesRegex(ValueError, "must be resume or restart"):
            ObservationJob(
                "options",
                "alpaca",
                "chain",
                symbol="AAPL",
                cursor_mode="invalid",
            )

    def test_forecast_outcome_polling_rotates_overdue_tickers_past_api_cap(self):
        audit = AuditLedger(self.db_path)
        audit.initialize()
        runner = ShadowIngestionRunner(self.store, self.ledger, audit=audit)
        for index in range(101):
            symbol = f"KXCAP-{index:03d}"
            instrument = Instrument(
                f"kalshi:prediction:{symbol}",
                "kalshi",
                symbol,
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(instrument)
            target_time = self.now - timedelta(hours=201 - index)
            observed_at = self.now if index < 100 else target_time - timedelta(minutes=1)
            rule = MarketEvent(
                f"rule-{symbol}",
                MarketEventType.CONTRACT_RULE,
                "kalshi",
                instrument.instrument_id,
                observed_at,
                observed_at,
                "fixture",
                {},
                ingested_at=observed_at,
            )
            self.store.append_event(rule)
            audit.append_forecast(
                Forecast(
                    f"forecast-{symbol}",
                    "prediction-market-calibration-baseline-v3",
                    "baseline-v3",
                    instrument.instrument_id,
                    ForecastKind.BINARY_PROBABILITY,
                    target_time - timedelta(hours=1),
                    target_time,
                    {"probability": 0.5, "target_time": target_time.isoformat()},
                    0.25,
                    {"market_spread": 0.1},
                    (rule.event_id,),
                    ("fixture",),
                )
            )

        tickers = runner._pending_prediction_tickers(self.now, 100)

        self.assertEqual(len(tickers), 100)
        self.assertIn("KXCAP-100", tickers)
        self.assertNotIn("KXCAP-099", tickers)

    def test_checked_in_option_plan_pairs_breadth_and_repeat_cohorts(self):
        plan = load_plan("config/shadow-ingestion.json")
        option_jobs = [job for job in plan.jobs if job.dataset == "chain"]
        by_symbol = {
            symbol: {job.cursor_mode for job in option_jobs if job.symbol == symbol}
            for symbol in ("SPY", "QQQ", "AAPL", "NVDA")
        }
        self.assertEqual(
            by_symbol,
            {symbol: {"resume", "restart"} for symbol in by_symbol},
        )

    def test_checked_in_crypto_plan_covers_validated_liquid_universe(self):
        plan = load_plan("config/shadow-ingestion.json")
        hourly_jobs = [
            job
            for job in plan.jobs
            if job.venue == "coinbase"
            and job.dataset == "candles"
            and job.granularity == "ONE_HOUR"
        ]

        self.assertEqual(
            {job.symbol for job in hourly_jobs},
            {
                "BTC-USD",
                "ETH-USD",
                "SOL-USD",
                "DOGE-USD",
                "XRP-USD",
                "ADA-USD",
                "AVAX-USD",
                "LINK-USD",
                "LTC-USD",
                "BCH-USD",
                "HYPE-USD",
                "ZEC-USD",
                "XLM-USD",
                "ONDO-USD",
                "HBAR-USD",
                "NEAR-USD",
                "SUI-USD",
                "UNI-USD",
                "TAO-USD",
                "PUMP-USD",
            },
        )
        self.assertTrue(
            all(job.limit == 30 for job in hourly_jobs)
        )
        intraday_jobs = [
            job
            for job in plan.jobs
            if job.venue == "coinbase"
            and job.dataset == "candles"
            and job.granularity == "FIFTEEN_MINUTE"
        ]
        self.assertEqual(
            {job.symbol for job in intraday_jobs},
            {
                "BTC-USD",
                "ETH-USD",
                "SOL-USD",
                "DOGE-USD",
                "XRP-USD",
                "ADA-USD",
                "AVAX-USD",
                "LINK-USD",
                "HYPE-USD",
                "PUMP-USD",
            },
        )
        self.assertTrue(all(job.limit == 350 for job in intraday_jobs))

    def test_checked_in_perpetual_plan_pairs_btc_and_eth_books(self):
        plan = load_plan("config/shadow-ingestion.json")
        book_jobs = [
            job
            for job in plan.jobs
            if job.venue == "coinbase" and job.dataset == "book"
        ]

        self.assertEqual(
            {job.symbol for job in book_jobs},
            {
                "BTC-USD",
                "BIP-20DEC30-CDE",
                "ETH-USD",
                "ETP-20DEC30-CDE",
            },
        )
        self.assertTrue(all(job.limit == 100 for job in book_jobs))

    def test_checked_in_prediction_plan_excludes_mve_and_tracks_outcomes(self):
        plan = load_plan("config/shadow-ingestion.json")
        open_job = next(job for job in plan.jobs if job.job_id == "kalshi-open-markets")
        fast_job = next(
            job for job in plan.jobs if job.job_id == "kalshi-fast-settling-markets"
        )
        outcome_job = next(
            job for job in plan.jobs if job.job_id == "kalshi-forecast-outcomes"
        )
        latest_job = next(
            job for job in plan.jobs if job.job_id == "kalshi-latest-settlements"
        )

        self.assertEqual(open_job.mve_filter, "exclude")
        self.assertEqual(open_job.close_lookahead_hours, 48)
        self.assertEqual(open_job.cursor_mode, "restart")
        self.assertEqual(fast_job.mve_filter, "exclude")
        self.assertEqual(fast_job.limit, 1000)
        self.assertIsNone(fast_job.close_lookahead_hours)
        self.assertEqual(fast_job.cursor_mode, "resume")
        self.assertEqual(outcome_job.dataset, "forecast_outcomes")
        self.assertEqual(outcome_job.cursor_mode, "restart")
        self.assertEqual(latest_job.cursor_mode, "restart")
        self.assertEqual(latest_job.mve_filter, "exclude")

    def test_checked_in_holder_concentration_batch_stays_below_public_rpc_capacity(self):
        plan = load_plan("config/shadow-ingestion.json")
        holder_job = next(
            job for job in plan.jobs if job.job_id == "solana-finalized-holder-concentrations"
        )

        self.assertEqual(holder_job.venue, "solana")
        self.assertEqual(holder_job.dataset, "holder_concentrations")
        self.assertEqual(holder_job.limit, 10)
        self.assertEqual(holder_job.activation_profile, "solana_read_only_rpc")

    def test_alpaca_activation_requires_both_read_only_environment_values(self):
        job = ObservationJob(
            "options",
            "alpaca",
            "chain",
            symbol="AAPL",
            activation_profile="alpaca_market_data",
        )
        self.assertFalse(job.is_active({}))
        self.assertEqual(
            job.missing_activation_environment(
                {"ALPACA_MARKET_DATA_KEY_ID": "present"}
            ),
            ("ALPACA_MARKET_DATA_SECRET_KEY",),
        )
        self.assertTrue(
            job.is_active(
                {
                    "ALPACA_MARKET_DATA_KEY_ID": "present",
                    "ALPACA_MARKET_DATA_SECRET_KEY": "present",
                }
            )
        )

    def test_solana_activation_requires_dedicated_read_only_endpoint(self):
        job = ObservationJob(
            "holder-concentrations",
            "solana",
            "holder_concentrations",
            limit=10,
            activation_profile="solana_read_only_rpc",
        )
        self.assertFalse(job.is_active({}))
        self.assertEqual(
            job.missing_activation_environment({}),
            ("SOLANA_READ_ONLY_RPC_URL",),
        )
        self.assertTrue(job.is_active({"SOLANA_READ_ONLY_RPC_URL": "https://rpc.example"}))
        with self.assertRaisesRegex(ValueError, "only valid for Solana"):
            ObservationJob(
                "wrong-venue",
                "coinbase",
                "products",
                activation_profile="solana_read_only_rpc",
            )

    def test_runner_skips_credential_gated_jobs_without_constructing_collector(self):
        plan = ShadowIngestionPlan(
            "credential-gated",
            (
                ObservationJob(
                    "options",
                    "alpaca",
                    "chain",
                    symbol="AAPL",
                    activation_profile="alpaca_market_data",
                ),
            ),
        )
        runner = ShadowIngestionRunner(
            self.store,
            self.ledger,
            collector_factory=lambda venue, dataset: self.fail(
                "inactive collector must not be constructed"
            ),
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(runner.run_plan(plan, collected_at=self.now), ())

    def test_missing_solana_rpc_does_not_block_other_observation_jobs(self):
        plan = ShadowIngestionPlan(
            "mixed-optional-observation",
            (
                ObservationJob(
                    "solana-holders",
                    "solana",
                    "holder_concentrations",
                    limit=10,
                    activation_profile="solana_read_only_rpc",
                ),
                ObservationJob("coinbase-products", "coinbase", "products"),
            ),
        )
        coinbase = FakeCoinbaseCollector(CollectionBatch("coinbase"))

        def factory(venue, dataset):
            if venue == "solana":
                self.fail("inactive Solana collector must not be constructed")
            self.assertEqual((venue, dataset), ("coinbase", "products"))
            return coinbase

        runner = ShadowIngestionRunner(self.store, self.ledger, collector_factory=factory)
        with patch.dict("os.environ", {}, clear=True):
            records = runner.run_plan(plan, collected_at=self.now)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].job_id, "coinbase-products")
        self.assertEqual(records[0].status, IngestionRunStatus.SUCCESS)

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
