import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
from trading_bot.evaluation.costs import (
    CostBasis,
    EconomicCostModel,
    EconomicCostRegistry,
)
from trading_bot.evaluation.scorecard import (
    AlertSeverity,
    RapidCryptoCadenceSummary,
    ScorecardStatus,
    build_daily_scorecard,
    build_rapid_lane_continuity_report,
    rapid_lane_continuity_passes,
    render_github_alerts,
    render_rapid_lane_continuity,
    render_scorecard,
)
from trading_bot.evaluation.outcomes import forecast_label_deadline
from trading_bot.evaluation.scoring import ScoreKind
from trading_bot.evaluation.scoring import score_return_forecast
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
    IngestionObservationOrigin,
    IngestionRunRecord,
    IngestionRunStatus,
)


class DailyScorecardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "scorecard.db"
        self.now = datetime(2026, 7, 22, 7, tzinfo=timezone.utc)
        self.store = PointInTimeStore(self.path)
        self.store.initialize()
        self.audit = AuditLedger(self.path)
        self.audit.initialize()
        self.ingestion = IngestionRunLedger(self.path)
        self.ingestion.initialize()
        self.costs = EconomicCostRegistry(
            "scorecard-costs-v1",
            (
                EconomicCostModel(
                    "prediction-costs",
                    "prediction-market-calibration-baseline",
                    ScoreKind.BINARY,
                    CostBasis.BINARY_CONTRACT,
                    "https://example.com/fees",
                    date(2026, 7, 1),
                    binary_fee_coefficient=0.07,
                ),
            ),
        )

    def tearDown(self):
        self.temp.cleanup()

    def append_public_run(self):
        finished_at = self.now - timedelta(minutes=5)
        self.ingestion.append(
            IngestionRunRecord(
                "public-run",
                "scorecard-plan",
                "coinbase-products",
                "coinbase",
                "products",
                IngestionRunStatus.SUCCESS,
                finished_at - timedelta(seconds=2),
                finished_at,
                1,
                1,
            )
        )

    def append_crypto_event(self):
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(instrument)
        self.store.append_event(
            MarketEvent(
                "btc-book",
                MarketEventType.BOOK_SNAPSHOT,
                "coinbase",
                instrument.instrument_id,
                self.now - timedelta(minutes=5),
                self.now - timedelta(minutes=5),
                "fixture",
                {"bid_price": 100, "ask_price": 101},
                ingested_at=self.now - timedelta(minutes=5),
            )
        )

    def append_rapid_crypto_run(self, run_id, started_at):
        self.ingestion.append(
            IngestionRunRecord(
                run_id,
                "scorecard-plan",
                "coinbase-btc-fifteen-minute-candles",
                "coinbase",
                "candles",
                IngestionRunStatus.SUCCESS,
                started_at,
                started_at + timedelta(seconds=1),
                1,
                1,
                observation_origin=IngestionObservationOrigin.SCHEDULED,
            )
        )

    def append_fast_prediction_run(self, run_id, started_at):
        self.ingestion.append(
            IngestionRunRecord(
                run_id,
                "scorecard-plan",
                "kalshi-fast-settling-markets",
                "kalshi",
                "markets",
                IngestionRunStatus.SUCCESS,
                started_at,
                started_at + timedelta(seconds=1),
                1,
                1,
                observation_origin=IngestionObservationOrigin.SCHEDULED,
            )
        )

    def test_scorecard_combines_coverage_health_and_credential_alerts(self):
        self.append_public_run()
        self.append_crypto_event()
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob("coinbase-products", "coinbase", "products"),
                ObservationJob(
                    "alpaca-spy-options",
                    "alpaca",
                    "chain",
                    symbol="SPY",
                    activation_profile="alpaca_market_data",
                ),
            ),
        )
        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )
        self.assertEqual(scorecard.status, ScorecardStatus.ATTENTION)
        self.assertEqual(scorecard.totals.events, 1)
        crypto = next(
            item for item in scorecard.coverage if item.asset_class is AssetClass.CRYPTO
        )
        self.assertEqual(crypto.instruments, 1)
        self.assertEqual(crypto.events, 1)
        self.assertEqual(scorecard.alerts[0].severity, AlertSeverity.WARNING)
        self.assertIn("stock/options", scorecard.alerts[0].message)
        payload = json.loads(render_scorecard(scorecard, "json"))
        self.assertEqual(payload["totals"]["events"], 1)
        self.assertFalse(payload["paper"]["ready"])
        self.assertTrue(payload["paper"]["kill_switch_active"])
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Daily shadow scorecard", markdown)
        self.assertIn("Paper execution", markdown)
        self.assertIn("alpaca-spy-options", markdown)
        self.assertIn("::warning", render_github_alerts(scorecard))

    def test_fast_prediction_label_deadline_waits_for_preregistered_finalization_window(self):
        forecast = Forecast(
            "fast-awaiting-finalization",
            "prediction-market-fast-settlement-baseline-v6",
            "baseline-v6",
            "kalshi:prediction:FAST-AWAITING",
            ForecastKind.BINARY_PROBABILITY,
            self.now - timedelta(hours=1),
            self.now,
            {
                "probability": 0.5,
                "market_probability": 0.5,
                "event_ticker": "FAST-AWAITING-EVENT",
                "target_time": self.now.isoformat(),
                "settlement_deadline": (self.now + timedelta(minutes=75)).isoformat(),
            },
            0.5,
            {},
            ("fast-evidence",),
            ("test fixture",),
        )

        self.assertEqual(
            forecast_label_deadline(forecast), self.now + timedelta(minutes=75)
        )

    def test_scorecard_distinguishes_solana_read_only_activation_from_alpaca(self):
        self.append_public_run()
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob("coinbase-products", "coinbase", "products"),
                ObservationJob(
                    "alpaca-spy-options",
                    "alpaca",
                    "chain",
                    symbol="SPY",
                    activation_profile="alpaca_market_data",
                ),
                ObservationJob(
                    "solana-holder-concentrations",
                    "solana",
                    "holder_concentrations",
                    limit=10,
                    activation_profile="solana_read_only_rpc",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "market_data_credentials_waiting"
        )
        self.assertIn("1 Alpaca stock/options job(s)", alert.message)
        self.assertIn("1 Solana safety-observation job(s)", alert.message)
        self.assertIn("blocked-unverified", alert.message)

    def test_scorecard_warns_when_rapid_crypto_observation_cycles_are_gapped(self):
        self.append_rapid_crypto_run("rapid-early", self.now - timedelta(minutes=50))
        self.append_rapid_crypto_run("rapid-late", self.now - timedelta(minutes=5))
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        cadence = scorecard.rapid_crypto_cadence
        self.assertEqual(cadence.job_ids, ("coinbase-btc-fifteen-minute-candles",))
        self.assertEqual(cadence.observed_cycles, 2)
        self.assertEqual(cadence.latest_started_at, self.now - timedelta(minutes=5))
        self.assertEqual(cadence.largest_gap_minutes, 1390.0)
        self.assertEqual(cadence.max_allowed_gap_minutes, 30.0)
        self.assertEqual(cadence.lookback_hours, 24.0)
        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "rapid_crypto_observation_cadence_gap"
        )
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertIn("1390.0 minutes", alert.message)
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Rapid crypto collection cadence", markdown)
        self.assertIn("largest observed gap: **1390.0 minutes**", markdown)
        self.assertFalse(rapid_lane_continuity_passes(cadence))

    def test_rapid_lane_continuity_requires_an_observed_gap_within_bound(self):
        cadence = RapidCryptoCadenceSummary(
            ("coinbase-btc-fifteen-minute-candles",),
            96,
            self.now,
            15.0,
            30.0,
            24.0,
        )

        self.assertTrue(rapid_lane_continuity_passes(cadence))

    def test_receipt_only_rapid_continuity_report_passes_for_bounded_lanes(self):
        for minute in range(0, 24 * 60, 15):
            started_at = self.now - timedelta(minutes=minute)
            self.append_rapid_crypto_run(f"crypto-{minute}", started_at)
            self.append_fast_prediction_run(f"prediction-{minute}", started_at)
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
                ObservationJob("kalshi-fast-settling-markets", "kalshi", "markets"),
            ),
        )

        report = build_rapid_lane_continuity_report(self.path, plan, as_of=self.now)

        self.assertTrue(rapid_lane_continuity_passes(report.rapid_crypto))
        self.assertTrue(rapid_lane_continuity_passes(report.fast_prediction))
        self.assertIn("**pass**", render_rapid_lane_continuity(report, "markdown"))
        payload = json.loads(render_rapid_lane_continuity(report, "json"))
        self.assertEqual(payload["rapid_crypto"]["observed_cycles"], 96)
        self.assertEqual(payload["fast_prediction"]["observed_cycles"], 96)

    def test_scorecard_excludes_manual_cycles_from_rapid_continuity(self):
        self.ingestion.append(
            IngestionRunRecord(
                "manual-rapid",
                "scorecard-plan",
                "coinbase-btc-fifteen-minute-candles",
                "coinbase",
                "candles",
                IngestionRunStatus.SUCCESS,
                self.now - timedelta(minutes=5),
                self.now - timedelta(minutes=4),
                1,
                1,
            )
        )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path, plan, self.costs, as_of=self.now, environment={}
        )

        self.assertEqual(scorecard.rapid_crypto_cadence.observed_cycles, 0)
        self.assertTrue(
            any(
                item.code == "rapid_crypto_observation_cadence_gap"
                for item in scorecard.alerts
            )
        )

    def test_scorecard_rejects_recently_started_rapid_crypto_lane(self):
        self.append_rapid_crypto_run("rapid-old-early", self.now - timedelta(hours=26))
        self.append_rapid_crypto_run("rapid-old-late", self.now - timedelta(hours=25))
        self.append_rapid_crypto_run("rapid-recent-early", self.now - timedelta(minutes=20))
        self.append_rapid_crypto_run("rapid-recent-late", self.now - timedelta(minutes=5))
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        cadence = scorecard.rapid_crypto_cadence
        self.assertEqual(cadence.observed_cycles, 2)
        self.assertEqual(cadence.largest_gap_minutes, 1420.0)
        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "rapid_crypto_observation_cadence_gap"
        )
        self.assertIn("1420.0 minutes", alert.message)

    def test_scorecard_warns_when_rapid_crypto_collection_is_stale_after_last_cycle(self):
        self.append_rapid_crypto_run("rapid-last", self.now - timedelta(minutes=35))
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(scorecard.rapid_crypto_cadence.largest_gap_minutes, 1405.0)
        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "rapid_crypto_observation_cadence_gap"
        )
        self.assertIn("1405.0 minutes", alert.message)

    def test_scorecard_warns_when_fast_prediction_observation_cycles_are_gapped(self):
        self.append_fast_prediction_run("fast-early", self.now - timedelta(minutes=50))
        self.append_fast_prediction_run("fast-late", self.now - timedelta(minutes=5))
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "kalshi-fast-settling-markets",
                    "kalshi",
                    "markets",
                    status="open",
                    limit=1000,
                    cursor_mode="resume",
                    mve_filter="exclude",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        cadence = scorecard.fast_prediction_cadence
        self.assertEqual(cadence.job_ids, ("kalshi-fast-settling-markets",))
        self.assertEqual(cadence.observed_cycles, 2)
        self.assertEqual(cadence.latest_started_at, self.now - timedelta(minutes=5))
        self.assertEqual(cadence.largest_gap_minutes, 1390.0)
        self.assertEqual(cadence.max_allowed_gap_minutes, 30.0)
        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "fast_prediction_observation_cadence_gap"
        )
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertIn("1390.0 minutes", alert.message)
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Fast prediction collection cadence", markdown)
        self.assertIn("Bounded close-window public market jobs", markdown)
        self.assertIn("largest collection gap: **1390.0 minutes**", markdown)

    def test_scorecard_excludes_manual_cycles_from_fast_prediction_continuity(self):
        self.ingestion.append(
            IngestionRunRecord(
                "manual-fast",
                "scorecard-plan",
                "kalshi-fast-settling-markets",
                "kalshi",
                "markets",
                IngestionRunStatus.SUCCESS,
                self.now - timedelta(minutes=5),
                self.now - timedelta(minutes=4),
                1,
                1,
            )
        )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "kalshi-fast-settling-markets",
                    "kalshi",
                    "markets",
                    status="open",
                    limit=1000,
                    cursor_mode="resume",
                    mve_filter="exclude",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path, plan, self.costs, as_of=self.now, environment={}
        )

        self.assertEqual(scorecard.fast_prediction_cadence.observed_cycles, 0)
        self.assertTrue(
            any(
                item.code == "fast_prediction_observation_cadence_gap"
                for item in scorecard.alerts
            )
        )

    def test_scorecard_warns_when_fast_prediction_collection_is_stale_after_last_cycle(self):
        self.append_fast_prediction_run("fast-last", self.now - timedelta(minutes=35))
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "kalshi-fast-settling-markets",
                    "kalshi",
                    "markets",
                    status="open",
                    limit=1000,
                    cursor_mode="resume",
                    mve_filter="exclude",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(scorecard.fast_prediction_cadence.largest_gap_minutes, 1405.0)
        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "fast_prediction_observation_cadence_gap"
        )
        self.assertIn("1405.0 minutes", alert.message)

    def test_scorecard_cadence_excludes_future_ingestion_runs(self):
        self.append_rapid_crypto_run("rapid-last", self.now - timedelta(minutes=35))
        self.append_rapid_crypto_run("rapid-future", self.now + timedelta(minutes=5))
        self.append_fast_prediction_run("fast-last", self.now - timedelta(minutes=35))
        self.append_fast_prediction_run("fast-future", self.now + timedelta(minutes=5))
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
                ObservationJob(
                    "kalshi-fast-settling-markets",
                    "kalshi",
                    "markets",
                    status="open",
                    limit=1000,
                    cursor_mode="resume",
                    mve_filter="exclude",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(scorecard.rapid_crypto_cadence.observed_cycles, 1)
        self.assertEqual(scorecard.rapid_crypto_cadence.latest_started_at, self.now - timedelta(minutes=35))
        self.assertEqual(scorecard.rapid_crypto_cadence.largest_gap_minutes, 1405.0)
        self.assertEqual(scorecard.fast_prediction_cadence.observed_cycles, 1)
        self.assertEqual(scorecard.fast_prediction_cadence.latest_started_at, self.now - timedelta(minutes=35))
        self.assertEqual(scorecard.fast_prediction_cadence.largest_gap_minutes, 1405.0)

    def test_scorecard_accepts_full_window_continuity_for_both_rapid_lanes(self):
        for minutes in range(24 * 60, -1, -30):
            self.append_rapid_crypto_run(
                f"rapid-{minutes}", self.now - timedelta(minutes=minutes)
            )
            self.append_fast_prediction_run(
                f"fast-{minutes}", self.now - timedelta(minutes=minutes)
            )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (
                ObservationJob(
                    "coinbase-btc-fifteen-minute-candles",
                    "coinbase",
                    "candles",
                    symbol="BTC-USD",
                    granularity="FIFTEEN_MINUTE",
                ),
                ObservationJob(
                    "kalshi-fast-settling-markets",
                    "kalshi",
                    "markets",
                    status="open",
                    limit=1000,
                    cursor_mode="resume",
                    mve_filter="exclude",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(scorecard.rapid_crypto_cadence.largest_gap_minutes, 30.0)
        self.assertEqual(scorecard.fast_prediction_cadence.largest_gap_minutes, 30.0)
        self.assertTrue(rapid_lane_continuity_passes(scorecard.rapid_crypto_cadence))
        self.assertTrue(rapid_lane_continuity_passes(scorecard.fast_prediction_cadence))

    def test_unhealthy_ingestion_emits_error_annotation(self):
        plan = ShadowIngestionPlan(
            "missing-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )
        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )
        self.assertEqual(scorecard.status, ScorecardStatus.CRITICAL)
        self.assertIn("::error", render_github_alerts(scorecard))

    def test_scorecard_reports_empty_fast_lane_and_fail_closed_memecoin_research(self):
        self.append_public_run()
        fast_market = Instrument(
            "kalshi:prediction:FAST",
            "kalshi",
            "FAST",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(fast_market)
        for event_id, event_type, payload in (
            (
                "fast-rule",
                MarketEventType.CONTRACT_RULE,
                {
                    "market_type": "binary",
                    "event_ticker": "FAST-EVENT",
                    "status": "active",
                    "is_provisional": False,
                    "can_close_early": False,
                    "settlement_timer_seconds": 900,
                    "close_time": (self.now + timedelta(hours=1)).isoformat(),
                    "expected_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
                    "latest_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
                },
            ),
            (
                "fast-book",
                MarketEventType.BOOK_SNAPSHOT,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
            ),
        ):
            self.store.append_event(
                MarketEvent(
                    event_id,
                    event_type,
                    "kalshi",
                    fast_market.instrument_id,
                    self.now - timedelta(minutes=5),
                    self.now - timedelta(minutes=5),
                    "fixture",
                    payload,
                    ingested_at=self.now - timedelta(minutes=5),
                )
            )
        instrument = Instrument(
            "dexscreener:memecoin:solana:Token",
            "dexscreener",
            "Token",
            AssetClass.MEMECOIN,
            "USD",
        )
        self.store.register_instrument(instrument)
        payload = {
            "safety_status": "blocked_unverified",
            "onchain_authorities_observed": False,
            "holder_concentration_observed": False,
            "transfer_behavior_observed": False,
            "round_trip_simulation_observed": False,
        }
        for event_id, source in (
            ("token-profile", "dexscreener-public-token-profile-v1"),
            ("pool-observation", "dexscreener-public-token-pairs-v1"),
        ):
            self.store.append_event(
                MarketEvent(
                    event_id,
                    MarketEventType.ONCHAIN_STATE,
                    "dexscreener",
                    instrument.instrument_id,
                    self.now - timedelta(minutes=5),
                    self.now - timedelta(minutes=5),
                    source,
                    payload,
                    ingested_at=self.now - timedelta(minutes=5),
                )
            )
        authority_payload = {
            **payload,
            "onchain_authorities_observed": True,
            "transfer_behavior_observed": True,
        }
        self.store.append_event(
            MarketEvent(
                "authority-observation",
                MarketEventType.ONCHAIN_STATE,
                "solana",
                instrument.instrument_id,
                self.now - timedelta(minutes=4),
                self.now - timedelta(minutes=4),
                "solana-rpc-get-multiple-accounts-finalized-v2",
                authority_payload,
                ingested_at=self.now - timedelta(minutes=4),
            )
        )
        holder_payload = {
            **payload,
            "holder_concentration_observed": True,
        }
        self.store.append_event(
            MarketEvent(
                "holder-concentration-observation",
                MarketEventType.ONCHAIN_STATE,
                "solana",
                instrument.instrument_id,
                self.now - timedelta(minutes=3),
                self.now - timedelta(minutes=3),
                "solana-rpc-token-holder-concentration-finalized-v1",
                holder_payload,
                ingested_at=self.now - timedelta(minutes=3),
            )
        )
        self.store.append_event(
            MarketEvent(
                "holder-activity-observation",
                MarketEventType.ONCHAIN_STATE,
                "solana",
                instrument.instrument_id,
                self.now - timedelta(minutes=2),
                self.now - timedelta(minutes=2),
                "solana-rpc-finalized-holder-activity-v1",
                {
                    "safety_status": "blocked_unverified",
                    "holder_activity_observed": True,
                    "transfer_behavior_observed": False,
                    "round_trip_simulation_observed": False,
                },
                ingested_at=self.now - timedelta(minutes=2),
            )
        )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        fast = next(
            item
            for item in scorecard.research_lanes
            if item.specialist_id == "prediction-market-fast-settlement-baseline-v2"
        )
        self.assertEqual(fast.forecasts, 0)
        self.assertEqual(fast.scores, 0)
        self.assertIsNone(fast.latest_forecast_at)
        self.assertEqual(scorecard.fast_prediction_eligibility.paired_markets, 1)
        self.assertEqual(scorecard.fast_prediction_eligibility.executable_markets, 1)
        self.assertEqual(scorecard.fast_prediction_eligibility.selected_events, 1)
        memecoin = scorecard.memecoin_research
        self.assertEqual(memecoin.discovered_tokens, 1)
        self.assertEqual(memecoin.latest_profile_observations, 1)
        self.assertEqual(memecoin.latest_pool_observations, 1)
        self.assertEqual(memecoin.latest_authority_observations, 1)
        self.assertEqual(memecoin.transfer_control_observations, 1)
        self.assertEqual(memecoin.holder_concentration_observations, 1)
        self.assertEqual(memecoin.holder_activity_observations, 1)
        self.assertEqual(
            memecoin.latest_profile_observed_at, self.now - timedelta(minutes=5)
        )
        self.assertEqual(
            memecoin.latest_pool_observed_at, self.now - timedelta(minutes=5)
        )
        self.assertEqual(
            memecoin.latest_authority_observed_at, self.now - timedelta(minutes=4)
        )
        self.assertEqual(
            memecoin.latest_transfer_control_observed_at, self.now - timedelta(minutes=4)
        )
        self.assertEqual(
            memecoin.latest_holder_concentration_observed_at, self.now - timedelta(minutes=3)
        )
        self.assertEqual(
            memecoin.latest_holder_activity_observed_at, self.now - timedelta(minutes=2)
        )
        self.assertEqual(memecoin.blocked_unverified_tokens, 1)
        self.assertEqual(memecoin.safety_eligible_tokens, 0)
        self.assertEqual(
            memecoin.missing_hard_gates,
            ("round-trip simulation",),
        )
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Pre-registered research lanes", markdown)
        self.assertIn("prediction-market-fast-settlement-baseline-v2", markdown)
        self.assertIn("Fast-settlement prediction eligibility", markdown)
        self.assertIn("Memecoin shadow research", markdown)
        self.assertIn("Recorded public discoveries", markdown)
        self.assertIn("Most recent pool snapshot", markdown)
        self.assertIn("finalized authority observation", markdown)
        self.assertIn("transfer-control parse", markdown)
        self.assertIn("holder-concentration observation", markdown)
        self.assertIn("aggregate holder-activity observation", markdown)
        self.assertIn("1** blocked-unverified", markdown)

        # The scorecard uses the latest record for each safety category. A
        # newer unavailable RPC read is a diagnostic, not verified activity.
        self.store.append_event(
            MarketEvent(
                "holder-activity-unavailable",
                MarketEventType.ONCHAIN_STATE,
                "solana",
                instrument.instrument_id,
                self.now - timedelta(minutes=1),
                self.now - timedelta(minutes=1),
                "solana-rpc-finalized-holder-activity-v1",
                {
                    "safety_status": "blocked_unverified",
                    "holder_activity_observed": False,
                    "transfer_behavior_observed": False,
                    "round_trip_simulation_observed": False,
                },
                ingested_at=self.now - timedelta(minutes=1),
            )
        )
        refreshed = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        ).memecoin_research
        self.assertEqual(refreshed.holder_activity_observations, 0)
        self.assertIsNone(refreshed.latest_holder_activity_observed_at)

    def test_scorecard_does_not_aggregate_stale_memecoin_gates_into_eligibility(self):
        instrument = Instrument(
            "dexscreener:memecoin:solana:STALE",
            "dexscreener",
            "STALE",
            AssetClass.MEMECOIN,
            "USD",
        )
        self.store.register_instrument(instrument)
        stale_at = self.now - timedelta(hours=2)
        fully_screened = {
            "safety_status": "sandbox_eligible",
            "onchain_authorities_observed": True,
            "holder_concentration_observed": True,
            "transfer_behavior_observed": True,
            "round_trip_simulation_observed": True,
        }
        for event_id, venue, source in (
            ("stale-profile", "dexscreener", "dexscreener-public-token-profile-v1"),
            ("stale-pool", "dexscreener", "dexscreener-public-token-pairs-v1"),
            (
                "stale-authority",
                "solana",
                "solana-rpc-get-multiple-accounts-finalized-v2",
            ),
            (
                "stale-holders",
                "solana",
                "solana-rpc-token-holder-concentration-finalized-v1",
            ),
        ):
            self.store.append_event(
                MarketEvent(
                    event_id,
                    MarketEventType.ONCHAIN_STATE,
                    venue,
                    instrument.instrument_id,
                    stale_at,
                    stale_at,
                    source,
                    fully_screened,
                    ingested_at=stale_at,
                )
            )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            max_age=timedelta(minutes=90),
            environment={},
        )

        memecoin = scorecard.memecoin_research
        self.assertEqual(memecoin.discovered_tokens, 1)
        self.assertEqual(memecoin.safety_eligible_tokens, 0)
        self.assertEqual(memecoin.blocked_unverified_tokens, 1)
        self.assertEqual(
            memecoin.missing_hard_gates,
            (
                "holder concentration",
                "onchain authorities",
                "round-trip simulation",
                "transfer behavior",
            ),
        )

    def test_scorecard_explains_when_fast_lane_lacks_documented_close_policy(self):
        self.append_public_run()
        market = Instrument(
            "kalshi:prediction:EARLY-CLOSE",
            "kalshi",
            "EARLY-CLOSE",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        for event_id, event_type, payload in (
            (
                "early-close-rule",
                MarketEventType.CONTRACT_RULE,
                {
                    "market_type": "binary",
                    "event_ticker": "EARLY-CLOSE-EVENT",
                    "status": "active",
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": (
                        self.now + timedelta(hours=1)
                    ).isoformat(),
                    "latest_expiration_time": (
                        self.now + timedelta(hours=1)
                    ).isoformat(),
                },
            ),
            (
                "early-close-book",
                MarketEventType.BOOK_SNAPSHOT,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
            ),
        ):
            self.store.append_event(
                MarketEvent(
                    event_id,
                    event_type,
                    "kalshi",
                    market.instrument_id,
                    self.now - timedelta(minutes=5),
                    self.now - timedelta(minutes=5),
                    "fixture",
                    payload,
                    ingested_at=self.now - timedelta(minutes=5),
                )
            )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        alert = next(
            item
            for item in scorecard.alerts
            if item.code == "fast_prediction_close_policy_unavailable"
        )
        self.assertEqual(alert.severity, AlertSeverity.INFO)
        self.assertIn("documented boolean can_close_early", alert.message)
        self.assertIn("1 omit the field", alert.message)
        provisional_alert = next(
            item
            for item in scorecard.alerts
            if item.code == "fast_prediction_provisional_status_unavailable"
        )
        self.assertEqual(provisional_alert.severity, AlertSeverity.INFO)
        self.assertIn("omitted the required is_provisional=false field", provisional_alert.message)
        self.assertIn("not treated as non-provisional evidence", provisional_alert.message)
        self.assertEqual(scorecard.fast_prediction_eligibility.active_markets, 1)
        self.assertEqual(
            scorecard.fast_prediction_eligibility.explicitly_non_provisional_markets, 0
        )
        self.assertEqual(
            scorecard.fast_prediction_eligibility.missing_provisional_flag_markets, 1
        )
        self.assertEqual(scorecard.fast_prediction_eligibility.documented_close_policy_markets, 0)
        self.assertEqual(scorecard.fast_prediction_eligibility.early_close_enabled_markets, 0)
        self.assertEqual(scorecard.fast_prediction_eligibility.early_close_disabled_markets, 0)
        self.assertEqual(scorecard.fast_prediction_eligibility.missing_close_policy_markets, 1)
        self.assertEqual(scorecard.fast_prediction_eligibility.invalid_close_policy_markets, 0)

    def test_scorecard_counts_stable_specialist_ids_under_preregistered_lane(self):
        self.append_public_run()
        generated_at = self.now - timedelta(hours=1)
        forecast = Forecast(
            "intraday-forecast",
            "crypto-intraday-momentum-baseline",
            "baseline-v1",
            "coinbase:product:BTC-USD",
            ForecastKind.RETURN_DISTRIBUTION,
            generated_at,
            generated_at + timedelta(minutes=15),
            {"predicted_return": 0.001, "benchmark_return": 0.0},
            0.25,
            {},
            ("fixture-event",),
            ("test fixture",),
        )
        self.audit.append_forecast(forecast)
        self.audit.append_forecast_score(
            score_return_forecast(
                forecast,
                actual_return=0.002,
                target_time=forecast.valid_until,
                scored_at=self.now - timedelta(minutes=30),
            )
        )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        lane = next(
            item
            for item in scorecard.research_lanes
            if item.specialist_id == "crypto-intraday-momentum-baseline-v1"
        )
        self.assertEqual(lane.forecasts, 1)
        self.assertEqual(lane.scores, 1)
        self.assertEqual(lane.latest_forecast_at, generated_at)

    def test_scorecard_reports_fixed_intraday_momentum_funnel(self):
        self.append_public_run()
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(instrument)
        available_at = self.now - timedelta(minutes=5)
        for index in range(8):
            close = 100 + index
            event_time = self.now - timedelta(minutes=15 * (8 - index))
            self.store.append_event(
                MarketEvent(
                    f"btc-fifteen-minute-{index}",
                    MarketEventType.BAR,
                    "coinbase",
                    instrument.instrument_id,
                    event_time,
                    available_at,
                    "fixture",
                    {
                        "open": close - 0.5,
                        "high": close + 0.5,
                        "low": close - 1,
                        "close": close,
                        "volume": 10 + index,
                        "granularity_seconds": 900,
                    },
                    ingested_at=available_at,
                )
            )
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        funnel = scorecard.intraday_momentum_eligibility
        self.assertEqual(funnel.observed_instruments, 1)
        self.assertEqual(funnel.fresh_instruments, 1)
        self.assertEqual(funnel.adequate_lookback_instruments, 1)
        self.assertEqual(funnel.signal_instruments, 1)
        self.assertEqual(funnel.v2_assigned_instruments, 0)
        self.assertEqual(funnel.v2_signal_instruments, 0)
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Fifteen-minute crypto momentum eligibility", markdown)
        self.assertIn("eight completed bars", markdown)
        self.assertIn("v2 fixed-assignment funnel", markdown)

    def test_scorecard_separates_future_and_due_unscored_forecasts(self):
        self.append_public_run()
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )
        for forecast_id, specialist_id, valid_until in (
            (
                "due",
                "prediction-market-calibration-baseline-v3",
                self.now - timedelta(minutes=30),
            ),
            (
                "future",
                "prediction-market-calibration-baseline-v3",
                self.now + timedelta(hours=2),
            ),
            (
                "legacy",
                "prediction-market-calibration-baseline",
                self.now - timedelta(minutes=45),
            ),
        ):
            values = {"probability": 0.6, "market_probability": 0.5}
            if specialist_id.endswith("-v3"):
                values["target_time"] = valid_until.isoformat()
            self.audit.append_forecast(
                Forecast(
                    forecast_id,
                    specialist_id,
                    "baseline-v3" if specialist_id.endswith("-v3") else "baseline-v1",
                    "test-instrument",
                    ForecastKind.BINARY_PROBABILITY,
                    self.now - timedelta(hours=1),
                    valid_until,
                    values,
                    0.25,
                    {},
                    ("fixture-event",),
                    ("test fixture",),
                )
            )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(scorecard.outcome_queue.unscored, 3)
        self.assertEqual(scorecard.outcome_queue.not_due, 1)
        self.assertEqual(scorecard.outcome_queue.due_unmatched, 1)
        self.assertEqual(scorecard.outcome_queue.quarantined, 1)
        self.assertEqual(
            scorecard.outcome_queue.next_due_at, self.now + timedelta(hours=2)
        )
        self.assertEqual(
            scorecard.outcome_queue.oldest_due_at,
            self.now - timedelta(minutes=30),
        )
        self.assertEqual(len(scorecard.strategy_outcome_queues), 1)
        strategy_queue = scorecard.strategy_outcome_queues[0]
        self.assertEqual(
            strategy_queue.specialist_id,
            "prediction-market-calibration-baseline-v3",
        )
        self.assertEqual(strategy_queue.pending, 2)
        self.assertEqual(strategy_queue.not_due, 1)
        self.assertEqual(strategy_queue.due_unmatched, 1)
        self.assertEqual(
            strategy_queue.next_due_at,
            self.now + timedelta(hours=2),
        )
        self.assertEqual(
            strategy_queue.oldest_due_at,
            self.now - timedelta(minutes=30),
        )
        self.assertTrue(
            any(
                alert.code == "outcomes_awaiting_settlement"
                for alert in scorecard.alerts
            )
        )
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("### Outcome queue", markdown)
        self.assertIn("no attested tracked-ticker receipt", markdown)
        self.assertIn("### Pending outcomes by strategy", markdown)
        self.assertIn("due without outcome: **1**", markdown)
        self.assertIn(
            "prediction-market-calibration-baseline-v3", markdown
        )

    def test_scorecard_surfaces_uncorroborated_v9_early_prediction_labels(self):
        self.append_public_run()
        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )
        market = Instrument(
            "kalshi:prediction:FAST-POLICY",
            "kalshi",
            "FAST-POLICY",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        expected_expiration = self.now + timedelta(minutes=30)
        generated_at = self.now - timedelta(minutes=10)
        self.audit.append_forecast(
            Forecast(
                "fast-policy-forecast",
                "prediction-market-fast-settlement-baseline-v9",
                "baseline-v9",
                market.instrument_id,
                ForecastKind.BINARY_PROBABILITY,
                generated_at,
                expected_expiration,
                {
                    "probability": 0.5,
                    "market_probability": 0.5,
                    "event_ticker": "FAST-POLICY-EVENT",
                    "target_time": expected_expiration.isoformat(),
                    "settlement_deadline": (
                        expected_expiration + timedelta(minutes=75)
                    ).isoformat(),
                    "can_close_early": True,
                },
                0.5,
                {},
                ("fixture-event",),
                ("test fixture",),
            )
        )
        self.store.append_event(
            MarketEvent(
                "fast-policy-early-finalization",
                MarketEventType.SETTLEMENT,
                "kalshi",
                market.instrument_id,
                self.now,
                self.now,
                "fixture",
                {"result": "yes", "event_ticker": "FAST-POLICY-EVENT"},
                ingested_at=self.now,
            )
        )

        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(scorecard.outcome_queue.policy_inconsistent_early_labels, 1)
        self.assertTrue(
            any(
                alert.code == "fast_prediction_policy_inconsistent_labels"
                for alert in scorecard.alerts
            )
        )
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn(
            "policy-inconsistent or uncorroborated early labels excluded: **1**",
            markdown,
        )

    def test_scorecard_attests_incomplete_bounded_prediction_outcome_poll(self):
        finished_at = self.now - timedelta(minutes=5)
        self.ingestion.append(
            IngestionRunRecord(
                "forecast-outcomes-receipt",
                "outcome-poll-plan",
                "kalshi-forecast-outcomes",
                "kalshi",
                "forecast_outcomes",
                IngestionRunStatus.SUCCESS,
                finished_at - timedelta(seconds=2),
                finished_at,
                50,
                106,
                requested_instruments=100,
            )
        )
        plan = ShadowIngestionPlan(
            "outcome-poll-plan",
            (
                ObservationJob(
                    "kalshi-forecast-outcomes",
                    "kalshi",
                    "forecast_outcomes",
                    limit=100,
                    cursor_mode="restart",
                ),
            ),
        )

        scorecard = build_daily_scorecard(
            self.path, plan, self.costs, as_of=self.now, environment={}
        )

        polling = scorecard.prediction_outcome_polling
        self.assertEqual(polling.requested_instruments, 100)
        self.assertEqual(polling.returned_instruments, 50)
        self.assertEqual(polling.missing_instruments, 50)
        self.assertEqual(polling.finished_at, finished_at)
        self.assertTrue(
            any(
                alert.code == "prediction_outcome_polling_incomplete"
                for alert in scorecard.alerts
            )
        )
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("### Prediction outcome polling", markdown)
        self.assertIn("**100** requested → **50** returned → **50** missing", markdown)

    def test_scorecard_reports_prediction_calibration_cohort_readiness(self):
        self.append_public_run()
        probabilities = (0.05, 0.06, 0.40, 0.08, 0.09, 0.07)
        for index, probability in enumerate(probabilities):
            instrument = Instrument(
                f"kalshi:prediction:HISTORY-{index}",
                "kalshi",
                f"HISTORY-{index}",
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(instrument)
            occurrence = self.now - timedelta(hours=2)
            self.store.append_event(
                MarketEvent(
                    f"history-book-{index}",
                    MarketEventType.BOOK_SNAPSHOT,
                    "kalshi",
                    instrument.instrument_id,
                    self.now - timedelta(hours=4),
                    self.now - timedelta(hours=4),
                    "fixture",
                    {
                        "yes_bids": [[f"{probability - 0.01:.2f}", "10"]],
                        "no_bids": [[f"{1 - probability - 0.01:.2f}", "10"]],
                    },
                    ingested_at=self.now - timedelta(hours=4),
                )
            )
            self.store.append_event(
                MarketEvent(
                    f"history-settlement-{index}",
                    MarketEventType.SETTLEMENT,
                    "kalshi",
                    instrument.instrument_id,
                    self.now - timedelta(hours=1),
                    (
                        self.now
                        if index == len(probabilities) - 1
                        else self.now - timedelta(hours=1)
                    ),
                    "fixture",
                    {
                        "result": "no",
                        "event_ticker": f"HISTORY-EVENT-{index}",
                        "occurrence_datetime": occurrence.isoformat(),
                    },
                    ingested_at=(
                        self.now
                        if index == len(probabilities) - 1
                        else self.now - timedelta(hours=1)
                    ),
                )
            )

        target = Instrument(
            "kalshi:prediction:TARGET",
            "kalshi",
            "TARGET",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(target)
        self.store.append_event(
            MarketEvent(
                "target-book",
                MarketEventType.BOOK_SNAPSHOT,
                "kalshi",
                target.instrument_id,
                self.now - timedelta(minutes=1),
                self.now - timedelta(minutes=1),
                "fixture",
                {"yes_bids": [["0.09", "10"]], "no_bids": [["0.89", "10"]]},
                ingested_at=self.now - timedelta(minutes=1),
            )
        )
        self.store.append_event(
            MarketEvent(
                "target-rule",
                MarketEventType.CONTRACT_RULE,
                "kalshi",
                target.instrument_id,
                self.now - timedelta(minutes=1),
                self.now - timedelta(minutes=1),
                "fixture",
                {
                    "event_ticker": "TARGET-EVENT",
                    "occurrence_datetime": (self.now + timedelta(hours=2)).isoformat(),
                },
                ingested_at=self.now - timedelta(minutes=1),
            )
        )

        plan = ShadowIngestionPlan(
            "scorecard-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )
        scorecard = build_daily_scorecard(
            self.path,
            plan,
            self.costs,
            as_of=self.now,
            environment={},
        )

        self.assertEqual(
            scorecard.prediction_calibration.eligible_independent_events, 6
        )
        self.assertEqual(scorecard.prediction_calibration.eligible_open_events, 1)
        self.assertEqual(scorecard.prediction_calibration.strongest_bucket_events, 4)
        self.assertFalse(scorecard.prediction_calibration.ready)
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Prediction calibration readiness", markdown)
        self.assertIn("strongest fixed ten-cent bucket: **4/5**", markdown)


if __name__ == "__main__":
    unittest.main()
