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
    ScorecardStatus,
    build_daily_scorecard,
    render_github_alerts,
    render_scorecard,
)
from trading_bot.evaluation.scoring import ScoreKind
from trading_bot.evaluation.scoring import score_return_forecast
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
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
                    "event_ticker": "FAST-EVENT",
                    "status": "active",
                    "can_close_early": False,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
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
            if item.specialist_id == "prediction-market-fast-settlement-baseline-v1"
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
        self.assertEqual(
            memecoin.latest_profile_observed_at, self.now - timedelta(minutes=5)
        )
        self.assertEqual(
            memecoin.latest_pool_observed_at, self.now - timedelta(minutes=5)
        )
        self.assertEqual(memecoin.blocked_unverified_tokens, 1)
        self.assertEqual(memecoin.safety_eligible_tokens, 0)
        self.assertEqual(
            memecoin.missing_hard_gates,
            (
                "holder concentration",
                "onchain authorities",
                "round-trip simulation",
                "transfer behavior",
            ),
        )
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Pre-registered research lanes", markdown)
        self.assertIn("prediction-market-fast-settlement-baseline-v1", markdown)
        self.assertIn("Fast-settlement prediction eligibility", markdown)
        self.assertIn("Memecoin shadow research", markdown)
        self.assertIn("Recorded public discoveries", markdown)
        self.assertIn("Most recent pool snapshot", markdown)
        self.assertIn("1** blocked-unverified", markdown)

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
        self.assertIn("### Pending outcomes by strategy", markdown)
        self.assertIn("due without outcome: **1**", markdown)
        self.assertIn(
            "prediction-market-calibration-baseline-v3", markdown
        )

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
