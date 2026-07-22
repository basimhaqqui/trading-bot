import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
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
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Daily shadow scorecard", markdown)
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


if __name__ == "__main__":
    unittest.main()
