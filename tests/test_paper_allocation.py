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
from trading_bot.evaluation.scoring import ScoreKind, score_return_forecast
from trading_bot.execution.alpaca import AlpacaAccount, AlpacaOrder
from trading_bot.execution.operations import PaperControlStore
from trading_bot.execution.paper import (
    AlpacaPaperAllocator,
    PaperExecutionService,
    PaperRiskConfig,
    load_paper_risk_config,
)
from trading_bot.execution.risk import ApprovalSigner, RiskLimits
from trading_bot.execution.schemas import OrderSide, PortfolioSnapshot


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self, *, equity=100000, last_equity=100000):
        self.equity = equity
        self.last_equity = last_equity
        self.submitted = []

    def account(self, *, observed_at=None):
        return AlpacaAccount(
            "paper-account",
            "ACTIVE",
            self.equity,
            self.last_equity,
            self.equity,
            self.equity * 2,
            False,
            False,
            False,
            observed_at or NOW,
        )

    def positions(self):
        return ()

    def orders(self, *, status="all", limit=500):
        return ()

    def portfolio_snapshot(self, instruments_by_symbol, *, observed_at=None):
        return PortfolioSnapshot(
            observed_at or NOW, self.equity, self.equity * 2, ()
        )

    def order_by_client_id(self, client_order_id):
        return None

    def submit_order(self, request):
        self.submitted.append(request)
        return AlpacaOrder(
            "remote-order",
            request.client_order_id,
            request.symbol,
            request.asset_class,
            request.side,
            request.order_type.value,
            request.time_in_force.value,
            "accepted",
            request.quantity,
            0,
            request.limit_price,
            None,
            NOW,
            NOW,
        )


class PaperAllocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "paper.db"
        self.store = PointInTimeStore(self.path)
        self.store.initialize()
        self.audit = AuditLedger(self.path)
        self.audit.initialize()
        self.controls = PaperControlStore(self.path)
        self.controls.initialize()
        self.instrument = Instrument(
            "alpaca:equity:AAPL", "alpaca", "AAPL", AssetClass.EQUITY, "USD"
        )
        self.store.register_instrument(self.instrument)
        old_time = NOW - timedelta(hours=12)
        self.store.append_event(
            MarketEvent(
                "evidence-bar",
                MarketEventType.BAR,
                "alpaca",
                self.instrument.instrument_id,
                old_time,
                old_time + timedelta(minutes=1),
                "test-bars",
                {"close": 95.0},
                ingested_at=old_time + timedelta(minutes=1),
            )
        )
        fresh_time = NOW - timedelta(minutes=5)
        self.store.append_event(
            MarketEvent(
                "fresh-bar",
                MarketEventType.BAR,
                "alpaca",
                self.instrument.instrument_id,
                fresh_time,
                fresh_time,
                "test-bars",
                {"close": 100.0},
                ingested_at=fresh_time,
            )
        )
        self.costs = EconomicCostRegistry(
            "paper-test-costs-v1",
            (
                EconomicCostModel(
                    "paper-equity-costs",
                    "return-specialist",
                    ScoreKind.RETURN,
                    CostBasis.STATIC_BPS,
                    "https://example.com/costs",
                    date(2026, 1, 1),
                    fee_bps=10,
                ),
            ),
        )
        self.config = PaperRiskConfig(min_outcomes=2, min_economic_trades=2)

    def tearDown(self):
        self.temp.cleanup()

    def add_candidate_evidence(self):
        for index in range(4):
            generated = NOW - timedelta(hours=4 - index)
            actual = 0.04 if index % 2 == 0 else -0.04
            forecast = Forecast(
                f"forecast-{index}",
                "return-specialist",
                "v1",
                self.instrument.instrument_id,
                ForecastKind.RETURN_DISTRIBUTION,
                generated,
                NOW + timedelta(hours=1),
                {"predicted_return": actual, "benchmark_return": 0.0},
                0.7,
                {"sample_size": 1.0},
                ("evidence-bar",),
                ("edge gate fails",),
            )
            score = score_return_forecast(
                forecast,
                actual_return=actual,
                target_time=generated + timedelta(minutes=30),
                scored_at=generated + timedelta(minutes=30),
            )
            self.audit.append_forecast(forecast)
            self.audit.append_forecast_score(score)

    def unlock(self):
        self.controls.release_kill_switch(
            confirmation="PAPER-ONLY", reason="test", now=NOW
        )
        self.controls.enable(confirmation="PAPER-ONLY", reason="test", now=NOW)

    def test_allocator_requires_both_evidence_gates(self):
        self.unlock()
        plan = AlpacaPaperAllocator(
            self.store,
            self.audit,
            FakeClient(),
            self.controls,
            self.costs,
            self.config,
        ).plan(now=NOW)
        self.assertEqual(plan.intents, ())
        self.assertIn(
            "no strategy passed both forecast and after-cost gates", plan.skipped
        )

    def test_checked_in_paper_policy_is_strict_and_versioned(self):
        policy = load_paper_risk_config("config/paper-execution.json")
        self.assertEqual(policy.version, "alpaca-paper-v1")
        self.assertEqual(policy.min_outcomes, 30)
        self.assertEqual(policy.risk_per_trade_pct, 0.0025)

    def test_candidate_creates_small_limit_intent_and_daily_loss_blocks_it(self):
        self.add_candidate_evidence()
        self.unlock()
        plan = AlpacaPaperAllocator(
            self.store,
            self.audit,
            FakeClient(),
            self.controls,
            self.costs,
            self.config,
        ).plan(now=NOW)
        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].side, OrderSide.SELL)
        self.assertLessEqual(plan.intents[0].notional, 250)
        self.assertEqual(plan.intents[0].forecast_id, "forecast-3")

        loss_plan = AlpacaPaperAllocator(
            self.store,
            self.audit,
            FakeClient(equity=98000, last_equity=100000),
            self.controls,
            self.costs,
            self.config,
        ).plan(now=NOW)
        self.assertEqual(loss_plan.intents, ())
        self.assertIn("paper account breached the daily loss limit", loss_plan.skipped)

    def test_service_submits_only_when_all_interlocks_pass(self):
        self.add_candidate_evidence()
        self.unlock()
        client = FakeClient()
        service = PaperExecutionService(
            self.store,
            self.audit,
            client,
            self.controls,
            self.costs,
            ApprovalSigner(b"paper-service-key-long-enough"),
            RiskLimits(10000, 2000, 5000),
            config=self.config,
            submission_enabled=True,
        )
        result = service.run(now=NOW)
        self.assertEqual(len(result.receipts), 1)
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(result.rejected, ())
        self.assertEqual(len(self.audit.execution_receipts()), 1)

    def test_service_activates_kill_switch_on_daily_loss(self):
        self.add_candidate_evidence()
        self.unlock()
        result = PaperExecutionService(
            self.store,
            self.audit,
            FakeClient(equity=98000, last_equity=100000),
            self.controls,
            self.costs,
            ApprovalSigner(b"paper-service-key-long-enough"),
            RiskLimits(10000, 2000, 5000),
            config=self.config,
            submission_enabled=True,
        ).run(now=NOW)
        self.assertIn("daily loss limit activated kill switch", result.rejected)
        self.assertTrue(self.controls.status().kill_switch_active)
        self.assertFalse(self.controls.status().enabled)

    def test_service_rejects_missing_point_in_time_evidence(self):
        self.add_candidate_evidence()
        latest = Forecast(
            "forecast-missing-evidence",
            "return-specialist",
            "v1",
            self.instrument.instrument_id,
            ForecastKind.RETURN_DISTRIBUTION,
            NOW - timedelta(minutes=30),
            NOW + timedelta(hours=1),
            {"predicted_return": 0.04, "benchmark_return": 0.0},
            0.7,
            {"sample_size": 1.0},
            ("missing-event",),
            ("edge gate fails",),
        )
        self.audit.append_forecast(latest)
        self.unlock()
        client = FakeClient()
        result = PaperExecutionService(
            self.store,
            self.audit,
            client,
            self.controls,
            self.costs,
            ApprovalSigner(b"paper-service-key-long-enough"),
            RiskLimits(10000, 2000, 5000),
            config=self.config,
            submission_enabled=True,
        ).run(now=NOW)
        self.assertEqual(result.receipts, ())
        self.assertEqual(client.submitted, [])
        self.assertTrue(any("missing-event is missing" in item for item in result.rejected))


if __name__ == "__main__":
    unittest.main()
