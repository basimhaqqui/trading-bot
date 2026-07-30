import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.audit import AuditConflictError, AuditLedger, AuditRecordType
from trading_bot.core.schemas import AssetClass, Forecast, ForecastKind, Instrument
from trading_bot.execution.control import DeterministicExecutor, PaperLedgerAdapter
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
)
from trading_bot.evaluation.scoring import score_binary_forecast


class AuditLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "audit.db"
        self.ledger = AuditLedger(self.path)
        self.ledger.initialize()
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_record_type_timeline_index_is_initialized(self):
        with self.ledger.connect() as connection:
            indexes = {
                row[1]
                for row in connection.execute("PRAGMA index_list('audit_records')").fetchall()
            }
        self.assertIn("idx_audit_records_type_timeline", indexes)

    def test_full_decision_chain_is_append_only_and_verifiable(self):
        forecast = Forecast(
            "forecast-1",
            "specialist",
            "v1",
            "demo:SPY",
            ForecastKind.RETURN_DISTRIBUTION,
            self.now,
            self.now + timedelta(minutes=5),
            {"mean": 0.01},
            0.6,
            {"standard_deviation": 0.02},
            ("event-1",),
            ("spread widens",),
        )
        instrument = Instrument("demo:SPY", "demo", "SPY", AssetClass.EQUITY, "USD")
        intent = OrderIntent(
            "intent-1",
            "strategy",
            "v1",
            instrument.instrument_id,
            instrument.venue,
            instrument.asset_class,
            OrderSide.BUY,
            1000,
            ExecutionEnvironment.SHADOW,
            (OrderType.LIMIT,),
            self.now + timedelta(minutes=1),
            created_at=self.now,
        )
        signer = ApprovalSigner(b"audit-test-key-long-enough")
        governor = RiskGovernor(RiskLimits(10_000, 2_000, 5_000), signer)
        approval = governor.approve(
            intent,
            instrument=instrument,
            portfolio=PortfolioSnapshot(self.now, 10_000, 10_000),
            now=self.now,
        )
        receipt = DeterministicExecutor(
            signer, PaperLedgerAdapter(ExecutionEnvironment.SHADOW)
        ).execute(approval, now=self.now)

        self.assertTrue(self.ledger.append_forecast(forecast))
        self.assertTrue(self.ledger.append_order_intent(intent))
        self.assertTrue(self.ledger.append_risk_decision(approval.decision))
        self.assertTrue(self.ledger.append_approval(approval))
        self.assertTrue(self.ledger.append_execution_receipt(receipt))
        self.assertFalse(self.ledger.append_forecast(forecast))
        self.assertEqual(self.ledger.verify_integrity(), 5)
        self.assertEqual(sum(self.ledger.counts().values()), 5)
        self.assertEqual(self.ledger.counts()[AuditRecordType.FORECAST_SCORE], 0)

        with self.assertRaises(AuditConflictError):
            self.ledger.append_forecast(replace(forecast, confidence=0.7))
        with self.assertRaises(Exception):
            with self.ledger.connect() as connection:
                connection.execute(
                    "UPDATE audit_records SET payload_json = '{}' WHERE record_type = ?",
                    (AuditRecordType.FORECAST.value,),
                )

    def test_forecasts_and_scores_round_trip(self):
        forecast = Forecast(
            "binary-forecast",
            "binary-specialist",
            "v1",
            "kalshi:prediction:TEST",
            ForecastKind.BINARY_PROBABILITY,
            self.now,
            self.now + timedelta(hours=1),
            {"probability": 0.6, "market_probability": 0.5},
            0.5,
            {"sample_size": 1.0},
            ("book-event",),
            ("fails validation",),
        )
        score = score_binary_forecast(
            forecast,
            outcome=True,
            target_time=self.now + timedelta(hours=2),
            scored_at=self.now + timedelta(hours=2),
        )
        self.ledger.append_forecast(forecast)
        self.ledger.append_forecast_score(score)
        self.assertEqual(self.ledger.forecasts(), (forecast,))
        self.assertEqual(self.ledger.forecast_scores(), (score,))
        self.assertEqual(self.ledger.scored_forecast_ids(), frozenset({forecast.forecast_id}))


if __name__ == "__main__":
    unittest.main()
