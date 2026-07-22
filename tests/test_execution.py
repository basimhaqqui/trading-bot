import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from trading_bot.core.schemas import AssetClass, Instrument
from trading_bot.execution.control import DeterministicExecutor, PaperLedgerAdapter
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ApprovedOrderIntent,
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
)


class ExecutionControlTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.signer = ApprovalSigner(b"test-risk-key-long-enough")
        self.governor = RiskGovernor(
            RiskLimits(
                100_000,
                10_000,
                30_000,
                {AssetClass.MEMECOIN: 0, AssetClass.PREDICTION: 5_000},
                allow_live=False,
            ),
            self.signer,
        )
        self.instrument = Instrument(
            "demo:SPY", "demo", "SPY", AssetClass.EQUITY, "USD"
        )
        self.portfolio = PortfolioSnapshot(self.now, 100_000, 100_000)

    def intent(self, **changes):
        values = dict(
            intent_id="intent-1",
            strategy_id="test",
            model_version="v1",
            instrument_id=self.instrument.instrument_id,
            venue=self.instrument.venue,
            asset_class=self.instrument.asset_class,
            side=OrderSide.BUY,
            notional=5_000,
            environment=ExecutionEnvironment.SHADOW,
            allowed_order_types=(OrderType.LIMIT,),
            expires_at=self.now + timedelta(minutes=1),
            created_at=self.now,
        )
        values.update(changes)
        return OrderIntent(**values)

    def test_signed_shadow_intent_executes_once(self):
        approval = self.governor.approve(
            self.intent(), instrument=self.instrument, portfolio=self.portfolio, now=self.now
        )
        executor = DeterministicExecutor(
            self.signer, PaperLedgerAdapter(ExecutionEnvironment.SHADOW)
        )
        self.assertEqual(executor.execute(approval, now=self.now).status, "recorded")
        with self.assertRaises(PermissionError):
            executor.execute(approval, now=self.now)

    def test_tampered_approval_is_rejected(self):
        approval = self.governor.approve(
            self.intent(), instrument=self.instrument, portfolio=self.portfolio, now=self.now
        )
        tampered = replace(approval, intent=replace(approval.intent, notional=9_000))
        executor = DeterministicExecutor(
            self.signer, PaperLedgerAdapter(ExecutionEnvironment.SHADOW)
        )
        with self.assertRaises(PermissionError):
            executor.execute(tampered, now=self.now)

    def test_live_intent_is_rejected(self):
        decision = self.governor.evaluate(
            self.intent(environment=ExecutionEnvironment.LIVE),
            instrument=self.instrument,
            portfolio=self.portfolio,
            now=self.now,
        )
        self.assertFalse(decision.approved)
        self.assertIn("live execution is disabled", decision.reasons)

    def test_memecoin_cap_is_zero(self):
        instrument = Instrument(
            "demo:MEME", "demo", "MEME", AssetClass.MEMECOIN, "USD"
        )
        decision = self.governor.evaluate(
            self.intent(
                instrument_id=instrument.instrument_id,
                asset_class=AssetClass.MEMECOIN,
                notional=1,
            ),
            instrument=instrument,
            portfolio=self.portfolio,
            now=self.now,
        )
        self.assertFalse(decision.approved)
        self.assertTrue(any("memecoin" in reason for reason in decision.reasons))


if __name__ == "__main__":
    unittest.main()
