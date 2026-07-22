import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import AssetClass
from trading_bot.execution.alpaca import AlpacaAccount, AlpacaOrder
from trading_bot.execution.control import ExecutionReceipt
from trading_bot.execution.operations import (
    PaperControlStore,
    PaperExecutionLedger,
    PaperReconciler,
)
from trading_bot.execution.schemas import ExecutionEnvironment, OrderSide


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def account():
    return AlpacaAccount(
        "paper-account",
        "ACTIVE",
        100000,
        100000,
        100000,
        200000,
        False,
        False,
        False,
        NOW,
    )


def order(client_id="tb-expected"):
    return AlpacaOrder(
        "order-1",
        client_id,
        "AAPL",
        AssetClass.EQUITY,
        OrderSide.BUY,
        "limit",
        "day",
        "accepted",
        10,
        0,
        100,
        None,
        NOW,
        NOW,
    )


class FakeClient:
    def __init__(self, orders):
        self._orders = tuple(orders)

    def account(self, *, observed_at=None):
        return account()

    def positions(self):
        return ()

    def orders(self, *, status="all"):
        return self._orders


class PaperOperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "paper.db"

    def tearDown(self):
        self.temp.cleanup()

    def test_control_defaults_locked_and_requires_confirmation(self):
        controls = PaperControlStore(self.path)
        status = controls.status()
        self.assertFalse(status.ready)
        self.assertTrue(status.kill_switch_active)
        with self.assertRaises(PermissionError):
            controls.release_kill_switch(confirmation="yes", reason="test", now=NOW)
        controls.release_kill_switch(
            confirmation="PAPER-ONLY", reason="test", now=NOW
        )
        ready = controls.enable(confirmation="PAPER-ONLY", reason="test", now=NOW)
        self.assertTrue(ready.ready)
        killed = controls.activate_kill_switch(reason="incident")
        self.assertFalse(killed.enabled)
        self.assertTrue(killed.kill_switch_active)

    def test_reconciliation_is_append_only_and_detects_remote_mismatch(self):
        audit = AuditLedger(self.path)
        audit.initialize()
        audit.append_execution_receipt(
            ExecutionReceipt(
                "intent",
                ExecutionEnvironment.PAPER,
                "accepted",
                NOW,
                "order-1",
                "tb-expected",
            )
        )
        ledger = PaperExecutionLedger(self.path)
        clean = PaperReconciler(FakeClient((order(),)), ledger, audit).run(observed_at=NOW)
        self.assertTrue(clean.clean)
        self.assertEqual(clean.order_events_added, 1)
        repeated = PaperReconciler(FakeClient((order(),)), ledger, audit).run(
            observed_at=NOW
        )
        self.assertEqual(repeated.order_events_added, 0)
        mismatch = PaperReconciler(
            FakeClient((order("tb-unexpected"),)), ledger, audit
        ).run(observed_at=NOW)
        self.assertFalse(mismatch.clean)
        self.assertEqual(mismatch.missing_remote_client_order_ids, ("tb-expected",))
        self.assertEqual(mismatch.unexpected_remote_client_order_ids, ("tb-unexpected",))
        self.assertGreaterEqual(ledger.verify_integrity(), 3)


if __name__ == "__main__":
    unittest.main()
