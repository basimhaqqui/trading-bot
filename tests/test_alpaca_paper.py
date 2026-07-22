import unittest
from datetime import datetime, timedelta, timezone

from trading_bot.core.schemas import AssetClass, Instrument
from trading_bot.execution.alpaca import (
    AlpacaOrder,
    AlpacaPaperAdapter,
    AlpacaPaperClient,
    PaperOrderRequest,
    PinnedTradingHttpTransport,
)
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    TimeInForce,
)


class FakeTransport:
    def __init__(self):
        self.posts = []

    def get_json(self, path, *, query=None):
        if path == "/v2/account":
            return {
                "id": "paper-account",
                "status": "ACTIVE",
                "equity": "100000",
                "last_equity": "99000",
                "cash": "75000",
                "buying_power": "200000",
                "trading_blocked": False,
                "account_blocked": False,
                "trade_suspended_by_user": False,
            }
        if path == "/v2/positions":
            return []
        if path == "/v2/orders":
            return []
        raise AssertionError(path)

    def post_json(self, path, payload):
        self.posts.append((path, payload))
        raise AssertionError("unexpected post")

    def delete_json(self, path):
        return []


class FakePaperClient:
    def __init__(self, existing=None):
        self.existing = existing
        self.submitted = []

    def order_by_client_id(self, client_order_id):
        return self.existing

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


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class AlpacaPaperTests(unittest.TestCase):
    def test_transport_is_pinned_to_paper_host_and_has_no_patch_api(self):
        with self.assertRaises(ValueError):
            PinnedTradingHttpTransport(
                "key",
                "secret",
                base_url="https://api.alpaca.markets",
                allowed_host="api.alpaca.markets",
            )
        transport = PinnedTradingHttpTransport("key", "secret")
        self.assertFalse(hasattr(transport, "patch_json"))
        with self.assertRaises(ValueError):
            transport.get_json("https://evil.example/account")

    def test_account_parsing_and_daily_return(self):
        account = AlpacaPaperClient(
            "key", "secret", transport=FakeTransport()
        ).account(observed_at=NOW)
        self.assertTrue(account.can_trade)
        self.assertAlmostEqual(account.daily_return, 100000 / 99000 - 1)
        self.assertEqual(account.buying_power, 200000)

    def test_option_orders_require_whole_contracts_and_day_tif(self):
        with self.assertRaises(ValueError):
            PaperOrderRequest(
                "client",
                "AAPL260101C00100000",
                AssetClass.OPTION,
                OrderSide.BUY,
                OrderType.LIMIT,
                TimeInForce.DAY,
                1.5,
                2.0,
            )
        with self.assertRaises(ValueError):
            PaperOrderRequest(
                "client",
                "AAPL260101C00100000",
                AssetClass.OPTION,
                OrderSide.BUY,
                OrderType.LIMIT,
                TimeInForce.GTC,
                1.0,
                2.0,
            )

    def _approval(self):
        instrument = Instrument(
            "alpaca:equity:AAPL", "alpaca", "AAPL", AssetClass.EQUITY, "USD"
        )
        intent = OrderIntent(
            "paper-intent",
            "strategy",
            "v1",
            instrument.instrument_id,
            instrument.venue,
            instrument.asset_class,
            OrderSide.BUY,
            1000,
            ExecutionEnvironment.PAPER,
            (OrderType.LIMIT,),
            NOW + timedelta(minutes=1),
            max_price=101,
            created_at=NOW,
            quantity=10,
            forecast_id="forecast",
        )
        signer = ApprovalSigner(b"paper-test-key-long-enough")
        approval = RiskGovernor(RiskLimits(10000, 2000, 5000), signer).approve(
            intent,
            instrument=instrument,
            portfolio=PortfolioSnapshot(NOW, 10000, 10000),
            now=NOW,
        )
        return instrument, approval

    def test_adapter_requires_interlock_and_uses_remote_idempotency(self):
        instrument, approval = self._approval()
        client = FakePaperClient()
        locked = AlpacaPaperAdapter(client, lambda _: instrument, trading_enabled=False)
        with self.assertRaises(PermissionError):
            locked.submit(approval, now=NOW)

        adapter = AlpacaPaperAdapter(client, lambda _: instrument, trading_enabled=True)
        receipt = adapter.submit(approval, now=NOW)
        self.assertEqual(receipt.status, "accepted")
        self.assertEqual(len(client.submitted), 1)

        existing = client.submitted[0]
        client.existing = AlpacaOrder(
            "remote-order",
            existing.client_order_id,
            existing.symbol,
            existing.asset_class,
            existing.side,
            existing.order_type.value,
            existing.time_in_force.value,
            "accepted",
            existing.quantity,
            0,
            existing.limit_price,
            None,
            NOW,
            NOW,
        )
        adapter.submit(approval, now=NOW)
        self.assertEqual(len(client.submitted), 1)


if __name__ == "__main__":
    unittest.main()
