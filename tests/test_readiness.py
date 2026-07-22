import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.readiness import data_readiness


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PointInTimeStore(Path(self.temp.name) / "readiness.db")
        self.store.initialize()
        self.now = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def append(self, event_id, event_type, instrument, payload, event_time, available_at):
        self.store.append_event(
            MarketEvent(
                event_id,
                event_type,
                instrument.venue,
                instrument.instrument_id,
                event_time,
                available_at,
                "fixture",
                payload,
                ingested_at=available_at,
            )
        )

    def test_perpetual_readiness_requires_distinct_periods_and_both_books(self):
        perpetual = Instrument(
            "coinbase:product:BIP-20DEC30-CDE",
            "coinbase",
            "BIP-20DEC30-CDE",
            AssetClass.PERPETUAL,
            "USD",
        )
        spot = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(perpetual)
        self.store.register_instrument(spot)
        for index, funding_time in enumerate(
            ("2026-07-21T19:00:00Z", "2026-07-21T20:00:00Z")
        ):
            self.append(
                f"funding-{index}",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_time": funding_time, "funding_rate": 0.0001},
                self.now - timedelta(hours=2 - index),
                self.now - timedelta(hours=2 - index),
            )
        for instrument in (perpetual, spot):
            self.append(
                f"book-{instrument.symbol}",
                MarketEventType.BOOK_SNAPSHOT,
                instrument,
                {"bid_price": 100, "ask_price": 101},
                self.now,
                self.now,
            )
        report = data_readiness(self.store, as_of=self.now)
        self.assertTrue(report[0].ready)
        self.assertEqual(report[0].observations, 2)

    def test_post_settlement_book_is_not_a_calibration_label(self):
        market = Instrument(
            "kalshi:prediction:TEST",
            "kalshi",
            "TEST",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        settlement_time = self.now - timedelta(hours=1)
        self.append(
            "settlement",
            MarketEventType.SETTLEMENT,
            market,
            {"result": "yes"},
            settlement_time,
            settlement_time,
        )
        self.append(
            "late-book",
            MarketEventType.BOOK_SNAPSHOT,
            market,
            {"yes_bids": [["0.99", "1"]], "no_bids": [["0.00", "1"]]},
            self.now,
            self.now,
        )
        report = data_readiness(self.store, as_of=self.now)
        self.assertEqual(report[2].observations, 0)
        self.assertFalse(report[2].ready)

    def test_repeated_settlement_polls_count_as_one_resolved_market(self):
        market = Instrument(
            "kalshi:prediction:REPEATED",
            "kalshi",
            "REPEATED",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        settlement_time = self.now - timedelta(hours=1)
        self.append(
            "pre-settlement-book",
            MarketEventType.BOOK_SNAPSHOT,
            market,
            {"yes_bids": [["0.50", "1"]], "no_bids": [["0.49", "1"]]},
            settlement_time - timedelta(hours=1),
            settlement_time - timedelta(hours=1),
        )
        for index in range(2):
            self.append(
                f"settlement-poll-{index}",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes"},
                settlement_time,
                settlement_time + timedelta(minutes=index),
            )
        report = data_readiness(self.store, as_of=self.now)
        self.assertEqual(report[2].observations, 1)


if __name__ == "__main__":
    unittest.main()
