import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.audit import AuditLedger, AuditRecordType
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.shadow import ShadowResearchRunner


class ShadowResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "shadow.db"
        self.store = PointInTimeStore(self.path)
        self.store.initialize()
        self.audit = AuditLedger(self.path)
        self.audit.initialize()
        self.runner = ShadowResearchRunner(self.store, self.audit)
        self.now = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def event(
        self,
        event_id,
        event_type,
        instrument,
        payload,
        *,
        event_time,
        available_at=None,
    ):
        available_at = available_at or event_time
        return MarketEvent(
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

    def test_perpetual_forecast_is_idempotent_and_scores_next_distinct_period(self):
        perpetual = Instrument(
            "coinbase:product:BTC-PERP",
            "coinbase",
            "BTC-PERP",
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
        for event in (
            self.event(
                "funding-1",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0002", "funding_time": "2026-07-21T18:00:00Z"},
                event_time=self.now - timedelta(hours=2),
            ),
            self.event(
                "funding-2",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0004", "funding_time": "2026-07-21T19:00:00Z"},
                event_time=self.now - timedelta(hours=1),
            ),
            self.event(
                "perpetual-book",
                MarketEventType.BOOK_SNAPSHOT,
                perpetual,
                {"bid_price": 100.9, "ask_price": 101.1},
                event_time=self.now - timedelta(minutes=1),
            ),
            self.event(
                "spot-book",
                MarketEventType.BOOK_SNAPSHOT,
                spot,
                {"bid_price": 99.9, "ask_price": 100.1},
                event_time=self.now - timedelta(minutes=1),
            ),
        ):
            self.store.append_event(event)

        first = self.runner.run(as_of=self.now)
        self.assertEqual(first.generation.appended, 1)
        self.assertEqual(first.scoring.appended, 0)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(forecast.values["current_funding_time"], "2026-07-21T19:00:00Z")

        repeated = self.runner.run(as_of=self.now)
        self.assertEqual(repeated.generation.appended, 0)
        self.assertEqual(repeated.generation.existing, 1)

        next_time = self.now + timedelta(hours=1)
        self.store.append_event(
            self.event(
                "funding-3",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0001", "funding_time": next_time.isoformat()},
                event_time=next_time,
            )
        )
        scored = self.runner.score_available(as_of=next_time)
        self.assertEqual(scored.appended, 1)
        self.assertEqual(self.runner.score_available(as_of=next_time).unscored, 0)
        counts = self.audit.counts()
        self.assertEqual(counts[AuditRecordType.FORECAST], 1)
        self.assertEqual(counts[AuditRecordType.FORECAST_SCORE], 1)
        self.assertEqual(counts[AuditRecordType.ORDER_INTENT], 0)

    def test_prediction_forecast_waits_for_future_public_settlement(self):
        market = Instrument(
            "kalshi:prediction:TARGET",
            "kalshi",
            "TARGET",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        self.store.append_event(
            self.event(
                "target-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {"rules_primary": "Named public source."},
                event_time=self.now - timedelta(minutes=2),
            )
        )
        self.store.append_event(
            self.event(
                "target-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now - timedelta(minutes=1),
            )
        )
        generated = self.runner.run(as_of=self.now)
        self.assertEqual(generated.generation.appended, 1)
        self.assertEqual(self.runner.score_available(as_of=self.now).matched, 0)

        late_label_time = self.now + timedelta(minutes=1)
        self.store.append_event(
            self.event(
                "late-label-for-prior-outcome",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes"},
                event_time=self.now - timedelta(minutes=2),
                available_at=late_label_time,
            )
        )
        self.assertEqual(
            self.runner.score_available(as_of=late_label_time).matched,
            0,
        )

        settlement_time = self.now + timedelta(days=2)
        self.store.append_event(
            self.event(
                "target-settlement",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes"},
                event_time=settlement_time,
                available_at=settlement_time + timedelta(minutes=5),
            )
        )
        scored = self.runner.score_available(
            as_of=settlement_time + timedelta(minutes=5)
        )
        self.assertEqual(scored.appended, 1)

    def test_option_forecast_scores_only_at_the_full_horizon(self):
        option = Instrument(
            "alpaca:option:AAPL260918C00200000",
            "alpaca",
            "AAPL260918C00200000",
            AssetClass.OPTION,
            "USD",
            100,
            metadata={"underlying_symbol": "AAPL"},
        )
        self.store.register_instrument(option)
        for index, implied in enumerate((0.20, 0.22, 0.24), start=3):
            quote_time = self.now - timedelta(minutes=index)
            self.store.append_event(
                self.event(
                    f"quote-{index}",
                    MarketEventType.QUOTE,
                    option,
                    {
                        "bid_price": 4.9,
                        "ask_price": 5.1,
                        "implied_volatility": implied,
                        "feed": "indicative",
                    },
                    event_time=quote_time,
                )
            )
        generated = self.runner.run(as_of=self.now)
        self.assertEqual(generated.generation.appended, 1)
        forecast = self.audit.forecasts()[0]

        early_time = forecast.generated_at + timedelta(hours=2)
        self.store.append_event(
            self.event(
                "early-quote",
                MarketEventType.QUOTE,
                option,
                {
                    "bid_price": 4.9,
                    "ask_price": 5.1,
                    "implied_volatility": 0.25,
                    "feed": "indicative",
                },
                event_time=early_time,
            )
        )
        self.assertEqual(self.runner.score_available(as_of=early_time).matched, 0)

        target_time = forecast.valid_until
        self.store.append_event(
            self.event(
                "horizon-quote",
                MarketEventType.QUOTE,
                option,
                {
                    "bid_price": 4.9,
                    "ask_price": 5.1,
                    "implied_volatility": 0.21,
                    "feed": "indicative",
                },
                event_time=target_time,
            )
        )
        scored = self.runner.score_available(as_of=target_time)
        self.assertEqual(scored.appended, 1)

    def test_breakout_forecast_scores_only_on_future_completed_bar(self):
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(instrument)
        for index in range(20):
            bar_time = self.now - timedelta(hours=20 - index)
            self.store.append_event(
                self.event(
                    f"range-{index}",
                    MarketEventType.BAR,
                    instrument,
                    {
                        "open": 100,
                        "high": 101,
                        "low": 99,
                        "close": 100,
                        "volume": 100,
                        "granularity_seconds": 3600,
                    },
                    event_time=bar_time,
                )
            )
        self.store.append_event(
            self.event(
                "breakout",
                MarketEventType.BAR,
                instrument,
                {
                    "open": 100,
                    "high": 103,
                    "low": 100,
                    "close": 102,
                    "volume": 150,
                    "granularity_seconds": 3600,
                },
                event_time=self.now,
            )
        )
        generated = self.runner.run(as_of=self.now)
        self.assertEqual(generated.generation.appended, 1)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(forecast.values["state"], "confirmed_up_range_breakout")
        self.assertEqual(self.runner.score_available(as_of=self.now).matched, 0)

        target_time = self.now + timedelta(hours=1)
        self.store.append_event(
            self.event(
                "future-bar",
                MarketEventType.BAR,
                instrument,
                {
                    "open": 102,
                    "high": 104,
                    "low": 101,
                    "close": 103,
                    "volume": 120,
                    "granularity_seconds": 3600,
                },
                event_time=target_time,
            )
        )
        scored = self.runner.score_available(as_of=target_time)
        self.assertEqual(scored.appended, 1)


if __name__ == "__main__":
    unittest.main()
