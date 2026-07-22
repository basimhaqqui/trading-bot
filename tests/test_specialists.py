import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.breakout import CryptoRangeBreakoutSpecialist
from trading_bot.agents.option_volatility import OptionVolatilitySpecialist
from trading_bot.agents.perpetual import PerpetualFundingBasisSpecialist
from trading_bot.agents.prediction import PredictionMarketCalibrationSpecialist
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.store import PointInTimeStore
from trading_bot.replay import ReplayEngine


class SpecialistTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def event(
        self,
        event_id,
        event_type,
        instrument,
        payload,
        *,
        minutes_ago=1,
        available_at=None,
    ):
        event_time = self.now - timedelta(minutes=minutes_ago)
        return MarketEvent(
            event_id,
            event_type,
            instrument.venue,
            instrument.instrument_id,
            event_time,
            available_at or event_time,
            "fixture",
            payload,
            ingested_at=available_at or event_time,
        )

    def test_perpetual_specialist_uses_related_spot_without_lookahead(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = PointInTimeStore(Path(temp.name) / "replay.db")
        store.initialize()
        perpetual = Instrument(
            "coinbase:product:BTC-PERP-USDC",
            "coinbase",
            "BTC-PERP-USDC",
            AssetClass.PERPETUAL,
            "USDC",
            settlement="cash",
        )
        spot = Instrument(
            "coinbase:product:BTC-USDC",
            "coinbase",
            "BTC-USDC",
            AssetClass.CRYPTO,
            "USDC",
        )
        store.register_instrument(perpetual)
        store.register_instrument(spot)
        events = (
            self.event(
                "funding-1",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0004"},
                minutes_ago=120,
            ),
            self.event(
                "funding-2",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0005"},
                minutes_ago=60,
            ),
            self.event(
                "perp-book",
                MarketEventType.BOOK_SNAPSHOT,
                perpetual,
                {"bid_price": 100.95, "ask_price": 101.05},
            ),
            self.event(
                "spot-book",
                MarketEventType.BOOK_SNAPSHOT,
                spot,
                {"bid_price": 99.95, "ask_price": 100.05},
            ),
        )
        for event in events:
            store.append_event(event)
        future_funding = MarketEvent(
            "future-funding",
            MarketEventType.FUNDING,
            perpetual.venue,
            perpetual.instrument_id,
            self.now,
            self.now + timedelta(minutes=1),
            "fixture",
            {"funding_rate": "-0.01"},
            ingested_at=self.now + timedelta(minutes=1),
        )
        store.append_event(future_funding)
        result = ReplayEngine(store).run(
            PerpetualFundingBasisSpecialist(),
            instrument_id=perpetual.instrument_id,
            related_instrument_ids=(spot.instrument_id,),
            decision_times=(self.now,),
        )
        forecast = result.forecasts[0]
        self.assertGreater(forecast.values["perpetual_spot_basis_bps"], 0)
        self.assertEqual(forecast.values["predicted_funding_rate"], 0.00045)
        self.assertNotIn("future-funding", forecast.evidence_event_ids)
        self.assertIn("spot-book", forecast.evidence_event_ids)

    def test_perpetual_specialist_does_not_count_repeat_polls_as_new_funding_periods(self):
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
        events = (
            self.event(
                "funding-poll-1",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0004", "funding_time": "2026-07-21T21:00:00Z"},
                minutes_ago=30,
            ),
            self.event(
                "funding-poll-2",
                MarketEventType.FUNDING,
                perpetual,
                {"funding_rate": "0.0004", "funding_time": "2026-07-21T21:00:00Z"},
                minutes_ago=10,
            ),
            self.event(
                "perp-repeat-book",
                MarketEventType.BOOK_SNAPSHOT,
                perpetual,
                {"bid_price": 100.9, "ask_price": 101.1},
            ),
            self.event(
                "spot-repeat-book",
                MarketEventType.BOOK_SNAPSHOT,
                spot,
                {"bid_price": 99.9, "ask_price": 100.1},
            ),
        )
        forecast = PerpetualFundingBasisSpecialist().evaluate(
            ReplayContext(self.now, perpetual, events, (spot,))
        )
        self.assertIsNone(forecast)

    def test_options_specialist_caps_indicative_confidence(self):
        option = Instrument(
            "alpaca:option:AAPL260918C00200000",
            "alpaca",
            "AAPL260918C00200000",
            AssetClass.OPTION,
            "USD",
            100,
        )
        events = tuple(
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
                minutes_ago=4 - index,
            )
            for index, implied in enumerate((0.20, 0.21, 0.30), start=1)
        )
        forecast = OptionVolatilitySpecialist().evaluate(
            ReplayContext(self.now, option, events)
        )
        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(forecast.values["state"], "implied_volatility_elevated")
        self.assertLessEqual(forecast.confidence, 0.25)
        self.assertEqual(forecast.evidence_event_ids, ("quote-1", "quote-2", "quote-3"))

    def test_options_specialist_reports_point_in_time_underlying_realized_volatility(self):
        option = Instrument(
            "alpaca:option:AAPL260918C00200000",
            "alpaca",
            "AAPL260918C00200000",
            AssetClass.OPTION,
            "USD",
            100,
        )
        equity = Instrument(
            "alpaca:equity:AAPL", "alpaca", "AAPL", AssetClass.EQUITY, "USD"
        )
        events = [
            self.event(
                f"option-quote-{index}",
                MarketEventType.QUOTE,
                option,
                {
                    "bid_price": 4.9,
                    "ask_price": 5.1,
                    "implied_volatility": implied,
                    "feed": "opra",
                },
                minutes_ago=4 - index,
            )
            for index, implied in enumerate((0.20, 0.21, 0.22), start=1)
        ]
        for index, close in enumerate((100, 101, 100, 103, 102, 104, 105)):
            event_time = self.now - timedelta(days=7 - index)
            events.append(
                MarketEvent(
                    f"underlying-bar-{index}",
                    MarketEventType.BAR,
                    "alpaca",
                    equity.instrument_id,
                    event_time,
                    self.now - timedelta(minutes=10),
                    "fixture",
                    {"close": close, "feed": "iex", "adjustment": "raw"},
                    ingested_at=self.now - timedelta(minutes=10),
                )
            )
        forecast = OptionVolatilitySpecialist().evaluate(
            ReplayContext(self.now, option, tuple(events), (equity,))
        )
        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertGreater(forecast.values["underlying_realized_volatility"], 0)
        self.assertIn("underlying-bar-6", forecast.evidence_event_ids)

    def test_prediction_specialist_only_adjusts_with_resolved_related_cohort(self):
        primary = Instrument(
            "kalshi:prediction:TARGET",
            "kalshi",
            "TARGET",
            AssetClass.PREDICTION,
            "USD",
        )
        events = [
            self.event(
                "target-rules",
                MarketEventType.CONTRACT_RULE,
                primary,
                {"rules_primary": "Settlement from the named source."},
                minutes_ago=60,
            ),
            self.event(
                "target-book",
                MarketEventType.BOOK_SNAPSHOT,
                primary,
                {"yes_bids": [["0.55", "10"]], "no_bids": [["0.40", "10"]]},
            ),
        ]
        related = []
        for index, result in enumerate(("yes", "yes", "yes", "yes", "no")):
            instrument = Instrument(
                f"kalshi:prediction:HIST-{index}",
                "kalshi",
                f"HIST-{index}",
                AssetClass.PREDICTION,
                "USD",
            )
            related.append(instrument)
            book = self.event(
                f"hist-book-{index}",
                MarketEventType.BOOK_SNAPSHOT,
                instrument,
                {"yes_bids": [["0.55", "10"]], "no_bids": [["0.40", "10"]]},
                minutes_ago=120,
            )
            settlement = self.event(
                f"settlement-{index}",
                MarketEventType.SETTLEMENT,
                instrument,
                {"result": result},
                minutes_ago=60,
            )
            events.extend((book, settlement))
            if index == 0:
                events.append(
                    self.event(
                        "settlement-0-repeat-poll",
                        MarketEventType.SETTLEMENT,
                        instrument,
                        {"result": result},
                        minutes_ago=30,
                    )
                )
        forecast = PredictionMarketCalibrationSpecialist().evaluate(
            ReplayContext(self.now, primary, tuple(events), tuple(related))
        )
        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(forecast.values["market_probability"], 0.575)
        self.assertGreater(forecast.values["probability"], 0.575)
        self.assertEqual(forecast.values["calibration_cohort_size"], 5.0)
        self.assertEqual(forecast.values["state"], "cohort_adjusted")

    def test_crypto_breakout_requires_completed_range_break_and_volume(self):
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        events = []
        for index in range(20):
            event_time = self.now - timedelta(hours=20 - index)
            events.append(
                MarketEvent(
                    f"range-bar-{index}",
                    MarketEventType.BAR,
                    "coinbase",
                    instrument.instrument_id,
                    event_time,
                    event_time,
                    "fixture",
                    {
                        "open": 100,
                        "high": 101,
                        "low": 99,
                        "close": 100,
                        "volume": 100,
                        "granularity_seconds": 3600,
                    },
                    ingested_at=event_time,
                )
            )
        events.append(
            MarketEvent(
                "breakout-bar",
                MarketEventType.BAR,
                "coinbase",
                instrument.instrument_id,
                self.now,
                self.now,
                "fixture",
                {
                    "open": 100,
                    "high": 103,
                    "low": 100,
                    "close": 102,
                    "volume": 150,
                    "granularity_seconds": 3600,
                },
                ingested_at=self.now,
            )
        )
        forecast = CryptoRangeBreakoutSpecialist().evaluate(
            ReplayContext(self.now, instrument, tuple(events))
        )
        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(forecast.values["direction"], "up")
        self.assertGreater(forecast.values["predicted_return"], 0)
        self.assertEqual(len(forecast.evidence_event_ids), 21)
        low_volume_payload = {**events[-1].payload, "volume": 10}
        low_volume_event = MarketEvent(
            **{
                **events[-1].__dict__,
                "event_id": "low-volume-breakout",
                "payload": low_volume_payload,
            }
        )
        self.assertIsNone(
            CryptoRangeBreakoutSpecialist().evaluate(
                ReplayContext(self.now, instrument, (*events[:-1], low_volume_event))
            )
        )


if __name__ == "__main__":
    unittest.main()
