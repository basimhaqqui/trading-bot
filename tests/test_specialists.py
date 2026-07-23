import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.breakout import CryptoRangeBreakoutSpecialist
from trading_bot.agents.option_volatility import OptionVolatilitySpecialist
from trading_bot.agents.perpetual import PerpetualFundingBasisSpecialist
from trading_bot.agents.prediction import (
    AdjustedPredictionMarketCalibrationSpecialist,
    PredictionMarketCalibrationSpecialist,
)
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
                {
                    "rules_primary": "Settlement from the named source.",
                    "event_ticker": "TARGET-EVENT",
                    "occurrence_datetime": "2026-07-21T22:00:00Z",
                },
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
        historical_occurrence = (self.now - timedelta(minutes=90)).isoformat()
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
                minutes_ago=180,
            )
            settlement = self.event(
                f"settlement-{index}",
                MarketEventType.SETTLEMENT,
                instrument,
                {
                    "result": result,
                    "event_ticker": f"HIST-EVENT-{index}",
                    "occurrence_datetime": historical_occurrence,
                },
                minutes_ago=60,
            )
            events.extend((book, settlement))
            if index == 0:
                events.extend(
                    (
                        self.event(
                            "hist-book-0-post-occurrence",
                            MarketEventType.BOOK_SNAPSHOT,
                            instrument,
                            {
                                "yes_bids": [["0.98", "10"]],
                                "no_bids": [["0.01", "10"]],
                            },
                            minutes_ago=80,
                        ),
                        self.event(
                            "settlement-0-repeat-poll",
                            MarketEventType.SETTLEMENT,
                            instrument,
                            {
                                "result": result,
                                "event_ticker": "HIST-EVENT-0",
                                "occurrence_datetime": historical_occurrence,
                            },
                            minutes_ago=30,
                        ),
                    )
                )
            if index == 1:
                events.append(
                    self.event(
                        "hist-book-1-wide-pre-occurrence",
                        MarketEventType.BOOK_SNAPSHOT,
                        instrument,
                        {
                            "yes_bids": [["0.10", "10"]],
                            "no_bids": [["0.10", "10"]],
                        },
                        minutes_ago=165,
                    )
                )
        duplicate_event_instrument = Instrument(
            "kalshi:prediction:HIST-OTHER-STRIKE",
            "kalshi",
            "HIST-OTHER-STRIKE",
            AssetClass.PREDICTION,
            "USD",
        )
        related.append(duplicate_event_instrument)
        events.extend(
            (
                self.event(
                    "hist-book-other-strike",
                    MarketEventType.BOOK_SNAPSHOT,
                    duplicate_event_instrument,
                    {"yes_bids": [["0.55", "10"]], "no_bids": [["0.40", "10"]]},
                    minutes_ago=180,
                ),
                self.event(
                    "settlement-other-strike",
                    MarketEventType.SETTLEMENT,
                    duplicate_event_instrument,
                    {
                        "result": "no",
                        "event_ticker": "HIST-EVENT-0",
                        "occurrence_datetime": historical_occurrence,
                    },
                    minutes_ago=60,
                ),
            )
        )
        independent_same_time = Instrument(
            "kalshi:prediction:HIST-INDEPENDENT-SAME-TIME",
            "kalshi",
            "HIST-INDEPENDENT-SAME-TIME",
            AssetClass.PREDICTION,
            "USD",
        )
        related.append(independent_same_time)
        events.extend(
            (
                self.event(
                    "hist-book-independent-same-time",
                    MarketEventType.BOOK_SNAPSHOT,
                    independent_same_time,
                    {"yes_bids": [["0.55", "10"]], "no_bids": [["0.40", "10"]]},
                    minutes_ago=180,
                ),
                self.event(
                    "settlement-independent-same-time",
                    MarketEventType.SETTLEMENT,
                    independent_same_time,
                    {
                        "result": "no",
                        "event_ticker": "HIST-INDEPENDENT-SAME-TIME",
                        "occurrence_datetime": historical_occurrence,
                    },
                    minutes_ago=60,
                ),
            )
        )
        forecast = PredictionMarketCalibrationSpecialist().evaluate(
            ReplayContext(self.now, primary, tuple(events), tuple(related))
        )
        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(forecast.values["market_probability"], 0.575)
        self.assertGreater(forecast.values["probability"], 0.575)
        self.assertEqual(forecast.values["calibration_cohort_size"], 6.0)
        self.assertEqual(forecast.values["state"], "cohort_adjusted")
        self.assertEqual(
            forecast.specialist_id, "prediction-market-calibration-baseline-v3"
        )
        self.assertEqual(forecast.model_version, "baseline-v3")
        self.assertEqual(forecast.values["event_ticker"], "TARGET-EVENT")
        self.assertEqual(forecast.values["outcome_cluster"], "TARGET-EVENT")
        self.assertEqual(forecast.values["target_time"], "2026-07-21T22:00:00+00:00")
        self.assertEqual(
            forecast.valid_until,
            datetime(2026, 7, 21, 22, tzinfo=timezone.utc),
        )
        adjusted_specialist = AdjustedPredictionMarketCalibrationSpecialist()
        self.assertEqual(adjusted_specialist.config.probability_bucket_radius, 0.10)
        self.assertEqual(adjusted_specialist.config.min_calibration_cohort, 5)
        self.assertEqual(adjusted_specialist.config.shrinkage_observations, 20.0)
        self.assertEqual(adjusted_specialist.config.max_book_spread, 0.10)
        self.assertEqual(
            adjusted_specialist.config.min_forecast_horizon, timedelta(hours=1)
        )
        self.assertEqual(adjusted_specialist.config.forecast_horizon, timedelta(hours=8))
        adjusted = adjusted_specialist.evaluate(
            ReplayContext(self.now, primary, tuple(events), tuple(related))
        )
        self.assertIsNotNone(adjusted)
        assert adjusted is not None
        self.assertEqual(
            adjusted.specialist_id, "prediction-market-calibration-baseline-v4"
        )
        self.assertEqual(adjusted.model_version, "baseline-v4")
        self.assertEqual(adjusted.values["state"], "cohort_adjusted")
        wide_book = self.event(
            "target-wide-book",
            MarketEventType.BOOK_SNAPSHOT,
            primary,
            {"yes_bids": [["0.10", "10"]], "no_bids": [["0.10", "10"]]},
            minutes_ago=0,
        )
        self.assertIsNone(
            PredictionMarketCalibrationSpecialist().evaluate(
                ReplayContext(
                    self.now,
                    primary,
                    tuple(event for event in events if event.event_id != "target-book")
                    + (wide_book,),
                    tuple(related),
                )
            )
        )

    def test_prediction_specialist_rejects_unsafe_occurrence_times(self):
        market = Instrument(
            "kalshi:prediction:TIMING",
            "kalshi",
            "TIMING",
            AssetClass.PREDICTION,
            "USD",
        )
        book = self.event(
            "timing-book",
            MarketEventType.BOOK_SNAPSHOT,
            market,
            {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
        )
        invalid_occurrences = (
            None,
            "not-a-time",
            self.now.isoformat(),
            (self.now - timedelta(minutes=1)).isoformat(),
            (self.now + timedelta(hours=1)).isoformat(),
            (self.now + timedelta(hours=9)).isoformat(),
            (self.now + timedelta(hours=49)).isoformat(),
        )
        specialist = PredictionMarketCalibrationSpecialist()
        for index, occurrence in enumerate(invalid_occurrences):
            payload = {"rules_primary": "Named public source."}
            if occurrence is not None:
                payload["occurrence_datetime"] = occurrence
            rule = self.event(
                f"timing-rule-{index}",
                MarketEventType.CONTRACT_RULE,
                market,
                payload,
                minutes_ago=2,
            )
            with self.subTest(occurrence=occurrence):
                self.assertIsNone(
                    specialist.evaluate(
                        ReplayContext(self.now, market, (rule, book), ())
                    )
                )

    def test_prediction_specialist_requires_stable_event_identity(self):
        market = Instrument(
            "kalshi:prediction:NO-EVENT",
            "kalshi",
            "NO-EVENT",
            AssetClass.PREDICTION,
            "USD",
        )
        rule = self.event(
            "missing-event-ticker",
            MarketEventType.CONTRACT_RULE,
            market,
            {
                "rules_primary": "Named public source.",
                "occurrence_datetime": (self.now + timedelta(hours=2)).isoformat(),
            },
            minutes_ago=2,
        )
        book = self.event(
            "missing-event-ticker-book",
            MarketEventType.BOOK_SNAPSHOT,
            market,
            {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
        )

        self.assertIsNone(
            PredictionMarketCalibrationSpecialist().evaluate(
                ReplayContext(self.now, market, (rule, book), ())
            )
        )
        identified_rule = self.event(
            "identified-event-ticker",
            MarketEventType.CONTRACT_RULE,
            market,
            {
                "rules_primary": "Named public source.",
                "event_ticker": "IDENTIFIED-EVENT",
                "occurrence_datetime": (self.now + timedelta(hours=2)).isoformat(),
            },
            minutes_ago=2,
        )
        prior = PredictionMarketCalibrationSpecialist().evaluate(
            ReplayContext(self.now, market, (identified_rule, book), ())
        )
        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertEqual(prior.values["state"], "executable_market_prior")
        self.assertIsNone(
            AdjustedPredictionMarketCalibrationSpecialist().evaluate(
                ReplayContext(self.now, market, (identified_rule, book), ())
            )
        )

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
