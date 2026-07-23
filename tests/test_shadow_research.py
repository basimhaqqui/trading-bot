import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from trading_bot.core.audit import AuditLedger, AuditRecordType
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    Instrument,
    MarketEvent,
    MarketEventType,
)
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.shadow import ShadowResearchConfig, ShadowResearchRunner


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

    def add_prediction_history(self, count=5):
        occurrence = self.now - timedelta(hours=2)
        for index in range(count):
            market = Instrument(
                f"kalshi:prediction:HISTORY-{index}",
                "kalshi",
                f"HISTORY-{index}",
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(market)
            self.store.append_event(
                self.event(
                    f"history-book-{index}",
                    MarketEventType.BOOK_SNAPSHOT,
                    market,
                    {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                    event_time=self.now - timedelta(hours=4),
                )
            )
            self.store.append_event(
                self.event(
                    f"history-settlement-{index}",
                    MarketEventType.SETTLEMENT,
                    market,
                    {
                        "result": "yes" if index < count - 1 else "no",
                        "event_ticker": f"HISTORY-EVENT-{index}",
                        "occurrence_datetime": occurrence.isoformat(),
                    },
                    event_time=self.now - timedelta(hours=1),
                )
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
        self.add_prediction_history()
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
                {
                    "rules_primary": "Named public source.",
                    "event_ticker": "TARGET-EVENT",
                    "occurrence_datetime": (
                        self.now + timedelta(hours=7)
                    ).isoformat(),
                },
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
        forecast = self.audit.forecasts()[0]
        self.assertEqual(
            forecast.specialist_id, "prediction-market-calibration-baseline-v4"
        )
        self.assertEqual(forecast.values["state"], "cohort_adjusted")
        self.assertEqual(forecast.values["outcome_cluster"], "TARGET-EVENT")
        self.assertEqual(forecast.values["target_time"], forecast.valid_until.isoformat())
        waiting = self.runner.score_available(as_of=self.now)
        self.assertEqual(waiting.matched, 0)
        self.assertEqual(waiting.not_due, 1)
        self.assertEqual(waiting.due_unmatched, 0)
        self.assertEqual(waiting.next_due_at, forecast.valid_until)
        self.assertIsNone(waiting.oldest_due_at)

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

        due = self.runner.score_available(as_of=forecast.valid_until)
        self.assertEqual(due.not_due, 0)
        self.assertEqual(due.due_unmatched, 1)
        self.assertIsNone(due.next_due_at)
        self.assertEqual(due.oldest_due_at, forecast.valid_until)

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
        self.assertEqual(
            self.audit.forecast_scores()[0].target_time,
            forecast.valid_until,
        )
        self.assertEqual(
            self.audit.forecast_scores()[0].scored_at,
            settlement_time + timedelta(minutes=5),
        )

    def test_prediction_history_cap_balances_independent_events(self):
        self.add_prediction_history()
        occurrence = self.now - timedelta(hours=2)
        for index in range(10):
            market = Instrument(
                f"kalshi:prediction:CROWDED-{index}",
                "kalshi",
                f"CROWDED-{index}",
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(market)
            self.store.append_event(
                self.event(
                    f"crowded-book-{index}",
                    MarketEventType.BOOK_SNAPSHOT,
                    market,
                    (
                        {
                            "yes_bids": [["0.45", "10"]],
                            "no_bids": [["0.53", "10"]],
                        }
                        if index == 0
                        else {
                            "yes_bids": [["0.15", "10"]],
                            "no_bids": [["0.83", "10"]],
                        }
                    ),
                    event_time=self.now - timedelta(hours=4),
                )
            )
            self.store.append_event(
                self.event(
                    f"crowded-settlement-{index}",
                    MarketEventType.SETTLEMENT,
                    market,
                    {
                        "result": "no",
                        "event_ticker": "CROWDED-EVENT",
                        "occurrence_datetime": occurrence.isoformat(),
                    },
                    event_time=self.now - timedelta(minutes=30),
                )
            )

        target = Instrument(
            "kalshi:prediction:BALANCED-TARGET",
            "kalshi",
            "BALANCED-TARGET",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(target)
        self.store.append_event(
            self.event(
                "balanced-target-rule",
                MarketEventType.CONTRACT_RULE,
                target,
                {
                    "rules_primary": "Named public source.",
                    "event_ticker": "BALANCED-TARGET-EVENT",
                    "occurrence_datetime": (
                        self.now + timedelta(hours=7)
                    ).isoformat(),
                },
                event_time=self.now - timedelta(minutes=2),
            )
        )
        self.store.append_event(
            self.event(
                "balanced-target-book",
                MarketEventType.BOOK_SNAPSHOT,
                target,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now - timedelta(minutes=1),
            )
        )
        runner = ShadowResearchRunner(
            self.store,
            self.audit,
            ShadowResearchConfig(max_prediction_history=5),
        )

        candidates = runner._prediction_candidates(self.now)
        self.assertEqual(len(candidates), 1)
        history_ids = candidates[0].related_instrument_ids
        self.assertEqual(len(history_ids), 5)
        self.assertEqual(
            sum(
                instrument_id.startswith("kalshi:prediction:CROWDED-")
                for instrument_id in history_ids
            ),
            1,
        )
        self.assertIn("kalshi:prediction:CROWDED-0", history_ids)

        generated = runner.generate_forecasts(as_of=self.now)
        self.assertEqual(generated.appended, 1)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(
            forecast.specialist_id,
            "prediction-market-calibration-baseline-v4",
        )
        self.assertEqual(forecast.values["calibration_cohort_size"], 5.0)

    def test_prediction_forecast_scores_early_public_settlement(self):
        self.add_prediction_history()
        market = Instrument(
            "kalshi:prediction:EARLY-TARGET",
            "kalshi",
            "EARLY-TARGET",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        occurrence = self.now + timedelta(hours=7)
        self.store.append_event(
            self.event(
                "early-target-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "rules_primary": "Named public source.",
                    "event_ticker": "EARLY-TARGET-EVENT",
                    "occurrence_datetime": occurrence.isoformat(),
                },
                event_time=self.now - timedelta(minutes=2),
            )
        )
        self.store.append_event(
            self.event(
                "early-target-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now - timedelta(minutes=1),
            )
        )
        generated = self.runner.run(as_of=self.now)
        self.assertEqual(generated.generation.appended, 1)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(forecast.valid_until, occurrence)

        settlement_time = self.now + timedelta(hours=2)
        self.store.append_event(
            self.event(
                "early-target-settlement",
                MarketEventType.SETTLEMENT,
                market,
                {
                    "result": "no",
                    "event_ticker": "EARLY-TARGET-EVENT",
                    "occurrence_datetime": occurrence.isoformat(),
                },
                event_time=settlement_time,
            )
        )
        scored = self.runner.score_available(as_of=settlement_time)

        self.assertEqual(scored.appended, 1)
        self.assertEqual(self.audit.forecast_scores()[0].target_time, occurrence)
        self.assertEqual(self.audit.forecast_scores()[0].scored_at, settlement_time)

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

    def test_option_candidate_discovery_uses_one_bulk_quote_read(self):
        for index in range(40):
            option = Instrument(
                f"alpaca:option:AAPL{index:08d}",
                "alpaca",
                f"AAPL{index:08d}",
                AssetClass.OPTION,
                "USD",
                100,
                metadata={"underlying_symbol": "AAPL"},
            )
            self.store.register_instrument(option)
            for quote_index in range(3):
                self.store.append_event(
                    self.event(
                        f"bulk-option-quote-{index}-{quote_index}",
                        MarketEventType.QUOTE,
                        option,
                        {
                            "bid_price": 4.9,
                            "ask_price": 5.1,
                            "implied_volatility": 0.2 + quote_index * 0.01,
                            "feed": "indicative",
                        },
                        event_time=self.now
                        - timedelta(minutes=3 - quote_index),
                    )
                )

        with patch.object(
            self.store,
            "events_available_at",
            wraps=self.store.events_available_at,
        ) as events_available_at:
            candidates = self.runner._option_candidates(self.now)

        self.assertEqual(len(candidates), 40)
        self.assertEqual(events_available_at.call_count, 1)

    def test_option_candidate_waits_for_active_forecast_horizon(self):
        option = Instrument(
            "alpaca:option:SPY260918C00600000",
            "alpaca",
            "SPY260918C00600000",
            AssetClass.OPTION,
            "USD",
            100,
            metadata={"underlying_symbol": "SPY"},
        )
        self.store.register_instrument(option)
        quote_ids = []
        for index in range(3):
            event_id = f"active-option-quote-{index}"
            quote_ids.append(event_id)
            self.store.append_event(
                self.event(
                    event_id,
                    MarketEventType.QUOTE,
                    option,
                    {
                        "bid_price": 4.9,
                        "ask_price": 5.1,
                        "implied_volatility": 0.2 + index * 0.01,
                        "feed": "indicative",
                    },
                    event_time=self.now - timedelta(minutes=3 - index),
                )
            )

        self.assertEqual(len(self.runner._option_candidates(self.now)), 1)
        self.audit.append_forecast(
            Forecast(
                "active-option-forecast",
                "options-implied-volatility-state-baseline",
                "baseline-v1",
                option.instrument_id,
                ForecastKind.VOLATILITY,
                self.now - timedelta(minutes=1),
                self.now + timedelta(days=1),
                {
                    "current_implied_volatility": 0.22,
                    "expected_implied_volatility": 0.21,
                },
                0.25,
                {"observations": 3.0},
                tuple(quote_ids),
                ("test fixture",),
            )
        )

        self.assertEqual(self.runner._option_candidates(self.now), [])

    def test_option_candidate_rejects_stale_source_with_fresh_receipt(self):
        option = Instrument(
            "alpaca:option:SPY260918C00600000",
            "alpaca",
            "SPY260918C00600000",
            AssetClass.OPTION,
            "USD",
            100,
            metadata={"underlying_symbol": "SPY"},
        )
        self.store.register_instrument(option)
        for index in range(3):
            self.store.append_event(
                self.event(
                    f"stale-source-option-{index}",
                    MarketEventType.QUOTE,
                    option,
                    {
                        "bid_price": 4.9,
                        "ask_price": 5.1,
                        "implied_volatility": 0.2 + index * 0.01,
                        "feed": "indicative",
                    },
                    event_time=self.now - timedelta(hours=5, minutes=index),
                    available_at=self.now - timedelta(minutes=1),
                )
            )

        self.assertEqual(self.runner._option_candidates(self.now), [])

    def test_prediction_candidate_discovery_uses_three_bulk_event_reads(self):
        self.add_prediction_history()
        for index in range(40):
            market = Instrument(
                f"kalshi:prediction:BULK-{index}",
                "kalshi",
                f"BULK-{index}",
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(market)
            self.store.append_event(
                self.event(
                    f"bulk-rule-{index}",
                    MarketEventType.CONTRACT_RULE,
                    market,
                    {
                        "rules_primary": "Named public source.",
                        "event_ticker": f"BULK-EVENT-{index}",
                        "occurrence_datetime": (
                            self.now + timedelta(hours=2, minutes=index)
                        ).isoformat(),
                    },
                    event_time=self.now - timedelta(minutes=2),
                )
            )
            self.store.append_event(
                self.event(
                    f"bulk-book-{index}",
                    MarketEventType.BOOK_SNAPSHOT,
                    market,
                    {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                    event_time=self.now - timedelta(minutes=1),
                )
            )

        with patch.object(
            self.store,
            "events_available_at",
            wraps=self.store.events_available_at,
        ) as events_available_at:
            candidates = self.runner._prediction_candidates(self.now)

        self.assertEqual(len(candidates), 25)
        self.assertEqual(events_available_at.call_count, 3)

        first = self.runner.generate_forecasts(as_of=self.now)
        second = self.runner.generate_forecasts(as_of=self.now)
        self.assertEqual(first.appended, 25)
        self.assertEqual(second.appended, 15)
        self.assertEqual(len(self.audit.forecasts()), 40)

    def test_prediction_timing_guard_rejects_post_occurrence_evidence(self):
        market = Instrument(
            "kalshi:prediction:PAST",
            "kalshi",
            "PAST",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        occurrence = self.now - timedelta(hours=1)
        rule = self.event(
            "past-rule",
            MarketEventType.CONTRACT_RULE,
            market,
            {
                "rules_primary": "Named public source.",
                "occurrence_datetime": occurrence.isoformat(),
            },
            event_time=self.now - timedelta(minutes=2),
        )
        self.store.append_event(rule)
        self.store.append_event(
            self.event(
                "past-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.80", "10"]], "no_bids": [["0.18", "10"]]},
                event_time=self.now - timedelta(minutes=1),
            )
        )
        self.assertEqual(self.runner._prediction_candidates(self.now), [])

        leaked = Forecast(
            "leaked-v2",
            "prediction-market-calibration-baseline-v2",
            "baseline-v2",
            market.instrument_id,
            ForecastKind.BINARY_PROBABILITY,
            self.now,
            self.now + timedelta(hours=1),
            {
                "probability": 0.8,
                "market_probability": 0.8,
                "outcome_cluster": occurrence.isoformat(),
            },
            0.5,
            {"market_spread": 0.02},
            (rule.event_id,),
            ("post-occurrence information invalidates the forecast",),
        )
        self.audit.append_forecast(leaked)
        settlement_time = self.now + timedelta(minutes=5)
        self.store.append_event(
            self.event(
                "past-settlement",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes"},
                event_time=settlement_time,
            )
        )
        self.assertEqual(
            self.runner.score_available(as_of=settlement_time).quarantined,
            1,
        )

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
