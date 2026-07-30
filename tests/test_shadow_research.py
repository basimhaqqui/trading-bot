import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from trading_bot.agents.crypto_momentum import CryptoIntradayMomentumV2Specialist
from trading_bot.agents.hypotheses import CRYPTO_INTRADAY_V2_PROPOSED_AT
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
from trading_bot.cli import _print_shadow_research
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

    def test_fast_prediction_funnel_output_is_explicitly_non_evidence(self):
        result = self.runner.run(as_of=self.now)
        eligibility = self.runner.fast_prediction_eligibility(as_of=self.now)

        with patch("builtins.print") as output:
            _print_shadow_research(result, eligibility)

        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("fast prediction eligibility (pre-generation; not evidence)", rendered)
        self.assertIn("documented_close_policy=0", rendered)
        self.assertIn("selected=0", rendered)

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
            forecast.specialist_id, "prediction-market-calibration-adjusted-v1"
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

    def test_legacy_fast_v4_identity_collision_is_quarantined(self):
        legacy = Forecast(
            "legacy-v4-collision",
            "prediction-market-fast-settlement-baseline-v4",
            "baseline-v4",
            "kalshi:prediction:LEGACY",
            ForecastKind.BINARY_PROBABILITY,
            self.now,
            self.now + timedelta(hours=1),
            {
                "probability": 0.6,
                "market_probability": 0.5,
                "event_ticker": "LEGACY-EVENT",
                "target_time": (self.now + timedelta(hours=1)).isoformat(),
            },
            0.5,
            {},
            ("legacy-book",),
            (),
        )
        self.audit.append_forecast(legacy)

        summary = self.runner.score_available(as_of=self.now + timedelta(hours=2))

        self.assertEqual(summary.quarantined, 1)
        self.assertEqual(summary.appended, 0)

    def test_prediction_selection_never_uses_rule_after_book_availability(self):
        market = Instrument(
            "kalshi:prediction:LATE-RULE",
            "kalshi",
            "LATE-RULE",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        book_time = self.now - timedelta(minutes=5)
        for event in (
            self.event(
                "original-short-horizon-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "rules_primary": "Named public source.",
                    "event_ticker": "LATE-RULE-EVENT",
                    "occurrence_datetime": (self.now + timedelta(minutes=30)).isoformat(),
                },
                event_time=book_time - timedelta(minutes=1),
            ),
            self.event(
                "book-before-rule-update",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=book_time,
            ),
            self.event(
                "later-long-horizon-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "rules_primary": "Named public source.",
                    "event_ticker": "LATE-RULE-EVENT",
                    "occurrence_datetime": (self.now + timedelta(hours=2)).isoformat(),
                },
                event_time=self.now - timedelta(minutes=1),
            ),
        ):
            self.store.append_event(event)

        self.assertEqual(self.runner._prediction_candidates(self.now), [])

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
            "prediction-market-calibration-adjusted-v1",
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

    def test_fast_prediction_candidates_cluster_to_one_event(self):
        for index, spread in enumerate((("0.45", "0.53"), ("0.40", "0.48"))):
            market = Instrument(
                f"kalshi:prediction:FAST-{index}",
                "kalshi",
                f"FAST-{index}",
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(market)
            self.store.append_event(
                self.event(
                    f"fast-rule-{index}",
                    MarketEventType.CONTRACT_RULE,
                    market,
                    {
                        "event_ticker": "FAST-EVENT",
                        "status": "active",
                        "can_close_early": False,
                        "settlement_timer_seconds": 900,
                        "expected_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
                        "latest_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
                    },
                    event_time=self.now - timedelta(minutes=2),
                )
            )
            self.store.append_event(
                self.event(
                    f"fast-book-{index}",
                    MarketEventType.BOOK_SNAPSHOT,
                    market,
                    {"yes_bids": [[spread[0], "10"]], "no_bids": [[spread[1], "10"]]},
                    event_time=self.now - timedelta(minutes=1),
                )
            )
        candidates = self.runner._fast_prediction_candidates(self.now)
        self.assertEqual(len(candidates), 1)
        eligibility = self.runner.fast_prediction_eligibility(as_of=self.now)
        self.assertEqual(eligibility.documented_close_policy_markets, 2)
        self.assertEqual(eligibility.early_close_enabled_markets, 0)
        self.assertEqual(eligibility.early_close_disabled_markets, 2)
        self.assertEqual(eligibility.missing_close_policy_markets, 0)
        self.assertEqual(eligibility.invalid_close_policy_markets, 0)
        generated = self.runner.generate_forecasts(as_of=self.now)
        self.assertEqual(generated.appended, 1)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(
            forecast.specialist_id, "prediction-market-fast-settlement-baseline-v6"
        )
        self.assertEqual(forecast.values["outcome_cluster"], "FAST-EVENT")

    def test_fast_prediction_selection_never_uses_rule_after_book_availability(self):
        market = Instrument(
            "kalshi:prediction:FAST-LATE-RULE",
            "kalshi",
            "FAST-LATE-RULE",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        book_time = self.now - timedelta(minutes=5)
        expiration = self.now + timedelta(hours=1)
        for event in (
            self.event(
                "fast-original-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "event_ticker": "FAST-LATE-RULE-EVENT",
                    "status": "active",
                    "can_close_early": True,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": expiration.isoformat(),
                    "latest_expiration_time": expiration.isoformat(),
                },
                event_time=book_time - timedelta(minutes=1),
            ),
            self.event(
                "fast-book-before-rule-update",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=book_time,
            ),
            self.event(
                "fast-later-rule-update",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "event_ticker": "FAST-LATE-RULE-EVENT",
                    "status": "active",
                    "can_close_early": False,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": expiration.isoformat(),
                    "latest_expiration_time": expiration.isoformat(),
                },
                event_time=self.now - timedelta(minutes=1),
            ),
        ):
            self.store.append_event(event)

        candidates = self.runner._fast_prediction_candidates(self.now)
        eligibility = self.runner.fast_prediction_eligibility(as_of=self.now)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(eligibility.paired_markets, 1)
        self.assertEqual(eligibility.fresh_book_markets, 1)
        self.assertEqual(eligibility.active_markets, 1)
        self.assertEqual(eligibility.documented_close_policy_markets, 1)
        self.assertEqual(eligibility.early_close_enabled_markets, 1)
        self.assertEqual(eligibility.early_close_disabled_markets, 0)
        self.assertEqual(eligibility.selected_events, 1)
        generated = self.runner.generate_forecasts(as_of=self.now)
        self.assertEqual(generated.appended, 1)
        self.assertTrue(self.audit.forecasts()[0].values["can_close_early"])

    def test_fast_prediction_eligibility_attributes_close_constraint_rejections(self):
        close_constraints = (
            ("EARLY", {"can_close_early": True}),
            ("MISSING", {}),
            ("INVALID", {"can_close_early": "false"}),
        )
        for suffix, constraint in close_constraints:
            market = Instrument(
                f"kalshi:prediction:FAST-{suffix}",
                "kalshi",
                f"FAST-{suffix}",
                AssetClass.PREDICTION,
                "USD",
            )
            self.store.register_instrument(market)
            rule_payload = {
                "event_ticker": f"FAST-{suffix}-EVENT",
                "status": "active",
                "settlement_timer_seconds": 900,
                "expected_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
                "latest_expiration_time": (self.now + timedelta(hours=1)).isoformat(),
                **constraint,
            }
            self.store.append_event(
                self.event(
                    f"fast-{suffix.lower()}-rule",
                    MarketEventType.CONTRACT_RULE,
                    market,
                    rule_payload,
                    event_time=self.now - timedelta(minutes=2),
                )
            )
            self.store.append_event(
                self.event(
                    f"fast-{suffix.lower()}-book",
                    MarketEventType.BOOK_SNAPSHOT,
                    market,
                    {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                    event_time=self.now - timedelta(minutes=1),
                )
            )

        eligibility = self.runner.fast_prediction_eligibility(as_of=self.now)

        self.assertEqual(eligibility.active_markets, 3)
        self.assertEqual(eligibility.documented_close_policy_markets, 1)
        self.assertEqual(eligibility.early_close_enabled_markets, 1)
        self.assertEqual(eligibility.early_close_disabled_markets, 0)
        self.assertEqual(eligibility.missing_close_policy_markets, 1)
        self.assertEqual(eligibility.invalid_close_policy_markets, 1)
        self.assertEqual(eligibility.selected_events, 1)

    def test_fast_prediction_rejects_settlement_after_recorded_deadline(self):
        market = Instrument(
            "kalshi:prediction:FAST-LATE",
            "kalshi",
            "FAST-LATE",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        expiration = self.now + timedelta(hours=1)
        self.store.append_event(
            self.event(
                "fast-late-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "event_ticker": "FAST-LATE-EVENT",
                    "status": "active",
                    "can_close_early": False,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": expiration.isoformat(),
                    "latest_expiration_time": expiration.isoformat(),
                },
                event_time=self.now - timedelta(minutes=1),
            )
        )
        self.store.append_event(
            self.event(
                "fast-late-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now,
            )
        )
        self.assertEqual(self.runner.generate_forecasts(as_of=self.now).appended, 1)
        deadline = expiration + timedelta(hours=1, minutes=15)
        self.store.append_event(
            self.event(
                "fast-late-settlement",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes", "event_ticker": "FAST-LATE-EVENT"},
                event_time=deadline + timedelta(seconds=1),
            )
        )

        scored = self.runner.score_available(as_of=deadline + timedelta(seconds=1))

        self.assertEqual(scored.appended, 0)
        self.assertEqual(scored.due_unmatched, 1)

    def test_fast_prediction_v6_accepts_early_close_within_recorded_deadline(self):
        market = Instrument(
            "kalshi:prediction:FAST-EARLY",
            "kalshi",
            "FAST-EARLY",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        expiration = self.now + timedelta(hours=1)
        self.store.append_event(
            self.event(
                "fast-early-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "event_ticker": "FAST-EARLY-EVENT",
                    "status": "active",
                    "can_close_early": True,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": expiration.isoformat(),
                    "latest_expiration_time": expiration.isoformat(),
                },
                event_time=self.now - timedelta(minutes=1),
            )
        )
        self.store.append_event(
            self.event(
                "fast-early-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now,
            )
        )
        self.assertEqual(self.runner.generate_forecasts(as_of=self.now).appended, 1)
        self.store.append_event(
            self.event(
                "fast-early-settlement",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes", "event_ticker": "FAST-EARLY-EVENT"},
                event_time=expiration - timedelta(seconds=1),
            )
        )

        scored = self.runner.score_available(as_of=expiration - timedelta(seconds=1))

        self.assertEqual(scored.appended, 1)
        self.assertEqual(scored.due_unmatched, 0)

    def test_fast_prediction_v6_rejects_settlement_from_a_different_event(self):
        market = Instrument(
            "kalshi:prediction:FAST-MISMATCH",
            "kalshi",
            "FAST-MISMATCH",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        expiration = self.now + timedelta(hours=1)
        self.store.append_event(
            self.event(
                "fast-mismatch-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "event_ticker": "FAST-EXPECTED-EVENT",
                    "status": "active",
                    "can_close_early": False,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": expiration.isoformat(),
                    "latest_expiration_time": expiration.isoformat(),
                },
                event_time=self.now - timedelta(minutes=1),
            )
        )
        self.store.append_event(
            self.event(
                "fast-mismatch-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now,
            )
        )
        self.assertEqual(self.runner.generate_forecasts(as_of=self.now).appended, 1)
        self.store.append_event(
            self.event(
                "fast-mismatch-settlement",
                MarketEventType.SETTLEMENT,
                market,
                {"result": "yes", "event_ticker": "FAST-OTHER-EVENT"},
                event_time=expiration + timedelta(minutes=1),
            )
        )

        scored = self.runner.score_available(as_of=expiration + timedelta(minutes=1))

        self.assertEqual(scored.appended, 0)
        self.assertEqual(scored.due_unmatched, 1)

    def test_fast_prediction_v6_records_long_latest_expiration_without_excluding_fast_target(self):
        market = Instrument(
            "kalshi:prediction:FAST-LONG-LATEST",
            "kalshi",
            "FAST-LONG-LATEST",
            AssetClass.PREDICTION,
            "USD",
        )
        self.store.register_instrument(market)
        expected = self.now + timedelta(hours=1)
        latest = self.now + timedelta(days=3)
        self.store.append_event(
            self.event(
                "fast-long-latest-rule",
                MarketEventType.CONTRACT_RULE,
                market,
                {
                    "event_ticker": "FAST-LONG-LATEST-EVENT",
                    "status": "active",
                    "can_close_early": True,
                    "settlement_timer_seconds": 900,
                    "expected_expiration_time": expected.isoformat(),
                    "latest_expiration_time": latest.isoformat(),
                },
                event_time=self.now - timedelta(minutes=1),
            )
        )
        self.store.append_event(
            self.event(
                "fast-long-latest-book",
                MarketEventType.BOOK_SNAPSHOT,
                market,
                {"yes_bids": [["0.45", "10"]], "no_bids": [["0.53", "10"]]},
                event_time=self.now,
            )
        )

        generated = self.runner.generate_forecasts(as_of=self.now)

        self.assertEqual(generated.appended, 1)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(forecast.valid_until, expected)
        self.assertEqual(forecast.values["latest_expiration_time"], latest.isoformat())
        self.assertEqual(
            forecast.values["settlement_deadline"],
            (expected + timedelta(hours=1, minutes=15)).isoformat(),
        )

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

    def test_intraday_momentum_forecast_scores_on_next_fifteen_minute_bar(self):
        instrument = Instrument(
            "coinbase:product:ETH-USD",
            "coinbase",
            "ETH-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(instrument)
        for index in range(8):
            event_time = self.now - timedelta(minutes=(7 - index) * 15)
            close = 100 + index * 0.2
            self.store.append_event(
                self.event(
                    f"intraday-{index}",
                    MarketEventType.BAR,
                    instrument,
                    {
                        "open": close - 0.1,
                        "high": close + 0.1,
                        "low": close - 0.2,
                        "close": close,
                        "volume": 100 + index,
                        "granularity_seconds": 900,
                    },
                    event_time=event_time,
                )
            )

        generated = self.runner.run(as_of=self.now)

        self.assertEqual(generated.generation.appended, 1)
        forecast = self.audit.forecasts()[0]
        self.assertEqual(
            forecast.specialist_id, "crypto-intraday-momentum-baseline"
        )
        self.assertEqual(
            forecast.valid_until, self.now + timedelta(minutes=15)
        )
        target_time = self.now + timedelta(minutes=15)
        self.store.append_event(
            self.event(
                "intraday-outcome",
                MarketEventType.BAR,
                instrument,
                {
                    "open": 101.4,
                    "high": 102,
                    "low": 101,
                    "close": 101.8,
                    "volume": 120,
                    "granularity_seconds": 900,
                },
                event_time=target_time,
            )
        )

        scored = self.runner.score_available(as_of=target_time)

        self.assertEqual(scored.appended, 1)

    def test_intraday_v2_only_uses_its_preassigned_symbol_after_registration(self):
        now = CRYPTO_INTRADAY_V2_PROPOSED_AT + timedelta(hours=2)
        target_time = now + timedelta(minutes=15)
        selected_symbol = CryptoIntradayMomentumV2Specialist.selected_symbol(target_time)
        selected = Instrument(
            f"coinbase:product:{selected_symbol}",
            "coinbase",
            selected_symbol,
            AssetClass.CRYPTO,
            "USD",
        )
        unselected_symbol = next(
            symbol
            for symbol in CryptoIntradayMomentumV2Specialist.assignment_universe
            if symbol != selected_symbol
        )
        unselected = Instrument(
            f"coinbase:product:{unselected_symbol}",
            "coinbase",
            unselected_symbol,
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(selected)
        self.store.register_instrument(unselected)
        for instrument in (selected, unselected):
            for index in range(8):
                event_time = now - timedelta(minutes=(7 - index) * 15)
                close = 100 + index
                self.store.append_event(
                    self.event(
                        f"v2-{instrument.symbol}-{index}",
                        MarketEventType.BAR,
                        instrument,
                        {
                            "open": close - 0.2,
                            "high": close + 0.2,
                            "low": close - 0.4,
                            "close": close,
                            "volume": 100 + index,
                            "granularity_seconds": 900,
                        },
                        event_time=event_time,
                    )
                )

        candidates = self.runner._intraday_momentum_v2_candidates(now)

        self.assertEqual(
            [candidate.instrument_id for candidate in candidates], [selected.instrument_id]
        )
        self.runner.run(as_of=now)
        v2 = [
            forecast
            for forecast in self.audit.forecasts()
            if forecast.specialist_id == "crypto-intraday-momentum-baseline-v2"
        ]
        self.assertEqual(len(v2), 1)
        self.assertEqual(v2[0].instrument_id, selected.instrument_id)

    def test_intraday_v2_does_not_substitute_an_unassigned_symbol(self):
        now = CRYPTO_INTRADAY_V2_PROPOSED_AT + timedelta(hours=3)
        target_time = now + timedelta(minutes=15)
        selected_symbol = CryptoIntradayMomentumV2Specialist.selected_symbol(target_time)
        unselected_symbol = next(
            symbol
            for symbol in CryptoIntradayMomentumV2Specialist.assignment_universe
            if symbol != selected_symbol
        )
        instrument = Instrument(
            f"coinbase:product:{unselected_symbol}",
            "coinbase",
            unselected_symbol,
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(instrument)
        for index in range(8):
            event_time = now - timedelta(minutes=(7 - index) * 15)
            close = 100 + index
            self.store.append_event(
                self.event(
                    f"v2-unselected-{index}",
                    MarketEventType.BAR,
                    instrument,
                    {
                        "open": close - 0.2,
                        "high": close + 0.2,
                        "low": close - 0.4,
                        "close": close,
                        "volume": 100 + index,
                        "granularity_seconds": 900,
                    },
                    event_time=event_time,
                )
            )

        self.assertEqual(self.runner._intraday_momentum_v2_candidates(now), [])


if __name__ == "__main__":
    unittest.main()
