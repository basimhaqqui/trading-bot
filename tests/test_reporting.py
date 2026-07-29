import unittest
from datetime import datetime, timedelta, timezone

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.evaluation.reporting import (
    CohortDimension,
    EdgeStatus,
    EvaluationGateConfig,
    build_walk_forward_report,
)
from trading_bot.evaluation.scoring import (
    score_binary_forecast,
    score_return_forecast,
    score_volatility_forecast,
)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def observation(
        self,
        index,
        *,
        predicted,
        benchmark,
        actual,
        instrument_id=None,
        target_time=None,
        generated_at=None,
        forecast_horizon=None,
        event_ticker=None,
    ):
        generated_at = generated_at or self.base + timedelta(days=index)
        target_time = target_time or generated_at + timedelta(hours=1)
        forecast_horizon = forecast_horizon or timedelta(minutes=30)
        forecast = Forecast(
            f"forecast-{index}-{generated_at.isoformat()}",
            "prediction-specialist",
            "v1",
            instrument_id or f"market-{index}",
            ForecastKind.BINARY_PROBABILITY,
            generated_at,
            generated_at + forecast_horizon,
            {
                "probability": predicted,
                "market_probability": benchmark,
                **({"event_ticker": event_ticker} if event_ticker else {}),
            },
            0.5,
            {"sample_size": 1.0},
            (f"event-{index}",),
            ("fails held-out validation",),
        )
        score = score_binary_forecast(
            forecast,
            outcome=bool(actual),
            target_time=target_time,
            scored_at=target_time,
        )
        return forecast, score

    def test_strong_paired_results_clear_controls_and_become_candidate(self):
        observations = [
            self.observation(
                index,
                predicted=float(index % 2),
                benchmark=0.5,
                actual=index % 2,
            )
            for index in range(40)
        ]
        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )
        group = report.groups[0]
        self.assertEqual(group.status, EdgeStatus.CANDIDATE)
        self.assertEqual(group.independent_outcomes, 40)
        self.assertGreater(group.lower_confidence_bound, 0)
        self.assertGreater(group.mean_improvement, group.delayed_control_improvement)
        self.assertGreater(group.mean_improvement, group.shuffled_control_improvement)

    def test_benchmark_equal_model_is_rejected_after_minimum_sample(self):
        observations = [
            self.observation(
                index,
                predicted=0.5,
                benchmark=0.5,
                actual=index % 2,
            )
            for index in range(30)
        ]
        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )
        group = report.groups[0]
        self.assertEqual(group.status, EdgeStatus.REJECTED)
        self.assertEqual(group.mean_improvement, 0)
        self.assertIn("mean loss does not beat benchmark", group.reasons)

    def test_insufficient_sample_stays_collecting(self):
        forecast, score = self.observation(
            0, predicted=1.0, benchmark=0.5, actual=1
        )
        report = build_walk_forward_report((forecast,), (score,))
        self.assertEqual(report.groups[0].status, EdgeStatus.COLLECTING)
        self.assertIn("needs 29 more outcome clusters", report.groups[0].reasons)

    def test_latest_forecast_wins_within_one_outcome_cluster(self):
        target = self.base + timedelta(days=1)
        first = self.observation(
            1,
            predicted=0.1,
            benchmark=0.5,
            actual=1,
            instrument_id="same-market",
            target_time=target,
            generated_at=self.base,
        )
        latest = self.observation(
            2,
            predicted=0.9,
            benchmark=0.5,
            actual=1,
            instrument_id="same-market",
            target_time=target,
            generated_at=self.base + timedelta(hours=1),
        )
        report = build_walk_forward_report(
            (first[0], latest[0]),
            (first[1], latest[1]),
            EvaluationGateConfig(min_independent_outcomes=2),
        )
        group = report.groups[0]
        self.assertEqual(group.raw_scores, 2)
        self.assertEqual(group.independent_outcomes, 1)
        self.assertAlmostEqual(group.mean_loss, 0.01)

    def test_prediction_strikes_from_one_event_are_one_outcome_cluster(self):
        observations = [
            self.observation(
                index,
                predicted=1.0,
                benchmark=0.5,
                actual=1,
                instrument_id=f"same-event-strike-{index}",
                event_ticker="SAME-EVENT",
                target_time=self.base + timedelta(days=1, seconds=index),
                generated_at=self.base + timedelta(minutes=index),
            )
            for index in range(30)
        ]

        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )

        group = report.groups[0]
        self.assertEqual(group.raw_scores, 30)
        self.assertEqual(group.independent_outcomes, 1)
        self.assertEqual(group.status, EdgeStatus.COLLECTING)

    def test_same_hour_crypto_breakouts_are_one_outcome_cluster(self):
        target_time = self.base + timedelta(hours=1)
        observations = []
        for index, instrument in enumerate(("BTC-USD", "SOL-USD", "DOGE-USD")):
            forecast = Forecast(
                f"crypto-{index}",
                "crypto-range-breakout-continuation-baseline",
                "baseline-v1",
                f"coinbase:product:{instrument}",
                ForecastKind.RETURN_DISTRIBUTION,
                self.base,
                target_time,
                {"predicted_return": -0.01, "benchmark_return": 0.0},
                0.3,
                {"sample_size": 20.0},
                (f"crypto-event-{index}",),
                ("correlated assets invalidate independence",),
            )
            score = score_return_forecast(
                forecast,
                actual_return=-0.02,
                target_time=target_time,
                scored_at=target_time,
            )
            observations.append((forecast, score))

        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )

        group = report.groups[0]
        self.assertEqual(group.raw_scores, 3)
        self.assertEqual(group.independent_outcomes, 1)
        self.assertEqual(group.status, EdgeStatus.COLLECTING)

    def test_same_intraday_crypto_block_is_one_outcome_cluster(self):
        target_time = self.base + timedelta(minutes=15)
        observations = []
        for index, instrument in enumerate(("BTC-USD", "ETH-USD", "SOL-USD")):
            forecast = Forecast(
                f"intraday-crypto-{index}",
                "crypto-intraday-momentum-baseline",
                "baseline-v1",
                f"coinbase:product:{instrument}",
                ForecastKind.RETURN_DISTRIBUTION,
                self.base,
                target_time,
                {
                    "predicted_return": 0.002,
                    "benchmark_return": 0.0,
                    "outcome_cluster": f"crypto-intraday:{target_time.isoformat()}",
                },
                0.3,
                {"lookback_bars": 8.0},
                (f"intraday-event-{index}",),
                ("correlated assets invalidate independence",),
            )
            score = score_return_forecast(
                forecast,
                actual_return=0.003,
                target_time=target_time,
                scored_at=target_time,
            )
            observations.append((forecast, score))

        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )

        group = report.groups[0]
        self.assertEqual(group.raw_scores, 3)
        self.assertEqual(group.independent_outcomes, 1)
        self.assertEqual(group.status, EdgeStatus.COLLECTING)

    def test_crypto_candidate_requires_instrument_diversity(self):
        observations = []
        for index in range(40):
            generated_at = self.base + timedelta(days=index)
            target_time = generated_at + timedelta(hours=1)
            predicted = 0.01 if index % 2 else -0.01
            forecast = Forecast(
                f"concentrated-crypto-{index}",
                "crypto-range-breakout-continuation-baseline",
                "baseline-v1",
                "coinbase:product:BTC-USD",
                ForecastKind.RETURN_DISTRIBUTION,
                generated_at,
                target_time,
                {"predicted_return": predicted, "benchmark_return": 0.0},
                0.3,
                {"sample_size": 20.0},
                (f"concentrated-event-{index}",),
                ("one asset invalidates the result",),
            )
            score = score_return_forecast(
                forecast,
                actual_return=predicted,
                target_time=target_time,
                scored_at=target_time,
            )
            observations.append((forecast, score))

        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )

        group = report.groups[0]
        self.assertEqual(group.independent_outcomes, 40)
        self.assertEqual(group.unique_instruments, 1)
        self.assertEqual(group.largest_instrument_share, 1.0)
        self.assertEqual(group.status, EdgeStatus.COLLECTING)
        self.assertIn(
            "needs outcomes from at least 2 instruments",
            group.reasons,
        )
        self.assertIn(
            "largest instrument share 100.0% exceeds 80.0% gate",
            group.reasons,
        )

    def test_crypto_candidate_accepts_predeclared_concentration_boundary(self):
        observations = []
        for index in range(40):
            generated_at = self.base + timedelta(days=index)
            target_time = generated_at + timedelta(hours=1)
            predicted = 0.01 if index % 2 else -0.01
            instrument = "BTC-USD" if index < 32 else "ETH-USD"
            forecast = Forecast(
                f"diverse-crypto-{index}",
                "crypto-range-breakout-continuation-baseline",
                "baseline-v1",
                f"coinbase:product:{instrument}",
                ForecastKind.RETURN_DISTRIBUTION,
                generated_at,
                target_time,
                {"predicted_return": predicted, "benchmark_return": 0.0},
                0.3,
                {"sample_size": 20.0},
                (f"diverse-event-{index}",),
                ("one asset invalidates the result",),
            )
            score = score_return_forecast(
                forecast,
                actual_return=predicted,
                target_time=target_time,
                scored_at=target_time,
            )
            observations.append((forecast, score))

        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )

        group = report.groups[0]
        self.assertEqual(group.unique_instruments, 2)
        self.assertEqual(group.largest_instrument_share, 0.8)
        self.assertEqual(group.status, EdgeStatus.CANDIDATE)

    def test_same_session_option_contracts_are_one_outcome_cluster(self):
        target_time = self.base + timedelta(days=1)
        observations = []
        for index in range(30):
            forecast = Forecast(
                f"option-{index}",
                "options-implied-volatility-state-baseline",
                "baseline-v1",
                f"alpaca:option:SPY-{index}",
                ForecastKind.VOLATILITY,
                self.base,
                target_time,
                {
                    "current_implied_volatility": 0.2,
                    "expected_implied_volatility": 0.25,
                    "outcome_cluster": "option-session:2026-01-02",
                },
                0.25,
                {"observations": 3.0},
                (f"option-event-{index}",),
                ("same-session contracts invalidate independence",),
            )
            score = score_volatility_forecast(
                forecast,
                actual_implied_volatility=0.3,
                target_time=target_time,
                scored_at=target_time,
            )
            observations.append((forecast, score))

        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
        )

        group = report.groups[0]
        self.assertEqual(group.raw_scores, 30)
        self.assertEqual(group.independent_outcomes, 1)
        self.assertEqual(group.status, EdgeStatus.COLLECTING)

    def test_fixed_horizon_cohorts_are_included_in_familywise_tests(self):
        horizons = (timedelta(minutes=30), timedelta(hours=4), timedelta(days=10))
        observations = [
            self.observation(
                index,
                predicted=float(index % 2),
                benchmark=0.5,
                actual=index % 2,
                target_time=self.base + timedelta(days=index) + horizon,
            )
            for index, horizon in enumerate(horizons)
        ]
        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
            EvaluationGateConfig(min_independent_outcomes=2),
        )
        outcome_labels = {
            item.label
            for item in report.cohorts
            if item.dimension is CohortDimension.OUTCOME_HORIZON
        }
        self.assertEqual(outcome_labels, {"<=1h", "1h-8h", "7d-30d"})
        self.assertEqual(report.confidence_tests, 5)

    def test_timing_guarded_prediction_uses_occurrence_not_settlement_horizon(self):
        generated_at = self.base
        occurrence = generated_at + timedelta(hours=4)
        forecast = Forecast(
            "timing-guarded-prediction",
            "prediction-market-calibration-adjusted-v1",
            "adjusted-v1",
            "kalshi:prediction:TIMING",
            ForecastKind.BINARY_PROBABILITY,
            generated_at,
            occurrence,
            {
                "probability": 0.8,
                "market_probability": 0.5,
                "event_ticker": "TIMING-EVENT",
                "outcome_cluster": "TIMING-EVENT",
                "target_time": occurrence.isoformat(),
            },
            0.5,
            {"sample_size": 5.0},
            ("timing-book",),
            ("post-occurrence information invalidates the forecast",),
        )
        score = score_binary_forecast(
            forecast,
            outcome=True,
            target_time=generated_at + timedelta(days=2),
            scored_at=generated_at + timedelta(days=2, minutes=5),
        )

        report = build_walk_forward_report(
            (forecast,),
            (score,),
            EvaluationGateConfig(min_independent_outcomes=2),
        )

        outcome_cohorts = [
            item
            for item in report.cohorts
            if item.dimension is CohortDimension.OUTCOME_HORIZON
        ]
        self.assertEqual(
            [(item.label, item.evaluation.raw_scores) for item in outcome_cohorts],
            [("1h-8h", 1)],
        )

    def test_legacy_fast_v4_identity_collision_is_excluded_from_evidence(self):
        forecast = Forecast(
            "legacy-v4-collision",
            "prediction-market-fast-settlement-baseline-v4",
            "baseline-v4",
            "kalshi:prediction:LEGACY",
            ForecastKind.BINARY_PROBABILITY,
            self.base,
            self.base + timedelta(hours=1),
            {
                "probability": 0.8,
                "market_probability": 0.5,
                "event_ticker": "LEGACY-EVENT",
                "target_time": (self.base + timedelta(hours=1)).isoformat(),
            },
            0.5,
            {},
            ("legacy-book",),
            (),
        )
        score = score_binary_forecast(
            forecast,
            outcome=True,
            target_time=self.base + timedelta(hours=1),
            scored_at=self.base + timedelta(hours=1),
        )

        report = build_walk_forward_report((forecast,), (score,))

        self.assertEqual(report.groups, ())

    def test_one_underpowered_horizon_keeps_aggregate_collecting(self):
        observations = [
            self.observation(
                index,
                predicted=float(index % 2),
                benchmark=0.5,
                actual=index % 2,
            )
            for index in range(10)
        ]
        observations.append(
            self.observation(
                10,
                predicted=0.0,
                benchmark=0.5,
                actual=0,
                target_time=self.base + timedelta(days=20),
            )
        )
        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
            EvaluationGateConfig(min_independent_outcomes=10),
        )
        group = report.groups[0]
        self.assertEqual(group.status, EdgeStatus.COLLECTING)
        self.assertIn(
            "outcome_horizon=7d-30d needs 9 more outcome clusters", group.reasons
        )

    def test_weak_mature_horizon_rejects_strong_aggregate(self):
        observations = [
            self.observation(
                index,
                predicted=float(index % 2),
                benchmark=0.5,
                actual=index % 2,
            )
            for index in range(30)
        ]
        observations.extend(
            self.observation(
                index,
                predicted=0.5,
                benchmark=0.5,
                actual=index % 2,
                target_time=self.base + timedelta(days=index + 10),
            )
            for index in range(30, 40)
        )
        report = build_walk_forward_report(
            tuple(item[0] for item in observations),
            tuple(item[1] for item in observations),
            EvaluationGateConfig(min_independent_outcomes=10),
        )
        group = report.groups[0]
        self.assertGreater(group.mean_improvement, 0)
        self.assertEqual(group.status, EdgeStatus.REJECTED)
        self.assertIn("outcome_horizon=7d-30d fails its edge gate", group.reasons)


if __name__ == "__main__":
    unittest.main()
