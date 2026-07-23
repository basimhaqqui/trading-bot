import unittest
from datetime import datetime, timedelta, timezone

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.evaluation.reporting import (
    CohortDimension,
    EdgeStatus,
    EvaluationGateConfig,
    build_walk_forward_report,
)
from trading_bot.evaluation.scoring import score_binary_forecast


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
                target_time=self.base + timedelta(days=1),
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
