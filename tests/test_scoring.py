import unittest
from datetime import datetime, timedelta, timezone

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.evaluation.scoring import (
    ScoreKind,
    score_binary_forecast,
    score_funding_forecast,
    score_return_forecast,
)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def forecast(self, kind, values):
        return Forecast(
            "forecast-1",
            "specialist",
            "v1",
            "instrument",
            kind,
            self.now,
            self.now + timedelta(hours=1),
            values,
            0.5,
            {"unknown": 1.0},
            ("event-1",),
            ("fails held-out validation",),
        )

    def test_binary_score_compares_adjustment_to_market_prior(self):
        score = score_binary_forecast(
            self.forecast(
                ForecastKind.BINARY_PROBABILITY,
                {"probability": 0.7, "market_probability": 0.6},
            ),
            outcome=True,
            target_time=self.now + timedelta(hours=1),
            scored_at=self.now + timedelta(hours=1),
        )
        self.assertLess(score.loss, score.benchmark_loss)
        self.assertGreater(score.metrics["brier_improvement_vs_market"], 0)

    def test_funding_score_compares_to_latest_rate_benchmark(self):
        score = score_funding_forecast(
            self.forecast(
                ForecastKind.FUNDING_RATE,
                {"predicted_funding_rate": 0.0004, "current_funding_rate": 0.0008},
            ),
            actual_rate=0.0005,
            target_time=self.now + timedelta(hours=1),
            scored_at=self.now + timedelta(hours=1),
        )
        self.assertLess(score.loss, score.benchmark_loss)
        self.assertGreater(
            score.metrics["squared_error_improvement_vs_latest"], 0
        )

    def test_return_score_compares_to_zero_return_benchmark(self):
        score = score_return_forecast(
            self.forecast(
                ForecastKind.RETURN_DISTRIBUTION,
                {"predicted_return": 0.01, "benchmark_return": 0.0},
            ),
            actual_return=0.012,
            target_time=self.now + timedelta(hours=1),
            scored_at=self.now + timedelta(hours=1),
        )
        self.assertEqual(score.kind, ScoreKind.RETURN)
        self.assertLess(score.loss, score.benchmark_loss)


if __name__ == "__main__":
    unittest.main()
