import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.core.store import PointInTimeStore
from trading_bot.evaluation.checkpoint import (
    checkpointed_walk_forward_report,
    locked_walk_forward_report,
)
from trading_bot.evaluation.reporting import (
    DECISION_SCOPE_AGGREGATE,
    EdgeStatus,
    EvaluationGateConfig,
    build_walk_forward_report,
)
from trading_bot.evaluation.scorecard import build_daily_scorecard, render_scorecard
from trading_bot.evaluation.scoring import ScoreKind, score_binary_forecast
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan
from trading_bot.ingestion.runner import IngestionRunLedger
from trading_bot.evaluation.costs import (
    CostBasis,
    EconomicCostModel,
    EconomicCostRegistry,
)


class DecisionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "checkpoint.db"
        self.audit = AuditLedger(self.path)
        self.audit.initialize()
        self.base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def observation(self, index, *, winning):
        generated_at = self.base + timedelta(hours=index)
        target_time = generated_at + timedelta(hours=1)
        actual = 1 if index % 4 in (0, 3) else 0
        predicted = float(actual) if winning else float(1 - actual)
        forecast = Forecast(
            f"forecast-{index}",
            "prediction-specialist",
            "v1",
            f"market-{index}",
            ForecastKind.BINARY_PROBABILITY,
            generated_at,
            generated_at + timedelta(minutes=30),
            {"probability": predicted, "market_probability": 0.5},
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

    def append_observations(self, start, count, *, winning):
        for index in range(start, start + count):
            forecast, score = self.observation(index, winning=winning)
            self.audit.append_forecast(forecast)
            self.audit.append_forecast_score(score)

    def group(self, report):
        return next(
            item
            for item in report.groups
            if item.specialist_id == "prediction-specialist"
        )

    def test_rejected_decision_cannot_requalify_through_continued_sampling(self):
        self.append_observations(0, 30, winning=False)
        report, recorded = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=2)
        )
        rejected = self.group(report)
        self.assertIs(rejected.status, EdgeStatus.REJECTED)
        self.assertIs(rejected.locked_status, EdgeStatus.REJECTED)
        scopes = {decision.scope for decision in recorded}
        self.assertIn(DECISION_SCOPE_AGGREGATE, scopes)
        self.assertIn("forecast_horizon=<=1h", scopes)
        self.assertIn("outcome_horizon=<=1h", scopes)

        self.append_observations(30, 300, winning=True)
        unfrozen = build_walk_forward_report(
            self.audit.forecasts(), self.audit.forecast_scores()
        )
        self.assertIs(self.group(unfrozen).status, EdgeStatus.CANDIDATE)

        report, recorded = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=30)
        )
        self.assertEqual(recorded, ())
        locked = self.group(report)
        self.assertIs(locked.status, EdgeStatus.REJECTED)
        self.assertIs(locked.locked_status, EdgeStatus.REJECTED)
        self.assertEqual(locked.locked_outcomes, 30)
        cohort = next(
            item.evaluation
            for item in report.cohorts
            if item.label == "<=1h" and item.dimension.value == "forecast_horizon"
        )
        self.assertIs(cohort.status, EdgeStatus.REJECTED)
        self.assertIs(cohort.locked_status, EdgeStatus.REJECTED)
        self.assertIs(cohort.monitoring_status, EdgeStatus.CANDIDATE)
        self.assertTrue(
            any("continued sampling cannot revise" in reason for reason in cohort.reasons)
        )
        self.assertIs(self.group(locked_walk_forward_report(self.audit)).status, EdgeStatus.REJECTED)

    def test_candidate_lock_does_not_authorize_after_later_degradation(self):
        self.append_observations(0, 30, winning=True)
        report, recorded = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=2)
        )
        candidate = self.group(report)
        self.assertIs(candidate.status, EdgeStatus.CANDIDATE)
        self.assertIs(candidate.locked_status, EdgeStatus.CANDIDATE)
        self.assertTrue(recorded)

        self.append_observations(30, 300, winning=False)
        report, _ = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=30)
        )
        degraded = self.group(report)
        self.assertIs(degraded.status, EdgeStatus.REJECTED)
        self.assertIs(degraded.locked_status, EdgeStatus.CANDIDATE)
        self.assertIs(degraded.monitoring_status, EdgeStatus.REJECTED)
        self.assertTrue(
            any("does not authorize" in reason for reason in degraded.reasons)
        )

    def test_first_decision_is_recorded_once_and_immutable(self):
        self.append_observations(0, 30, winning=False)
        _, first = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=2)
        )
        self.assertTrue(first)
        _, second = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=3)
        )
        self.assertEqual(second, ())
        stored = self.audit.evaluation_decisions()
        self.assertEqual(len(stored), len(first))

        original = stored[0]
        tampered = replace(
            original,
            status=EdgeStatus.CANDIDATE,
            decided_at=original.decided_at + timedelta(days=1),
        )
        self.assertFalse(self.audit.append_evaluation_decision(tampered))
        self.assertEqual(self.audit.evaluation_decisions()[0], original)

    def test_decisions_round_trip_through_the_ledger(self):
        self.append_observations(0, 30, winning=False)
        _, recorded = checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=2)
        )
        stored = self.audit.evaluation_decisions()
        self.assertEqual(sorted(recorded, key=lambda item: item.decision_id),
                         sorted(stored, key=lambda item: item.decision_id))
        for decision in stored:
            self.assertEqual(decision.boundary, 30)
            self.assertGreaterEqual(decision.independent_outcomes, 30)
            self.assertIsNot(decision.status, EdgeStatus.COLLECTING)

    def test_nonmatching_boundary_decisions_are_ignored(self):
        self.append_observations(0, 30, winning=False)
        checkpointed_walk_forward_report(
            self.audit, as_of=self.base + timedelta(days=2)
        )
        report = locked_walk_forward_report(
            self.audit, EvaluationGateConfig(min_independent_outcomes=40)
        )
        self.assertIsNone(self.group(report).locked_status)

    def test_scorecard_distinguishes_locked_decision_from_monitoring(self):
        PointInTimeStore(self.path).initialize()
        IngestionRunLedger(self.path).initialize()
        self.append_observations(0, 30, winning=False)
        plan = ShadowIngestionPlan(
            "checkpoint-plan",
            (ObservationJob("coinbase-products", "coinbase", "products"),),
        )
        costs = EconomicCostRegistry(
            "checkpoint-costs-v1",
            (
                EconomicCostModel(
                    "prediction-costs",
                    "prediction-specialist",
                    ScoreKind.BINARY,
                    CostBasis.BINARY_CONTRACT,
                    "https://example.com/fees",
                    date(2026, 1, 1),
                    binary_fee_coefficient=0.07,
                ),
            ),
        )
        as_of = self.base + timedelta(days=2)
        build_daily_scorecard(self.path, plan, costs, as_of=as_of, environment={})
        self.append_observations(30, 300, winning=True)
        scorecard = build_daily_scorecard(
            self.path, plan, costs, as_of=self.base + timedelta(days=30), environment={}
        )
        strategy = next(
            item
            for item in scorecard.strategies
            if item.specialist_id == "prediction-specialist"
        )
        self.assertIs(strategy.status, EdgeStatus.REJECTED)
        self.assertIs(strategy.locked_status, EdgeStatus.REJECTED)
        self.assertEqual(strategy.locked_outcomes, 30)
        text = render_scorecard(scorecard, "text")
        self.assertIn("decision: locked rejected at 30 outcomes", text)
        markdown = render_scorecard(scorecard, "markdown")
        self.assertIn("Locked decision", markdown)
        self.assertIn("rejected @ 30", markdown)


if __name__ == "__main__":
    unittest.main()
