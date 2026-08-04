import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trading_bot.evaluation.launch_readiness import (
    LaunchReadinessStatus,
    _paper_review_candidate_counts,
    build_launch_readiness_report,
    load_launch_readiness_config,
    render_launch_readiness_report,
)
from trading_bot.evaluation.scorecard import (
    FastPredictionCadenceSummary,
    RapidCryptoCadenceSummary,
    build_daily_scorecard,
)
from trading_bot.evaluation.costs import load_cost_registry
from trading_bot.evaluation.economics import EconomicStatus
from trading_bot.evaluation.reporting import EdgeStatus
from trading_bot.ingestion.plan import load_plan


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


class LaunchReadinessTests(unittest.TestCase):
    def test_policy_is_strict_and_cannot_authorize_live_execution(self):
        config = load_launch_readiness_config("config/launch-readiness.json")

        self.assertEqual(config.roadmap_milestones, 18)
        self.assertFalse(config.allow_live_execution)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(Path("config/launch-readiness.json").read_text())
            payload["allow_live_execution"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot authorize live"):
                load_launch_readiness_config(path)

    def test_fresh_database_completes_roadmap_but_remains_no_go(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_launch_readiness_report(
                Path(directory) / "readiness.db",
                plan_path="config/shadow-ingestion.json",
                costs_path="config/economic-costs.json",
                as_of=NOW,
            )

        self.assertEqual((report.roadmap_completed, report.roadmap_total), (18, 18))
        self.assertIs(report.status, LaunchReadinessStatus.NO_GO)
        self.assertTrue(report.technical_successful)
        self.assertFalse(report.live_execution_authorized)
        self.assertEqual(report.real_orders_placed, 0)
        self.assertTrue(all(gate.passed for gate in report.gates if gate.category.value == "sandbox"))
        self.assertIn("ingestion-health", {gate.gate_id for gate in report.blockers})
        self.assertIn("forecast-candidates", {gate.gate_id for gate in report.blockers})
        self.assertIn("after-cost-candidates", {gate.gate_id for gate in report.blockers})

    def test_reports_render_machine_and_human_status(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_launch_readiness_report(
                Path(directory) / "readiness.db",
                plan_path="config/shadow-ingestion.json",
                costs_path="config/economic-costs.json",
                as_of=NOW,
            )

        self.assertIn("NO_GO", render_launch_readiness_report(report, "text"))
        self.assertIn("roadmap 18/18", render_launch_readiness_report(report, "markdown"))
        payload = json.loads(render_launch_readiness_report(report, "json"))
        self.assertEqual(payload["status"], "no_go")
        self.assertFalse(payload["live_execution_authorized"])
        self.assertTrue(payload["technical_successful"])

    def test_requires_observed_continuity_for_both_rapid_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "readiness.db"
            report = build_launch_readiness_report(
                database,
                plan_path="config/shadow-ingestion.json",
                costs_path="config/economic-costs.json",
                as_of=NOW,
            )

        gates = {gate.gate_id: gate for gate in report.gates}
        self.assertFalse(gates["rapid-crypto-continuity"].passed)
        self.assertIn("no observed collection cycles", gates["rapid-crypto-continuity"].detail)
        self.assertFalse(gates["fast-prediction-continuity"].passed)
        self.assertIn("no observed collection cycles", gates["fast-prediction-continuity"].detail)

    def test_superseded_fast_lane_cannot_satisfy_paper_review_candidate_counts(self):
        scorecard = SimpleNamespace(
            strategies=(
                SimpleNamespace(
                    specialist_id="prediction-market-fast-settlement-baseline-v14",
                    status=EdgeStatus.CANDIDATE,
                ),
                SimpleNamespace(
                    specialist_id="prediction-market-fast-settlement-baseline-v15",
                    status=EdgeStatus.CANDIDATE,
                ),
            ),
            economics=(
                SimpleNamespace(
                    specialist_id="prediction-market-fast-settlement-baseline-v6",
                    status=EconomicStatus.CANDIDATE,
                ),
                SimpleNamespace(
                    specialist_id="prediction-market-fast-settlement-baseline-v15",
                    status=EconomicStatus.CANDIDATE,
                ),
            ),
        )

        self.assertEqual(_paper_review_candidate_counts(scorecard), (1, 1))

    def test_accepts_rapid_lane_cadence_only_within_each_fixed_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "readiness.db"
            build_launch_readiness_report(
                database,
                plan_path="config/shadow-ingestion.json",
                costs_path="config/economic-costs.json",
                as_of=NOW,
            )
            base = build_daily_scorecard(
                database,
                load_plan("config/shadow-ingestion.json"),
                load_cost_registry("config/economic-costs.json"),
                as_of=NOW,
                environment={},
            )
            scorecard = replace(
                base,
                rapid_crypto_cadence=RapidCryptoCadenceSummary(
                    ("coinbase-btc-fifteen-minute-candles",),
                    4,
                    NOW,
                    30.0,
                    30.0,
                    24.0,
                ),
                fast_prediction_cadence=FastPredictionCadenceSummary(
                    ("kalshi-fast-settling-markets",),
                    4,
                    NOW,
                    31.0,
                    30.0,
                    24.0,
                ),
            )
            with patch(
                "trading_bot.evaluation.launch_readiness.build_daily_scorecard",
                return_value=scorecard,
            ):
                report = build_launch_readiness_report(
                    database,
                    plan_path="config/shadow-ingestion.json",
                    costs_path="config/economic-costs.json",
                    as_of=NOW,
                )

        gates = {gate.gate_id: gate for gate in report.gates}
        self.assertTrue(gates["rapid-crypto-continuity"].passed)
        self.assertIn("within the 30-minute bound", gates["rapid-crypto-continuity"].detail)
        self.assertFalse(gates["fast-prediction-continuity"].passed)
        self.assertIn("exceeds the 30-minute bound", gates["fast-prediction-continuity"].detail)


if __name__ == "__main__":
    unittest.main()
