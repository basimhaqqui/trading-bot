import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.evaluation.launch_readiness import (
    LaunchReadinessStatus,
    build_launch_readiness_report,
    load_launch_readiness_config,
    render_launch_readiness_report,
)


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


if __name__ == "__main__":
    unittest.main()
