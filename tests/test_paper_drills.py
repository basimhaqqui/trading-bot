import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.execution.drills import (
    DrillStatus,
    render_paper_drill_report,
    run_paper_drills,
    scenario_names,
)
from trading_bot.execution.operations import (
    PaperControlStore,
    activate_paper_emergency_stop,
)


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


class PaperDrillTests(unittest.TestCase):
    def test_all_isolated_drills_pass_without_network_or_credentials(self):
        report = run_paper_drills("all", generated_at=NOW)

        self.assertEqual(len(report.scenarios), 8)
        self.assertEqual(report.passed, 8)
        self.assertEqual(report.failed, 0)
        self.assertTrue(report.successful)
        self.assertFalse(report.network_access)
        self.assertFalse(report.broker_credentials_used)
        self.assertTrue(
            all(item.status is DrillStatus.PASSED for item in report.scenarios)
        )
        self.assertEqual(
            scenario_names(),
            (
                "duplicate-submission",
                "ambiguous-timeout",
                "partial-fill",
                "remote-rejection",
                "stale-market-data",
                "reconciliation-mismatch",
                "daily-loss-shutdown",
                "emergency-stop-recovery",
            ),
        )

    def test_one_scenario_can_be_selected(self):
        report = run_paper_drills("ambiguous-timeout", generated_at=NOW)

        self.assertEqual(len(report.scenarios), 1)
        self.assertEqual(report.scenarios[0].scenario_id, "ambiguous-timeout")
        self.assertTrue(report.successful)

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown paper drill scenario"):
            run_paper_drills("not-a-drill", generated_at=NOW)

    def test_reports_render_as_text_markdown_and_json(self):
        report = run_paper_drills("partial-fill", generated_at=NOW)

        text = render_paper_drill_report(report, "text")
        markdown = render_paper_drill_report(report, "markdown")
        payload = json.loads(render_paper_drill_report(report, "json"))
        self.assertIn("network=false credentials=false", text)
        self.assertIn("Paper incident drills", markdown)
        self.assertIn("Partial fill", markdown)
        self.assertTrue(payload["successful"])
        self.assertEqual(payload["passed"], 1)
        self.assertFalse(payload["network_access"])

    def test_emergency_stop_locks_before_cancellation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            controls = PaperControlStore(Path(directory) / "paper.db")
            controls.release_kill_switch(
                confirmation="PAPER-ONLY", reason="test setup", now=NOW
            )
            controls.enable(confirmation="PAPER-ONLY", reason="test setup", now=NOW)

            def fail_cancellation():
                raise RuntimeError("simulated cancellation outage")

            with self.assertRaisesRegex(RuntimeError, "cancellation outage"):
                activate_paper_emergency_stop(
                    controls,
                    reason="test emergency",
                    cancel_open_orders=fail_cancellation,
                )

            status = controls.status()
            self.assertTrue(status.kill_switch_active)
            self.assertFalse(status.enabled)


if __name__ == "__main__":
    unittest.main()
