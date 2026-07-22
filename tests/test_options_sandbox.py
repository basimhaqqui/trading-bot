import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.execution.options_sandbox import (
    OptionContract,
    OptionMarketState,
    OptionRight,
    OptionsSandboxConfig,
    OptionsScenarioStatus,
    load_options_sandbox_config,
    options_scenario_names,
    render_options_sandbox_report,
    run_options_sandbox_scenarios,
)


NOW = datetime(2026, 1, 2, 15, tzinfo=timezone.utc)


class OptionsSandboxTests(unittest.TestCase):
    def test_checked_in_policy_is_strict_versioned_and_sourced(self):
        config = load_options_sandbox_config("config/options-sandbox.json")

        self.assertEqual(config.version, "options-lifecycle-sandbox-v1")
        self.assertEqual(config.contract_multiplier, 100)
        self.assertTrue(config.exercise_source_url.startswith("https://"))
        self.assertTrue(config.activity_source_url.startswith("https://"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(Path("config/options-sandbox.json").read_text())
            payload["credential"] = "forbidden"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys mismatch"):
                load_options_sandbox_config(path)

    def test_all_scenarios_pass_without_network_credentials_or_orders(self):
        report = run_options_sandbox_scenarios(generated_at=NOW)

        self.assertEqual(report.passed, 8)
        self.assertEqual(report.failed, 0)
        self.assertTrue(report.successful)
        self.assertFalse(report.network_access)
        self.assertFalse(report.venue_credentials_used)
        self.assertEqual(report.real_orders_placed, 0)
        self.assertTrue(
            all(item.status is OptionsScenarioStatus.PASSED for item in report.scenarios)
        )
        self.assertEqual(
            options_scenario_names(),
            (
                "call-exercise",
                "put-exercise",
                "do-not-exercise",
                "otm-expiry",
                "risk-sellout",
                "delta-hedge",
                "stale-and-gates",
                "limits",
            ),
        )

    def test_reports_render_and_unknown_scenario_is_rejected(self):
        report = run_options_sandbox_scenarios("delta-hedge", generated_at=NOW)

        self.assertIn("real_orders=0", render_options_sandbox_report(report, "text"))
        self.assertIn("Options lifecycle sandbox", render_options_sandbox_report(report, "markdown"))
        payload = json.loads(render_options_sandbox_report(report, "json"))
        self.assertTrue(payload["successful"])
        with self.assertRaisesRegex(ValueError, "unknown options sandbox scenario"):
            run_options_sandbox_scenarios("unknown", generated_at=NOW)

    def test_nonstandard_contract_and_invalid_market_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "standard American"):
            OptionContract("option:test", "TEST", OptionRight.CALL, 100, NOW, 10)
        with self.assertRaisesRegex(ValueError, "finite"):
            OptionMarketState("option:test", 1, 2, 100, math.inf, NOW)
        with self.assertRaisesRegex(ValueError, "finite"):
            OptionsSandboxConfig(initial_cash=math.nan)


if __name__ == "__main__":
    unittest.main()
