import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.execution.memecoin_sandbox import (
    MemecoinRiskSnapshot,
    MemecoinRiskStatus,
    MemecoinSandboxConfig,
    MemecoinScenarioStatus,
    evaluate_memecoin_risk,
    load_memecoin_sandbox_config,
    memecoin_scenario_names,
    render_memecoin_sandbox_report,
    run_memecoin_sandbox_scenarios,
)


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


def safe_snapshot() -> MemecoinRiskSnapshot:
    return MemecoinRiskSnapshot(
        "solana:SAFE", "solana", NOW, 500_000, 720, 20, 5, 0, 50, 10,
        100, 98, False, False, False, False, False, True, True, True,
    )


class MemecoinSandboxTests(unittest.TestCase):
    def test_checked_in_policy_is_strict_versioned_and_sourced(self):
        config = load_memecoin_sandbox_config("config/memecoin-sandbox.json")

        self.assertEqual(config.version, "memecoin-safety-sandbox-v1")
        self.assertTrue(config.token_authority_source_url.startswith("https://"))
        self.assertTrue(config.token_risk_source_url.startswith("https://"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(Path("config/memecoin-sandbox.json").read_text())
            payload["private_key"] = "forbidden"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys mismatch"):
                load_memecoin_sandbox_config(path)

    def test_all_scenarios_pass_without_network_wallet_or_transactions(self):
        report = run_memecoin_sandbox_scenarios(generated_at=NOW)

        self.assertEqual(report.passed, 8)
        self.assertEqual(report.failed, 0)
        self.assertTrue(report.successful)
        self.assertFalse(report.network_access)
        self.assertFalse(report.wallet_credentials_used)
        self.assertEqual(report.real_transactions_signed, 0)
        self.assertTrue(
            all(item.status is MemecoinScenarioStatus.PASSED for item in report.scenarios)
        )
        self.assertEqual(
            memecoin_scenario_names(),
            (
                "safe-token",
                "mint-authority",
                "freeze-delegate",
                "token-extensions",
                "liquidity-concentration",
                "market-integrity",
                "sell-simulation",
                "runtime-limits",
            ),
        )

    def test_risk_evaluation_explains_blocking_authority(self):
        decision = evaluate_memecoin_risk(
            replace(safe_snapshot(), mint_authority_active=True),
            MemecoinSandboxConfig(),
            now=NOW,
        )

        self.assertIs(decision.status, MemecoinRiskStatus.BLOCKED)
        self.assertIn("mint authority can inflate supply", decision.reasons)

    def test_reports_render_and_nonboolean_flags_fail_closed(self):
        report = run_memecoin_sandbox_scenarios("safe-token", generated_at=NOW)

        self.assertIn("transactions=0", render_memecoin_sandbox_report(report, "text"))
        self.assertIn("Memecoin safety sandbox", render_memecoin_sandbox_report(report, "markdown"))
        self.assertTrue(json.loads(render_memecoin_sandbox_report(report, "json"))["successful"])
        with self.assertRaisesRegex(ValueError, "flags must be boolean"):
            replace(safe_snapshot(), source_verified=1)
        with self.assertRaisesRegex(ValueError, "unknown memecoin sandbox scenario"):
            run_memecoin_sandbox_scenarios("unknown", generated_at=NOW)


if __name__ == "__main__":
    unittest.main()
