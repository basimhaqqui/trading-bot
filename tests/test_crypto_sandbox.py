import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.schemas import AssetClass
from trading_bot.execution.control import ExecutionReceipt
from trading_bot.execution.crypto_sandbox import (
    CryptoSandboxConfig,
    CryptoSandboxLedger,
    SandboxMarketState,
    SandboxScenarioStatus,
    load_crypto_sandbox_config,
    render_crypto_sandbox_report,
    run_crypto_sandbox_scenarios,
    sandbox_scenario_names,
)
from trading_bot.execution.schemas import (
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
)


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


class CryptoSandboxTests(unittest.TestCase):
    def test_checked_in_policy_is_strict_and_versioned(self):
        config = load_crypto_sandbox_config("config/crypto-sandbox.json")

        self.assertEqual(config.version, "crypto-perpetual-sandbox-v1")
        self.assertEqual(config.max_leverage, 5)
        self.assertLess(config.maintenance_margin_pct, 1 / config.max_leverage)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(Path("config/crypto-sandbox.json").read_text())
            payload["unexpected"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys mismatch"):
                load_crypto_sandbox_config(path)

    def test_all_scenarios_pass_without_network_or_credentials(self):
        report = run_crypto_sandbox_scenarios(generated_at=NOW)

        self.assertEqual(len(report.scenarios), 8)
        self.assertEqual(report.passed, 8)
        self.assertEqual(report.failed, 0)
        self.assertTrue(report.successful)
        self.assertFalse(report.network_access)
        self.assertFalse(report.broker_credentials_used)
        self.assertTrue(
            all(item.status is SandboxScenarioStatus.PASSED for item in report.scenarios)
        )
        self.assertEqual(
            sandbox_scenario_names(),
            (
                "spot-fill",
                "perpetual-margin",
                "reduce-only",
                "funding",
                "liquidation",
                "stale-market",
                "eligibility-gates",
                "post-only",
            ),
        )

    def test_one_scenario_can_be_selected_and_unknown_is_rejected(self):
        report = run_crypto_sandbox_scenarios("funding", generated_at=NOW)

        self.assertTrue(report.successful)
        self.assertEqual(report.scenarios[0].scenario_id, "funding")
        with self.assertRaisesRegex(ValueError, "unknown crypto sandbox scenario"):
            run_crypto_sandbox_scenarios("not-a-scenario", generated_at=NOW)

    def test_reports_render_as_text_markdown_and_json(self):
        report = run_crypto_sandbox_scenarios("liquidation", generated_at=NOW)

        text = render_crypto_sandbox_report(report, "text")
        markdown = render_crypto_sandbox_report(report, "markdown")
        payload = json.loads(render_crypto_sandbox_report(report, "json"))
        self.assertIn("network=false credentials=false", text)
        self.assertIn("Crypto and perpetual sandbox", markdown)
        self.assertIn("Cross-margin liquidation", markdown)
        self.assertTrue(payload["successful"])
        self.assertEqual(payload["passed"], 1)

    def test_nonfinite_policy_and_market_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            CryptoSandboxConfig(initial_cash=math.inf)
        with self.assertRaisesRegex(ValueError, "positive prices"):
            SandboxMarketState(
                "sandbox:crypto:BTC-USD",
                AssetClass.CRYPTO,
                99,
                math.nan,
                100,
                NOW,
            )

    def test_reused_intent_id_with_different_contents_fails_closed(self):
        ledger = CryptoSandboxLedger(CryptoSandboxConfig())
        intent = OrderIntent(
            "sandbox-idempotency",
            "sandbox-strategy",
            "v1",
            "sandbox:crypto:BTC-USD",
            "sandbox",
            AssetClass.CRYPTO,
            OrderSide.BUY,
            100,
            ExecutionEnvironment.PAPER,
            (OrderType.LIMIT,),
            NOW + timedelta(minutes=1),
            max_price=100,
            created_at=NOW,
            quantity=1,
        )
        receipt = ExecutionReceipt(
            intent.intent_id,
            ExecutionEnvironment.PAPER,
            "posted",
            NOW,
        )
        ledger.record_receipt(receipt, intent, detail={"test": True})

        with self.assertRaisesRegex(RuntimeError, "reused with different contents"):
            ledger.receipt(replace(intent, notional=101))


if __name__ == "__main__":
    unittest.main()
