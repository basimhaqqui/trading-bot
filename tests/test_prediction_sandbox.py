import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from trading_bot.execution.prediction_sandbox import (
    PredictionAction,
    PredictionApprovalSigner,
    PredictionLiquidity,
    PredictionMarketState,
    PredictionMarketStatus,
    PredictionOrder,
    PredictionOutcome,
    PredictionSandboxConfig,
    PredictionScenarioStatus,
    load_prediction_sandbox_config,
    prediction_scenario_names,
    prediction_trade_fee,
    render_prediction_sandbox_report,
    run_prediction_sandbox_scenarios,
)


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


class PredictionSandboxTests(unittest.TestCase):
    def test_checked_in_policy_is_strict_versioned_and_sourced(self):
        config = load_prediction_sandbox_config("config/prediction-sandbox.json")

        self.assertEqual(config.version, "prediction-settlement-sandbox-v1")
        self.assertEqual(config.fee_schedule_effective_date.isoformat(), "2026-07-07")
        self.assertEqual(config.taker_fee_coefficient, Decimal("0.07"))
        self.assertTrue(config.fee_source_url.startswith("https://"))
        self.assertTrue(config.settlement_source_url.startswith("https://"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(Path("config/prediction-sandbox.json").read_text())
            payload["credential"] = "forbidden"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys mismatch"):
                load_prediction_sandbox_config(path)

    def test_all_scenarios_pass_without_network_credentials_or_orders(self):
        report = run_prediction_sandbox_scenarios(generated_at=NOW)

        self.assertEqual(len(report.scenarios), 8)
        self.assertEqual(report.passed, 8)
        self.assertEqual(report.failed, 0)
        self.assertTrue(report.successful)
        self.assertFalse(report.network_access)
        self.assertFalse(report.venue_credentials_used)
        self.assertEqual(report.real_orders_placed, 0)
        self.assertTrue(
            all(
                item.status is PredictionScenarioStatus.PASSED
                for item in report.scenarios
            )
        )
        self.assertEqual(
            prediction_scenario_names(),
            (
                "yes-settlement",
                "no-settlement",
                "scalar-rounding",
                "lifecycle",
                "idempotency",
                "fees",
                "market-safety",
                "risk-gates",
            ),
        )

    def test_one_scenario_renders_and_unknown_is_rejected(self):
        report = run_prediction_sandbox_scenarios("lifecycle", generated_at=NOW)

        text = render_prediction_sandbox_report(report, "text")
        markdown = render_prediction_sandbox_report(report, "markdown")
        payload = json.loads(render_prediction_sandbox_report(report, "json"))
        self.assertIn("real_orders=0", text)
        self.assertIn("Prediction settlement sandbox", markdown)
        self.assertTrue(payload["successful"])
        self.assertEqual(payload["real_orders_placed"], 0)
        with self.assertRaisesRegex(ValueError, "unknown prediction sandbox scenario"):
            run_prediction_sandbox_scenarios("not-a-scenario", generated_at=NOW)

    def test_market_prices_status_and_default_fee_multipliers_fail_closed(self):
        market = PredictionMarketState(
            "sandbox:prediction:TEST", Decimal("0.49"), Decimal("0.51"), NOW
        )
        self.assertEqual(market.taker_fee_multiplier, 1)
        self.assertEqual(market.maker_fee_multiplier, 0)
        with self.assertRaisesRegex(ValueError, "0 <= bid <= ask <= 1"):
            PredictionMarketState(
                "sandbox:prediction:TEST", Decimal("0.60"), Decimal("0.50"), NOW
            )
        with self.assertRaisesRegex(ValueError, "status is invalid"):
            replace(market, status="active")

    def test_fee_rounding_uses_liquidity_and_series_multiplier(self):
        config = PredictionSandboxConfig()

        self.assertEqual(
            prediction_trade_fee(
                Decimal("0.50"), 100, PredictionLiquidity.TAKER, 1, config
            ),
            Decimal("1.7500"),
        )
        self.assertEqual(
            prediction_trade_fee(
                Decimal("0.50"), 100, PredictionLiquidity.MAKER, 0, config
            ),
            0,
        )
        with self.assertRaisesRegex(ValueError, "positive whole number"):
            prediction_trade_fee(
                Decimal("0.50"), 1.5, PredictionLiquidity.TAKER, 1, config
            )

    def test_signature_tampering_is_detected(self):
        signer = PredictionApprovalSigner(b"prediction-sandbox-signing-key")
        order = PredictionOrder(
            "prediction-signature",
            "prediction-sandbox-strategy",
            "sandbox:prediction:TEST",
            PredictionOutcome.YES,
            PredictionAction.BUY,
            1,
            Decimal("1"),
            NOW,
            NOW.replace(minute=13),
        )
        approval = signer.approve(order, now=NOW)

        self.assertTrue(signer.verify(approval))
        self.assertFalse(signer.verify(replace(approval, order=replace(order, quantity=2))))


if __name__ == "__main__":
    unittest.main()
