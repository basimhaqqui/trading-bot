import json
import unittest
from pathlib import Path


class ShadowWorkflowTests(unittest.TestCase):
    def test_rapid_crypto_cadence_preserves_a_post_close_buffer(self):
        full_workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        rapid_workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        rapid_deployment = Path(
            "deployment/rapid-shadow-ingestion.github-actions.yml"
        ).read_text()

        self.assertIn('cron: "7 * * * *"', full_workflow)
        expected_cron = 'cron: "22,37,52 * * * *"'
        self.assertIn(expected_cron, rapid_workflow)
        self.assertIn(expected_cron, rapid_deployment)
        self.assertIn("timeout-minutes: 12", rapid_workflow)

    def test_rapid_workflow_uses_persistent_postgres_evidence_state(self):
        workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        plan = json.loads(Path("config/rapid-shadow-ingestion.json").read_text())

        self.assertIn("group: shadow-market-observation", workflow)
        self.assertIn("TRADING_DB_PATH: ${{ secrets.SHADOW_DATABASE_URL }}", workflow)
        self.assertIn("Require persistent shadow database", workflow)
        self.assertNotIn("actions/cache/", workflow)
        self.assertNotIn("var/trading.db", workflow)
        self.assertIn("config/rapid-shadow-ingestion.json", workflow)
        self.assertEqual(plan["name"], "public-shadow-observation")
        self.assertEqual(
            {job["job_id"] for job in plan["jobs"]},
            {
                "kalshi-fast-settling-markets",
                "kalshi-forecast-outcomes",
                "coinbase-btc-fifteen-minute-candles",
                "coinbase-eth-fifteen-minute-candles",
                "coinbase-sol-fifteen-minute-candles",
                "coinbase-doge-fifteen-minute-candles",
                "coinbase-xrp-fifteen-minute-candles",
                "coinbase-ada-fifteen-minute-candles",
                "coinbase-avax-fifteen-minute-candles",
                "coinbase-link-fifteen-minute-candles",
                "coinbase-hype-fifteen-minute-candles",
                "coinbase-pump-fifteen-minute-candles",
            },
        )

    def test_only_scheduled_workflow_cycles_attest_scheduled_evidence(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        deployment = Path("deployment/shadow-ingestion.github-actions.yml").read_text()

        expected_origin = (
            "--observation-origin ${{ github.event_name == 'schedule' "
            "&& 'scheduled' || 'manual' }}"
        )
        self.assertIn(expected_origin, workflow)
        self.assertIn(expected_origin, deployment)

    def test_scheduled_runs_fail_when_rapid_evidence_continuity_is_broken(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        deployment = Path("deployment/shadow-ingestion.github-actions.yml").read_text()

        expected_flag = "--fail-on-rapid-continuity"
        self.assertIn(expected_flag, workflow)
        self.assertIn(expected_flag, deployment)

    def test_live_shadow_workflow_passes_the_read_only_solana_endpoint(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        deployment = Path("deployment/shadow-ingestion.github-actions.yml").read_text()

        expected_binding = (
            "SOLANA_READ_ONLY_RPC_URL: ${{ secrets.SOLANA_READ_ONLY_RPC_URL }}"
        )
        self.assertIn(expected_binding, workflow)
        self.assertIn(expected_binding, deployment)

    def test_full_workflow_requires_neon_and_never_falls_back_to_cache(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        self.assertIn("TRADING_DB_PATH: ${{ secrets.SHADOW_DATABASE_URL }}", workflow)
        self.assertIn("Require persistent shadow database", workflow)
        self.assertIn("SHADOW_DATABASE_URL must be configured", workflow)
        self.assertNotIn("actions/cache/", workflow)
        self.assertNotIn("shadow-database", workflow)
        self.assertNotIn("var/trading.db", workflow)

    def test_scorecards_are_retained_even_when_a_continuity_gate_fails(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        upload = workflow.split("- name: Upload daily scorecard", 1)[1]
        self.assertIn("if: ${{ always()", upload)
        self.assertIn("hashFiles('var/reports/daily-scorecard.json')", upload)
        self.assertNotIn("steps.recovery.outputs.create", upload)
        self.assertIn("retention-days: 90", upload)

    def test_deployment_template_matches_the_live_shadow_workflow(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        deployment = Path("deployment/shadow-ingestion.github-actions.yml").read_text()

        self.assertEqual(deployment, workflow)

    def test_rapid_deployment_template_matches_the_live_workflow(self):
        workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        deployment = Path(
            "deployment/rapid-shadow-ingestion.github-actions.yml"
        ).read_text()

        self.assertEqual(deployment, workflow)


if __name__ == "__main__":
    unittest.main()
