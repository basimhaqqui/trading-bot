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
        self.assertIn("--research-profile rapid", rapid_workflow)
        self.assertIn("--research-profile rapid", rapid_deployment)
        # The run must leave scheduler headroom while allowing the complete
        # Neon-backed observation to finish before the next 15-minute trigger.
        self.assertIn("timeout-minutes: 14", rapid_workflow)
        self.assertIn("timeout-minutes: 14", rapid_deployment)

    def test_rapid_workflow_uses_persistent_postgres_evidence_state(self):
        workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        plan = json.loads(Path("config/rapid-shadow-ingestion.json").read_text())

        self.assertIn("group: shadow-market-observation", workflow)
        self.assertIn("TRADING_DB_PATH: ${{ secrets.SHADOW_DATABASE_URL }}", workflow)
        self.assertIn("TRADING_DB_SCHEMA: shadow_evidence_v2", workflow)
        self.assertIn("Require persistent shadow database", workflow)
        self.assertIn("Verify persistent shadow database", workflow)
        self.assertIn("trading-bot persistence-check", workflow)
        self.assertIn("Restore disposable rapid working set", workflow)
        self.assertIn("actions/cache/restore@v5", workflow)
        self.assertIn("Save disposable rapid working set", workflow)
        self.assertIn("actions/cache/save@v5", workflow)
        self.assertIn("var/rapid-working-set.db", workflow)
        self.assertNotIn("var/trading.db", workflow)
        self.assertIn("--max-neon-egress-bytes 1400000", workflow)
        self.assertIn("--egress-report var/reports/rapid-neon-egress.md", workflow)
        self.assertIn("Publish rapid egress accounting", workflow)
        self.assertIn("config/rapid-shadow-ingestion.json", workflow)
        self.assertIn("Verify rapid evidence continuity", workflow)
        self.assertIn("trading-bot rapid-continuity", workflow)
        self.assertIn("rapid-continuity-${{ github.run_id }}", workflow)
        self.assertIn("retention-days: 90", workflow)
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

    def test_rapid_workflow_reports_health_after_collection_failure(self):
        workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()

        health_step = workflow.split("- name: Enforce rapid ingestion health", 1)[1]
        self.assertIn("id: persistence_check", workflow)
        self.assertIn(
            "if: ${{ always() && steps.persistence_check.outcome == 'success' }}",
            health_step,
        )

    def test_database_followups_skip_unavailable_persistent_evidence(self):
        rapid_workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        full_workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        expected_condition = (
            "if: ${{ always() && steps.persistence_check.outcome == 'success' }}"
        )
        self.assertIn(expected_condition, rapid_workflow)
        self.assertEqual(full_workflow.count(expected_condition), 2)

    def test_workflows_attest_when_persistent_evidence_is_unavailable(self):
        rapid_workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        full_workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        expected_condition = (
            "if: ${{ always() && steps.persistence_check.outcome == 'failure' }}"
        )
        for workflow in (rapid_workflow, full_workflow):
            self.assertIn("Attest unavailable persistent evidence", workflow)
            self.assertIn(expected_condition, workflow)
            self.assertIn("## Shadow evidence unavailable", workflow)
            self.assertIn("retrospective replacement evidence", workflow)
            self.assertIn("$GITHUB_STEP_SUMMARY", workflow)

    def test_persistent_preflight_failures_publish_sanitized_availability_artifacts(self):
        rapid_workflow = Path(".github/workflows/rapid-shadow-ingestion.yml").read_text()
        full_workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        for workflow, artifact_name in (
            (rapid_workflow, "rapid-persistence-unavailable-${{ github.run_id }}"),
            (full_workflow, "full-persistence-unavailable-${{ github.run_id }}"),
        ):
            self.assertIn("Upload persistent-evidence availability attestation", workflow)
            self.assertIn("persistence-unavailable.json", workflow)
            self.assertIn(artifact_name, workflow)
            self.assertIn('"status": "persistent_preflight_failed"', workflow)
            self.assertIn('"evidence_collected": false', workflow)
            self.assertIn('"retrospective_replacement_used": false', workflow)
            self.assertIn("retention-days: 90", workflow)

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

    def test_full_workflow_requires_neon_and_uses_only_a_disposable_working_cache(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        self.assertIn("TRADING_DB_PATH: ${{ secrets.SHADOW_DATABASE_URL }}", workflow)
        self.assertIn("TRADING_DB_SCHEMA: shadow_evidence_v2", workflow)
        self.assertIn("Require persistent shadow database", workflow)
        self.assertIn("SHADOW_DATABASE_URL must be configured", workflow)
        self.assertIn("Verify persistent shadow database", workflow)
        self.assertIn("trading-bot persistence-check", workflow)
        self.assertIn("Restore disposable full working set", workflow)
        self.assertIn("actions/cache/restore@v5", workflow)
        self.assertIn("Save disposable full working set", workflow)
        self.assertIn("actions/cache/save@v5", workflow)
        self.assertIn("var/full-working-set.db", workflow)
        self.assertNotIn("shadow-database", workflow)
        self.assertNotIn("var/trading.db", workflow)
        self.assertIn("--max-neon-egress-bytes 1400000", workflow)
        self.assertIn("--egress-report var/reports/full-neon-egress.md", workflow)

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

    def test_one_time_migration_is_manual_and_fails_closed(self):
        workflow = Path(".github/workflows/shadow-persistence-migration.yml").read_text()

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("MIGRATE-PERSISTENCE", workflow)
        self.assertIn("actions/cache/restore@v5", workflow)
        self.assertIn("fail-on-cache-miss: true", workflow)
        self.assertIn("trading-bot migrate-sqlite --source var/trading.db", workflow)
        self.assertIn("TRADING_DB_PATH: ${{ secrets.SHADOW_DATABASE_URL }}", workflow)
        self.assertIn("TRADING_DB_SCHEMA: shadow_evidence_v2", workflow)

    def test_migration_deployment_template_matches_live_workflow(self):
        workflow = Path(".github/workflows/shadow-persistence-migration.yml").read_text()
        deployment = Path("deployment/shadow-persistence-migration.github-actions.yml").read_text()

        self.assertEqual(deployment, workflow)

    def test_archive_is_manual_full_fidelity_and_release_write_isolated(self):
        workflow = Path(".github/workflows/shadow-evidence-archive.yml").read_text()

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("ARCHIVE-EVIDENCE", workflow)
        self.assertIn("actions/cache/restore@v5", workflow)
        self.assertIn("fail-on-cache-miss: true", workflow)
        self.assertIn("--json-output archive/snapshot-manifest.json", workflow)
        self.assertIn("zstd -T0 -19", workflow)
        self.assertIn("split --bytes=1900000000", workflow)
        self.assertIn("sha256sum > SHA256SUMS", workflow)
        self.assertIn("retention-days: 1", workflow)
        build = workflow.split("  build:", 1)[1].split("  release:", 1)[0]
        release = workflow.split("  release:", 1)[1]
        self.assertIn("contents: read", build)
        self.assertNotIn("contents: write", build)
        self.assertIn("contents: write", release)
        self.assertIn("GH_REPO: ${{ github.repository }}", release)
        self.assertIn("gh release create", release)
        self.assertIn("--draft", release)
        self.assertIn("gh release edit", release)
        self.assertIn(".immutable == true", release)

    def test_restore_requires_immutable_release_and_rebuilds_database(self):
        workflow = Path(".github/workflows/shadow-evidence-restore.yml").read_text()

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("contents: read", workflow)
        self.assertNotIn("contents: write", workflow)
        self.assertIn(".immutable == true", workflow)
        self.assertIn("sha256sum --check SHA256SUMS", workflow)
        self.assertIn("shadow-evidence.sqlite.zst.part-*", workflow)
        self.assertIn("zstd --decompress", workflow)
        self.assertIn("--output restore/rebuilt.sqlite", workflow)
        self.assertIn("trading-bot --db restore/rebuilt.sqlite doctor", workflow)
        self.assertIn("retention-days: 90", workflow)

    def test_archive_and_restore_deployment_templates_match(self):
        archive = Path(".github/workflows/shadow-evidence-archive.yml").read_text()
        archive_deployment = Path(
            "deployment/shadow-evidence-archive.github-actions.yml"
        ).read_text()
        restore = Path(".github/workflows/shadow-evidence-restore.yml").read_text()
        restore_deployment = Path(
            "deployment/shadow-evidence-restore.github-actions.yml"
        ).read_text()

        self.assertEqual(archive_deployment, archive)
        self.assertEqual(restore_deployment, restore)


if __name__ == "__main__":
    unittest.main()
