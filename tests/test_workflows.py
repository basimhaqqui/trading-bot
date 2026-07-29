import unittest
from pathlib import Path


class ShadowWorkflowTests(unittest.TestCase):
    def test_rapid_crypto_cadence_preserves_a_post_close_buffer(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        deployment = Path("deployment/shadow-ingestion.github-actions.yml").read_text()

        expected_cron = 'cron: "7,22,37,52 * * * *"'
        self.assertIn(expected_cron, workflow)
        self.assertIn(expected_cron, deployment)

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

    def test_manual_recovery_artifacts_are_opt_in(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()

        self.assertIn("create_recovery_checkpoint:", workflow)
        checkpoint_input = workflow.split("create_recovery_checkpoint:", 1)[1].split(
            "schedule:", 1
        )[0]
        self.assertIn("default: false", checkpoint_input)
        self.assertIn("type: boolean", checkpoint_input)
        self.assertIn("CREATE_RECOVERY_CHECKPOINT:", workflow)
        self.assertIn('if [[ "$CREATE_RECOVERY_CHECKPOINT" == "true"', workflow)
        self.assertNotIn('if [[ "$EVENT_NAME" == "workflow_dispatch"', workflow)
        self.assertIn("retention-days: 7", workflow)

    def test_deployment_template_matches_the_live_shadow_workflow(self):
        workflow = Path(".github/workflows/shadow-ingestion.yml").read_text()
        deployment = Path("deployment/shadow-ingestion.github-actions.yml").read_text()

        self.assertEqual(deployment, workflow)


if __name__ == "__main__":
    unittest.main()
