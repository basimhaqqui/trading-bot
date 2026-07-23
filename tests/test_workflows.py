import unittest
from pathlib import Path


class ShadowWorkflowTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
