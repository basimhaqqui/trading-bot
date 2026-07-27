import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.core.experiments import (
    ExperimentConflictError,
    ExperimentRegistry,
    ExperimentStatus,
)
from trading_bot.core.schemas import AssetClass, Hypothesis
from trading_bot.agents.hypotheses import (
    BASELINE_HYPOTHESES,
    PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS,
)


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(Path(self.temp.name) / "experiments.db")
        self.registry.initialize()
        self.hypothesis = Hypothesis(
            "hyp-1",
            "crypto-momentum",
            AssetClass.CRYPTO,
            "slow information diffusion creates return continuation",
            "next-day return",
            "1d",
            ("point-in-time trades",),
            ("effect disappears after costs",),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_registry_counts_rejected_trials(self):
        experiment = self.registry.start(
            self.hypothesis,
            config={"lookback": 7},
            code_version="test",
            data_cutoff=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        self.registry.finish(
            experiment.experiment_id,
            status=ExperimentStatus.REJECTED,
            metrics={"reason": "fails costs"},
        )
        stored = self.registry.get(experiment.experiment_id)
        self.assertEqual(stored.status, ExperimentStatus.REJECTED)
        self.assertEqual(self.registry.trial_count("crypto-momentum"), 1)

    def test_hypothesis_identity_cannot_be_silently_redefined(self):
        self.registry.register_hypothesis(self.hypothesis)
        changed = Hypothesis(
            **{**self.hypothesis.__dict__, "mechanism": "different mechanism"}
        )
        with self.assertRaises(ExperimentConflictError):
            self.registry.register_hypothesis(changed)

    def test_fast_settlement_v2_preserves_the_registered_v1_record(self):
        self.registry.register_hypothesis(PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS)
        for hypothesis in BASELINE_HYPOTHESES:
            self.registry.register_hypothesis(hypothesis)


if __name__ == "__main__":
    unittest.main()
