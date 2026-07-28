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
    CRYPTO_INTRADAY_MOMENTUM_V2_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS,
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

    def test_fast_settlement_versions_preserve_prior_registered_records(self):
        self.registry.register_hypothesis(PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS)
        self.registry.register_hypothesis(PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS)
        self.registry.register_hypothesis(PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS)
        self.registry.register_hypothesis(PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS)
        for hypothesis in BASELINE_HYPOTHESES:
            self.registry.register_hypothesis(hypothesis)

    def test_intraday_v2_is_registered_separately_from_v1(self):
        self.registry.register_hypothesis(CRYPTO_INTRADAY_MOMENTUM_V2_HYPOTHESIS)
        self.registry.register_hypothesis(CRYPTO_INTRADAY_MOMENTUM_V2_HYPOTHESIS)


if __name__ == "__main__":
    unittest.main()
