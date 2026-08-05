import unittest
from pathlib import Path


class DocumentationTests(unittest.TestCase):
    def test_readme_describes_active_fast_prediction_preregistration(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("Fast-settlement prediction v15", readme)
        self.assertIn("`event_ticker`", readme)
        self.assertIn("recorded `close_time`", readme)
        self.assertIn("fifteen minutes apart", readme)
        self.assertIn("fast-settling v15 baseline", readme)
        self.assertIn("cannot reuse the frozen or earlier fast-lane cohorts", readme)
        self.assertIn("does not use either as the scored boundary", readme)
        self.assertIn("rescheduled-close timestamps remain unscored", readme)
        self.assertIn("`can_close_early=true`", readme)
        self.assertIn("market_lifecycle", readme)
        self.assertIn("market_settlement", readme)
        self.assertIn("freshly restarted bounded fast Kalshi close-time window", readme)
        self.assertNotIn("Fast-settlement prediction v9 records", readme)


if __name__ == "__main__":
    unittest.main()
