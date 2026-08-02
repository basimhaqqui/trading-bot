import unittest
from pathlib import Path


class DocumentationTests(unittest.TestCase):
    def test_readme_describes_active_fast_prediction_preregistration(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("Fast-settlement prediction v11", readme)
        self.assertIn("`event_ticker`", readme)
        self.assertIn("recorded `close_time`", readme)
        self.assertIn("fifteen minutes apart", readme)
        self.assertIn("market_lifecycle", readme)
        self.assertIn("market_settlement", readme)
        self.assertNotIn("Fast-settlement prediction v9 records", readme)


if __name__ == "__main__":
    unittest.main()
