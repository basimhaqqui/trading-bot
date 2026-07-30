import unittest
from unittest.mock import patch

from trading_bot.cli import _persistence_check


class PersistenceCheckTests(unittest.TestCase):
    def test_rejects_non_postgres_database_before_any_integrity_check(self):
        with (
            patch("trading_bot.cli.is_postgres_location", return_value=False),
            patch("trading_bot.cli.postgres_integrity_ok") as integrity,
            self.assertRaisesRegex(RuntimeError, "requires a PostgreSQL"),
        ):
            _persistence_check("var/trading.db")

        integrity.assert_not_called()

    def test_verifies_configured_postgres_database(self):
        with (
            patch("trading_bot.cli.is_postgres_location", return_value=True),
            patch("trading_bot.cli.postgres_integrity_ok", return_value=True) as integrity,
        ):
            self.assertEqual(_persistence_check("postgresql://configured"), 0)

        integrity.assert_called_once_with("postgresql://configured")


if __name__ == "__main__":
    unittest.main()
