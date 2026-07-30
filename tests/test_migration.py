import sqlite3
import tempfile
import unittest
from pathlib import Path

from trading_bot.core.migration import (
    _legacy_connection,
    _require_locked_paper_control,
    _verified_overlap,
    migrate_sqlite_to_postgres,
)


class MigrationTests(unittest.TestCase):
    def test_existing_identical_and_target_only_rows_are_preserved(self):
        source = [("source-a", "venue-a"), ("shared", "venue-b")]
        target = [("shared", "venue-b"), ("target-c", "venue-c")]

        overlap = _verified_overlap("instruments", source, target)

        self.assertEqual(overlap, 1)

    def test_existing_conflicting_rows_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "conflicting rows"):
            _verified_overlap(
                "instruments",
                [("shared", "source-value")],
                [("shared", "target-value")],
            )

    def test_migration_refuses_non_postgres_target_before_reading_source(self):
        with self.assertRaisesRegex(ValueError, "target must be PostgreSQL"):
            migrate_sqlite_to_postgres(Path("missing.db"), "var/trading.db")

    def test_unlocked_legacy_paper_control_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE paper_execution_control "
                "(environment TEXT, enabled INTEGER, kill_switch_active INTEGER)"
            )
            connection.execute(
                "INSERT INTO paper_execution_control VALUES ('paper', 1, 0)"
            )
            connection.commit()
            connection.close()
            source = _legacy_connection(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "unlocked paper-control"):
                    _require_locked_paper_control(source)
            finally:
                source.close()


if __name__ == "__main__":
    unittest.main()
