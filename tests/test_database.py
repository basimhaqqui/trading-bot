import unittest

from trading_bot.core.database import (
    DatabaseConfigurationError,
    _postgres_sql,
    _postgres_row_factory,
    connect_database,
    database_display_name,
    is_postgres_location,
)


POOLER_URL = (
    "postgresql://shadow:secret@ep-shadow-pooler.us-east-2.aws.neon.tech/"
    "shadow?sslmode=require"
)


class DatabaseTests(unittest.TestCase):
    def test_row_factory_accepts_commands_without_result_metadata(self):
        class CommandCursor:
            description = None

        self.assertEqual(_postgres_row_factory(CommandCursor())(("ignored",)), ("ignored",))

    def test_pooled_neon_urls_select_postgres_without_exposing_the_url(self):
        self.assertTrue(is_postgres_location(POOLER_URL))
        self.assertEqual(database_display_name(POOLER_URL), "PostgreSQL")

    def test_postgres_translation_preserves_idempotency_and_json_paths(self):
        query = """
            INSERT OR IGNORE INTO audit_records (record_type, record_id)
            VALUES (?, ?)
            WHERE json_extract(payload_json, '$.values.event_ticker') = ?
        """

        translated = _postgres_sql(query)

        self.assertIn("INSERT INTO audit_records", translated)
        self.assertIn("ON CONFLICT DO NOTHING", translated)
        self.assertIn("(payload_json::jsonb #>> '{values,event_ticker}')", translated)
        self.assertNotIn("?", translated)

    def test_non_pooled_or_unencrypted_postgres_urls_are_rejected(self):
        with self.assertRaises(DatabaseConfigurationError):
            with connect_database("postgresql://shadow:secret@localhost/shadow?sslmode=require"):
                pass
        with self.assertRaises(DatabaseConfigurationError):
            with connect_database(
                "postgresql://shadow:secret@ep-shadow-pooler.us-east-2.aws.neon.tech/shadow"
            ):
                pass


if __name__ == "__main__":
    unittest.main()
