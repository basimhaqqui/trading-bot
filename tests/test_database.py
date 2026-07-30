import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trading_bot.core.database import (
    DatabaseConfigurationError,
    _postgres_sql,
    _postgres_row_factory,
    connect_database,
    database_display_name,
    is_postgres_location,
    PostgresConnection,
    postgres_schema,
)


POOLER_URL = (
    "postgresql://shadow:secret@ep-shadow-pooler.us-east-2.aws.neon.tech/"
    "shadow?sslmode=require"
)


class DatabaseTests(unittest.TestCase):
    def test_postgres_schema_is_validated_before_connection_use(self):
        with patch.dict("os.environ", {"TRADING_DB_SCHEMA": "shadow_evidence_v1"}):
            self.assertEqual(postgres_schema(), "shadow_evidence_v1")
        with patch.dict("os.environ", {"TRADING_DB_SCHEMA": "public; DROP SCHEMA public"}):
            with self.assertRaises(DatabaseConfigurationError):
                postgres_schema()

    def test_pooled_connection_sets_isolated_schema_after_connect(self):
        class FakeConnection:
            def __init__(self):
                self.calls = []
                self.committed = False
                self.closed = False

            def execute(self, *args):
                self.calls.append(args)

            def commit(self):
                self.committed = True

            def rollback(self):
                raise AssertionError("unexpected rollback")

            def close(self):
                self.closed = True

        fake_connection = FakeConnection()
        connect_calls = []

        def connect(*args, **kwargs):
            connect_calls.append((args, kwargs))
            return fake_connection

        with patch.dict(
            "sys.modules", {"psycopg": SimpleNamespace(connect=connect)}
        ), patch.dict("os.environ", {"TRADING_DB_SCHEMA": "shadow_evidence_v1"}):
            with connect_database(POOLER_URL):
                pass

        self.assertEqual(connect_calls[0][0], (POOLER_URL,))
        self.assertNotIn("options", connect_calls[0][1])
        self.assertEqual(
            fake_connection.calls,
            [("SET LOCAL search_path TO shadow_evidence_v1",)],
        )
        self.assertTrue(fake_connection.committed)
        self.assertTrue(fake_connection.closed)

    def test_batch_sql_uses_a_postgres_cursor(self):
        class FakeCursor:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def executemany(self, *args):
                self.calls.append(args)

        class FakeConnection:
            def __init__(self):
                self.batch_cursor = FakeCursor()

            def cursor(self):
                return self.batch_cursor

        fake = FakeConnection()
        PostgresConnection(fake).executemany("INSERT INTO sample VALUES (?)", [(1,), (2,)])

        self.assertEqual(
            fake.batch_cursor.calls,
            [("INSERT INTO sample VALUES (%s)", [(1,), (2,)])],
        )

    def test_command_sql_with_percent_does_not_bind_an_empty_parameter_sequence(self):
        class FakeConnection:
            def __init__(self):
                self.calls = []

            def execute(self, *args):
                self.calls.append(args)

        fake = FakeConnection()
        PostgresConnection(fake).execute("SELECT '100%'")
        self.assertEqual(fake.calls, [("SELECT '100%'",)])

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
