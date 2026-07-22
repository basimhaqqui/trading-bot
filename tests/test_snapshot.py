import hashlib
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.core.audit import AuditLedger
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.snapshot import create_verified_snapshot
from trading_bot.core.store import PointInTimeStore
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
    IngestionRunRecord,
    IngestionRunStatus,
)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.db"
        self.output = self.root / "backups" / "snapshot.db"
        self.store = PointInTimeStore(self.source)
        self.store.initialize()
        AuditLedger(self.source).initialize()
        self.ledger = IngestionRunLedger(self.source)
        self.ledger.initialize()
        self.now = datetime(2026, 7, 22, 6, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_snapshot_is_atomic_complete_and_digest_verified(self):
        instrument = Instrument(
            "coinbase:product:BTC-USD",
            "coinbase",
            "BTC-USD",
            AssetClass.CRYPTO,
            "USD",
        )
        self.store.register_instrument(instrument)
        self.store.append_event(
            MarketEvent(
                "snapshot-book",
                MarketEventType.BOOK_SNAPSHOT,
                "coinbase",
                instrument.instrument_id,
                self.now,
                self.now,
                "fixture",
                {"bid_price": 100, "ask_price": 101},
                ingested_at=self.now,
            )
        )
        self.ledger.append(
            IngestionRunRecord(
                "snapshot-run",
                "snapshot-plan",
                "book",
                "coinbase",
                "book",
                IngestionRunStatus.SUCCESS,
                self.now,
                self.now,
                1,
                1,
            )
        )

        summary = create_verified_snapshot(self.source, self.output)

        self.assertEqual(summary.events, 1)
        self.assertEqual(summary.audit_records, 0)
        self.assertEqual(summary.ingestion_runs, 1)
        self.assertEqual(summary.bytes_written, self.output.stat().st_size)
        self.assertEqual(
            summary.sha256, hashlib.sha256(self.output.read_bytes()).hexdigest()
        )
        with sqlite3.connect(self.output) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM market_events").fetchone()[0], 1)

    def test_snapshot_refuses_to_overwrite_source_database(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            create_verified_snapshot(self.source, self.source)


if __name__ == "__main__":
    unittest.main()
