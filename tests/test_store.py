import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.store import EventConflictError, PointInTimeStore
from trading_bot.data.schemas import CollectionBatch


class PointInTimeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PointInTimeStore(Path(self.temp.name) / "test.db")
        self.store.initialize()
        self.instrument = Instrument(
            "demo:BTC-USD", "demo", "BTC-USD", AssetClass.CRYPTO, "USD"
        )
        self.store.register_instrument(self.instrument)
        self.event_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.available_at = self.event_time + timedelta(minutes=5)
        self.event = MarketEvent(
            "event-1",
            MarketEventType.TRADE,
            "demo",
            self.instrument.instrument_id,
            self.event_time,
            self.available_at,
            "test",
            {"price": 100.0},
            sequence=1,
            ingested_at=self.available_at,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_instrument_type_availability_index_is_initialized(self):
        with self.store.connect() as connection:
            indexes = {
                row[1]
                for row in connection.execute("PRAGMA index_list('market_events')").fetchall()
            }
        self.assertIn("idx_events_instrument_type_available", indexes)

    def test_event_is_invisible_before_available_at(self):
        self.store.append_event(self.event)
        before = self.store.events_available_at(self.available_at - timedelta(microseconds=1))
        after = self.store.events_available_at(self.available_at)
        self.assertEqual(before, [])
        self.assertEqual([item.event_id for item in after], ["event-1"])

    def test_events_can_be_read_for_a_bounded_instrument_set(self):
        other = Instrument(
            "demo:ETH-USD", "demo", "ETH-USD", AssetClass.CRYPTO, "USD"
        )
        self.store.register_instrument(other)
        self.store.append_event(self.event)
        other_event = MarketEvent(
            "event-2",
            MarketEventType.TRADE,
            "demo",
            other.instrument_id,
            self.event_time,
            self.available_at,
            "test",
            {"price": 200.0},
            sequence=1,
            ingested_at=self.available_at,
        )
        self.store.append_event(other_event)

        selected = self.store.events_available_at(
            self.available_at, instrument_ids=(other.instrument_id,)
        )

        self.assertEqual([item.event_id for item in selected], ["event-2"])
        self.assertEqual(
            self.store.events_available_at(self.available_at, instrument_ids=()), []
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.store.events_available_at(
                self.available_at,
                instrument_id=self.instrument.instrument_id,
                instrument_ids=(other.instrument_id,),
            )

    def test_identical_event_is_idempotent(self):
        self.assertTrue(self.store.append_event(self.event))
        self.assertFalse(self.store.append_event(self.event))

    def test_repeat_receipt_keeps_first_availability(self):
        self.store.append_event(self.event)
        later_receipt = MarketEvent(
            **{
                **self.event.__dict__,
                "available_at": self.available_at + timedelta(minutes=1),
                "ingested_at": self.available_at + timedelta(minutes=1),
            }
        )
        self.assertFalse(self.store.append_event(later_receipt))
        stored = self.store.event(self.event.event_id)
        self.assertEqual(stored.available_at, self.available_at)

    def test_conflicting_event_id_is_rejected(self):
        self.store.append_event(self.event)
        changed = MarketEvent(
            **{**self.event.__dict__, "payload": {"price": 101.0}}
        )
        with self.assertRaises(EventConflictError):
            self.store.append_event(changed)

    def test_event_table_is_append_only(self):
        self.store.append_event(self.event)
        with self.assertRaises(Exception):
            with self.store.connect() as connection:
                connection.execute(
                    "UPDATE market_events SET source = 'changed' WHERE event_id = 'event-1'"
                )

    def test_batch_ingestion_is_atomic_and_idempotent(self):
        atomic_store = PointInTimeStore(Path(self.temp.name) / "atomic.db")
        atomic_store.initialize()
        conflict = MarketEvent(**{**self.event.__dict__, "payload": {"price": 101.0}})
        with self.assertRaises(EventConflictError):
            atomic_store.append_batch(
                CollectionBatch("demo", (self.instrument,), (self.event, conflict))
            )
        with self.assertRaises(KeyError):
            atomic_store.instrument(self.instrument.instrument_id)

        batch = CollectionBatch("demo", (self.instrument,), (self.event,))
        self.assertEqual(atomic_store.append_batch(batch), (1, 1))
        self.assertEqual(atomic_store.append_batch(batch), (1, 0))

    def test_batch_preserves_first_receipt_for_duplicate_event_ids(self):
        atomic_store = PointInTimeStore(Path(self.temp.name) / "duplicate.db")
        atomic_store.initialize()
        later_receipt = MarketEvent(
            **{
                **self.event.__dict__,
                "available_at": self.available_at + timedelta(minutes=1),
                "ingested_at": self.available_at + timedelta(minutes=1),
            }
        )

        self.assertEqual(
            atomic_store.append_batch(
                CollectionBatch("demo", (self.instrument,), (self.event, later_receipt))
            ),
            (1, 1),
        )
        self.assertEqual(
            atomic_store.event(self.event.event_id).available_at, self.available_at
        )


if __name__ == "__main__":
    unittest.main()
