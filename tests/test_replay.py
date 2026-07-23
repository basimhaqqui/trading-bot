import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from trading_bot.agents.base import ReplayContext
from trading_bot.agents.demo import DemoRegimeSpecialist
from trading_bot.core.schemas import (
    AssetClass,
    Forecast,
    ForecastKind,
    Instrument,
    MarketEvent,
    MarketEventType,
)
from trading_bot.core.store import PointInTimeStore
from trading_bot.replay import ReplayEngine, ReplayIntegrityError


class CheatingSpecialist:
    agent_id = "cheater"
    supported_asset_classes = frozenset({AssetClass.EQUITY})

    def evaluate(self, context: ReplayContext):
        return Forecast(
            "cheat-forecast",
            self.agent_id,
            "v1",
            context.instrument.instrument_id,
            ForecastKind.REGIME,
            context.decision_time,
            context.decision_time + timedelta(days=1),
            {"regime": "up"},
            1.0,
            {"unknown": 0.0},
            ("future-event",),
            ("none",),
        )


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PointInTimeStore(Path(self.temp.name) / "test.db")
        self.store.initialize()
        self.instrument = Instrument(
            "demo:SPY", "demo", "SPY", AssetClass.EQUITY, "USD"
        )
        self.store.register_instrument(self.instrument)
        self.base = datetime(2026, 1, 1, 20, tzinfo=timezone.utc)
        for index, close in enumerate((100.0, 102.0), start=1):
            event_time = self.base + timedelta(days=index - 1)
            available_at = event_time + timedelta(minutes=1)
            self.store.append_event(
                MarketEvent(
                    f"bar-{index}",
                    MarketEventType.BAR,
                    "demo",
                    self.instrument.instrument_id,
                    event_time,
                    available_at,
                    "test",
                    {"close": close},
                    index,
                    available_at,
                )
            )

    def tearDown(self):
        self.temp.cleanup()

    def test_replay_uses_only_available_events(self):
        result = ReplayEngine(self.store).run(
            DemoRegimeSpecialist(),
            instrument_id=self.instrument.instrument_id,
            decision_times=(self.base + timedelta(days=1), self.base + timedelta(days=2)),
        )
        self.assertEqual(len(result.forecasts), 1)
        self.assertEqual(result.forecasts[0].evidence_event_ids, ("bar-1", "bar-2"))

    def test_unavailable_evidence_is_rejected(self):
        with self.assertRaises(ReplayIntegrityError):
            ReplayEngine(self.store).run(
                CheatingSpecialist(),
                instrument_id=self.instrument.instrument_id,
                decision_times=(self.base + timedelta(days=1),),
            )

    def test_related_instruments_are_loaded_in_one_point_in_time_read(self):
        related = Instrument(
            "demo:QQQ", "demo", "QQQ", AssetClass.EQUITY, "USD"
        )
        self.store.register_instrument(related)
        with patch.object(
            self.store,
            "events_available_at",
            wraps=self.store.events_available_at,
        ) as events_available_at:
            ReplayEngine(self.store).run(
                DemoRegimeSpecialist(),
                instrument_id=self.instrument.instrument_id,
                related_instrument_ids=(related.instrument_id,),
                decision_times=(self.base + timedelta(days=2),),
            )

        self.assertEqual(events_available_at.call_count, 1)


if __name__ == "__main__":
    unittest.main()
