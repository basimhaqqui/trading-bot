import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.core.store import PointInTimeStore
from trading_bot.ingestion.health import ingestion_health, render_health
from trading_bot.ingestion.plan import ObservationJob, ShadowIngestionPlan
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
    IngestionRunRecord,
    IngestionRunStatus,
)


class IngestionHealthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "health.db"
        PointInTimeStore(self.path).initialize()
        self.ledger = IngestionRunLedger(self.path)
        self.ledger.initialize()
        self.now = datetime(2026, 7, 22, 6, tzinfo=timezone.utc)
        self.plan = ShadowIngestionPlan(
            "health-plan",
            (
                ObservationJob("markets", "kalshi", "markets"),
                ObservationJob("products", "coinbase", "products"),
            ),
        )

    def tearDown(self):
        self.temp.cleanup()

    def append_run(
        self,
        job_id: str,
        status: IngestionRunStatus,
        finished_at: datetime,
        *,
        next_cursor: str | None = None,
    ) -> None:
        venue = "kalshi" if job_id == "markets" else "coinbase"
        dataset = "markets" if job_id == "markets" else "products"
        self.ledger.append(
            IngestionRunRecord(
                run_id=f"{job_id}-{finished_at.isoformat()}-{status.value}",
                plan_name=self.plan.name,
                job_id=job_id,
                venue=venue,
                dataset=dataset,
                status=status,
                started_at=finished_at - timedelta(seconds=2),
                finished_at=finished_at,
                instruments_seen=0,
                events_inserted=0,
                next_cursor=next_cursor,
                error_type="TimeoutError" if status is IngestionRunStatus.FAILED else None,
                error_message="fixture timeout" if status is IngestionRunStatus.FAILED else None,
            )
        )

    def test_missing_or_stale_jobs_fail_health_gate(self):
        self.append_run(
            "markets", IngestionRunStatus.SUCCESS, self.now - timedelta(hours=2)
        )
        report = ingestion_health(
            self.path,
            self.plan,
            as_of=self.now,
            max_age=timedelta(minutes=90),
        )
        self.assertFalse(report.healthy)
        self.assertIn("120.0 minutes old", report.jobs[0].reasons[0])
        self.assertEqual(report.jobs[1].status, "missing")

    def test_recent_jobs_pass_and_render_machine_readable_health(self):
        self.append_run(
            "markets",
            IngestionRunStatus.SUCCESS,
            self.now - timedelta(minutes=10),
            next_cursor="page-2",
        )
        self.append_run(
            "products", IngestionRunStatus.DEGRADED, self.now - timedelta(minutes=5)
        )
        report = ingestion_health(self.path, self.plan, as_of=self.now)
        self.assertTrue(report.healthy)
        self.assertTrue(report.jobs[0].has_next_cursor)
        payload = json.loads(render_health(report, "json"))
        self.assertTrue(payload["healthy"])
        self.assertEqual(payload["jobs"][1]["status"], "degraded")
        self.assertIn("| `markets` |", render_health(report, "markdown"))

    def test_consecutive_failures_are_counted_until_last_success(self):
        self.append_run(
            "markets", IngestionRunStatus.SUCCESS, self.now - timedelta(minutes=20)
        )
        self.append_run(
            "markets", IngestionRunStatus.FAILED, self.now - timedelta(minutes=10)
        )
        self.append_run(
            "markets", IngestionRunStatus.FAILED, self.now - timedelta(minutes=5)
        )
        self.append_run(
            "products", IngestionRunStatus.SUCCESS, self.now - timedelta(minutes=5)
        )
        report = ingestion_health(self.path, self.plan, as_of=self.now)
        self.assertFalse(report.healthy)
        self.assertEqual(report.jobs[0].consecutive_failures, 2)
        self.assertIn("2 consecutive failure(s)", report.jobs[0].reasons[0])


if __name__ == "__main__":
    unittest.main()
