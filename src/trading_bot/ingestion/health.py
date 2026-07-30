from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from trading_bot.core.database import connect_database
from trading_bot.core.serialization import canonical_json, parse_datetime, require_aware
from trading_bot.ingestion.plan import ShadowIngestionPlan
from trading_bot.ingestion.runner import IngestionRunStatus


@dataclass(frozen=True)
class JobHealth:
    job_id: str
    venue: str
    dataset: str
    healthy: bool
    status: str
    finished_at: datetime | None
    age_minutes: float | None
    consecutive_failures: int
    diagnostics: int
    has_next_cursor: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class IngestionHealthReport:
    plan_name: str
    checked_at: datetime
    healthy: bool
    max_age_minutes: float
    max_consecutive_failures: int
    jobs: tuple[JobHealth, ...]


def ingestion_health(
    path: str | Path,
    plan: ShadowIngestionPlan,
    *,
    as_of: datetime,
    max_age: timedelta = timedelta(minutes=90),
    max_consecutive_failures: int = 0,
    environment: Mapping[str, str] | None = None,
) -> IngestionHealthReport:
    as_of = require_aware(as_of, "as_of")
    if max_age <= timedelta(0):
        raise ValueError("max_age must be positive")
    if max_consecutive_failures < 0:
        raise ValueError("max_consecutive_failures cannot be negative")

    jobs: list[JobHealth] = []
    with connect_database(path) as connection:
        for job in plan.jobs:
            if not job.enabled:
                continue
            missing_environment = job.missing_activation_environment(environment)
            if missing_environment:
                jobs.append(
                    JobHealth(
                        job.job_id,
                        job.venue,
                        job.dataset,
                        True,
                        "waiting_credentials",
                        None,
                        None,
                        0,
                        0,
                        False,
                        (
                            f"activation profile {job.activation_profile} is waiting for "
                            f"{', '.join(missing_environment)}",
                        ),
                    )
                )
                continue
            rows = connection.execute(
                """
                SELECT status, finished_at, record_json
                FROM ingestion_runs
                WHERE plan_name = ? AND job_id = ?
                ORDER BY finished_at DESC, run_id DESC
                """,
                (plan.name, job.job_id),
            ).fetchall()
            if not rows:
                jobs.append(
                    JobHealth(
                        job.job_id,
                        job.venue,
                        job.dataset,
                        False,
                        "missing",
                        None,
                        None,
                        0,
                        0,
                        False,
                        ("no completed ingestion run",),
                    )
                )
                continue

            latest = rows[0]
            finished_at = parse_datetime(latest["finished_at"])
            age_minutes = (as_of - finished_at).total_seconds() / 60
            failures = 0
            for row in rows:
                if row["status"] != IngestionRunStatus.FAILED.value:
                    break
                failures += 1
            payload = json.loads(latest["record_json"])
            diagnostics = payload.get("diagnostics", [])
            diagnostic_count = len(diagnostics) if isinstance(diagnostics, list) else 0
            reasons: list[str] = []
            if age_minutes < -5:
                reasons.append("latest run timestamp is in the future")
            elif age_minutes > max_age.total_seconds() / 60:
                reasons.append(f"latest run is {age_minutes:.1f} minutes old")
            if failures > max_consecutive_failures:
                reasons.append(
                    f"{failures} consecutive failure(s) exceeds allowed "
                    f"{max_consecutive_failures}"
                )
            status = str(latest["status"])
            if status not in {item.value for item in IngestionRunStatus}:
                reasons.append(f"unknown ingestion status: {status}")
            jobs.append(
                JobHealth(
                    job.job_id,
                    job.venue,
                    job.dataset,
                    not reasons,
                    status,
                    finished_at,
                    age_minutes,
                    failures,
                    diagnostic_count,
                    payload.get("next_cursor") is not None,
                    tuple(reasons),
                )
            )

    return IngestionHealthReport(
        plan.name,
        as_of,
        all(job.healthy for job in jobs),
        max_age.total_seconds() / 60,
        max_consecutive_failures,
        tuple(jobs),
    )


def render_health(report: IngestionHealthReport, output_format: str = "text") -> str:
    if output_format == "json":
        return canonical_json(report)
    if output_format == "markdown":
        state = "healthy" if report.healthy else "unhealthy"
        lines = [
            "## Shadow ingestion health",
            "",
            f"**{state}** — plan `{report.plan_name}` checked at "
            f"`{report.checked_at.isoformat()}`",
            "",
            "| Job | Feed | Status | Age | Failures | Diagnostics | Cursor |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for job in report.jobs:
            age = "n/a" if job.age_minutes is None else f"{job.age_minutes:.1f}m"
            status = job.status if not job.reasons else f"{job.status}: {'; '.join(job.reasons)}"
            status = status.replace("|", "\\|")
            lines.append(
                f"| `{job.job_id}` | {job.venue}/{job.dataset} | {status} | {age} | "
                f"{job.consecutive_failures} | {job.diagnostics} | "
                f"{'more pages' if job.has_next_cursor else 'terminal'} |"
            )
        return "\n".join(lines)
    if output_format != "text":
        raise ValueError("output_format must be text, json, or markdown")
    state = "healthy" if report.healthy else "unhealthy"
    lines = [
        f"shadow-health: {state} plan={report.plan_name} "
        f"checked_at={report.checked_at.isoformat()}"
    ]
    for job in report.jobs:
        age = "n/a" if job.age_minutes is None else f"{job.age_minutes:.1f}m"
        reason = "" if not job.reasons else f" reasons={'; '.join(job.reasons)}"
        lines.append(
            f"{job.job_id}: {job.status} age={age} "
            f"failures={job.consecutive_failures} diagnostics={job.diagnostics}"
            f"{reason}"
        )
    return "\n".join(lines)
