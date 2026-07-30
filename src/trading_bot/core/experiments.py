from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from trading_bot.core.database import DatabaseLocation, connect_database, initialize_schema
from trading_bot.core.schemas import Hypothesis
from trading_bot.core.serialization import canonical_json, parse_datetime, require_aware, utc_now


class ExperimentStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    REJECTED = "rejected"
    SHADOW = "shadow"
    COMPLETED = "completed"


TERMINAL_STATUSES = {ExperimentStatus.REJECTED, ExperimentStatus.SHADOW, ExperimentStatus.COMPLETED}


class ExperimentConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    hypothesis_id: str
    family: str
    status: ExperimentStatus
    config: Mapping[str, Any]
    code_version: str
    data_cutoff: datetime
    created_at: datetime
    completed_at: datetime | None
    metrics: Mapping[str, Any]
    notes: str


EXPERIMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    market TEXT NOT NULL,
    proposed_at TEXT NOT NULL,
    record_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    family TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    code_version TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    metrics_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_family ON experiments(family, created_at);
"""


class ExperimentRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path: DatabaseLocation = path

    def initialize(self) -> None:
        with connect_database(self.path) as connection:
            initialize_schema(connection, EXPERIMENT_SCHEMA)

    def register_hypothesis(self, hypothesis: Hypothesis) -> None:
        with connect_database(self.path) as connection:
            existing = connection.execute(
                "SELECT record_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis.hypothesis_id,),
            ).fetchone()
            record_json = canonical_json(hypothesis)
            if existing and existing[0] != record_json:
                raise ExperimentConflictError(
                    f"hypothesis {hypothesis.hypothesis_id} already has different contents"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO hypotheses
                (hypothesis_id, family, market, proposed_at, record_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    hypothesis.hypothesis_id,
                    hypothesis.family,
                    hypothesis.market.value,
                    hypothesis.proposed_at.isoformat(),
                    record_json,
                ),
            )

    def start(
        self,
        hypothesis: Hypothesis,
        *,
        config: Mapping[str, Any],
        code_version: str,
        data_cutoff: datetime,
    ) -> Experiment:
        self.register_hypothesis(hypothesis)
        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.hypothesis_id,
            family=hypothesis.family,
            status=ExperimentStatus.RUNNING,
            config=dict(config),
            code_version=code_version,
            data_cutoff=require_aware(data_cutoff, "data_cutoff"),
            created_at=utc_now(),
            completed_at=None,
            metrics={},
            notes="",
        )
        with connect_database(self.path) as connection:
            connection.execute(
                """
                INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.hypothesis_id,
                    experiment.family,
                    experiment.status.value,
                    canonical_json(experiment.config),
                    experiment.code_version,
                    experiment.data_cutoff.isoformat(),
                    experiment.created_at.isoformat(),
                    None,
                    "{}",
                    "",
                ),
            )
        return experiment

    def finish(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus,
        metrics: Mapping[str, Any],
        notes: str = "",
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("finished experiment needs a terminal status")
        with connect_database(self.path) as connection:
            row = connection.execute(
                "SELECT status FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(experiment_id)
            if row[0] != ExperimentStatus.RUNNING.value:
                raise ValueError(f"experiment is {row[0]}, not running")
            connection.execute(
                """
                UPDATE experiments
                SET status = ?, completed_at = ?, metrics_json = ?, notes = ?
                WHERE experiment_id = ?
                """,
                (status.value, utc_now().isoformat(), canonical_json(metrics), notes, experiment_id),
            )

    def trial_count(self, family: str) -> int:
        with connect_database(self.path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE family = ?", (family,)
            ).fetchone()
        return int(row[0])

    def get(self, experiment_id: str) -> Experiment:
        with connect_database(self.path) as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return Experiment(
            experiment_id=row["experiment_id"],
            hypothesis_id=row["hypothesis_id"],
            family=row["family"],
            status=ExperimentStatus(row["status"]),
            config=json.loads(row["config_json"]),
            code_version=row["code_version"],
            data_cutoff=parse_datetime(row["data_cutoff"]),
            created_at=parse_datetime(row["created_at"]),
            completed_at=parse_datetime(row["completed_at"]) if row["completed_at"] else None,
            metrics=json.loads(row["metrics_json"]),
            notes=row["notes"],
        )
