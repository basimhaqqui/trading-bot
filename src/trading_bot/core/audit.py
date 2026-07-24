from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.core.serialization import (
    canonical_json,
    parse_datetime,
    require_aware,
    sha256_digest,
    utc_now,
)
from trading_bot.execution.control import ExecutionReceipt
from trading_bot.execution.schemas import (
    ApprovedOrderIntent,
    ExecutionEnvironment,
    OrderIntent,
    RiskDecision,
)
from trading_bot.evaluation.reporting import EdgeStatus, EvaluationDecision
from trading_bot.evaluation.scoring import ForecastScore, ScoreKind


class AuditRecordType(StrEnum):
    FORECAST = "forecast"
    ORDER_INTENT = "order_intent"
    RISK_DECISION = "risk_decision"
    APPROVAL = "approval"
    EXECUTION_RECEIPT = "execution_receipt"
    FORECAST_SCORE = "forecast_score"
    EVALUATION_DECISION = "evaluation_decision"


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS audit_records (
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (record_type, record_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_timeline
ON audit_records(occurred_at, record_type, record_id);

CREATE TRIGGER IF NOT EXISTS audit_records_no_update
BEFORE UPDATE ON audit_records BEGIN
    SELECT RAISE(ABORT, 'audit_records is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_records_no_delete
BEFORE DELETE ON audit_records BEGIN
    SELECT RAISE(ABORT, 'audit_records is append-only');
END;
"""


class AuditConflictError(RuntimeError):
    pass


class AuditIntegrityError(RuntimeError):
    pass


class AuditLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def append_forecast(self, forecast: Forecast) -> bool:
        return self._append(
            AuditRecordType.FORECAST,
            forecast.forecast_id,
            forecast.generated_at,
            forecast,
        )

    def append_order_intent(self, intent: OrderIntent) -> bool:
        return self._append(
            AuditRecordType.ORDER_INTENT,
            intent.intent_id,
            intent.created_at,
            intent,
        )

    def append_risk_decision(self, decision: RiskDecision) -> bool:
        record_id = f"{decision.intent_id}:{decision.evaluated_at.isoformat()}"
        return self._append(
            AuditRecordType.RISK_DECISION,
            record_id,
            decision.evaluated_at,
            decision,
        )

    def append_approval(self, approval: ApprovedOrderIntent) -> bool:
        record_id = f"{approval.intent.intent_id}:{approval.signed_at.isoformat()}"
        return self._append(
            AuditRecordType.APPROVAL,
            record_id,
            approval.signed_at,
            approval,
        )

    def append_execution_receipt(self, receipt: ExecutionReceipt) -> bool:
        record_id = f"{receipt.intent_id}:{receipt.executed_at.isoformat()}"
        return self._append(
            AuditRecordType.EXECUTION_RECEIPT,
            record_id,
            receipt.executed_at,
            receipt,
        )

    def append_forecast_score(self, score: ForecastScore) -> bool:
        return self._append(
            AuditRecordType.FORECAST_SCORE,
            score.score_id,
            score.scored_at,
            score,
        )

    def append_evaluation_decision(self, decision: EvaluationDecision) -> bool:
        if self.has_evaluation_decision(decision.decision_id):
            return False
        return self._append(
            AuditRecordType.EVALUATION_DECISION,
            decision.decision_id,
            decision.decided_at,
            decision,
        )

    def has_evaluation_decision(self, decision_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM audit_records WHERE record_type = ? AND record_id = ?",
                (AuditRecordType.EVALUATION_DECISION.value, decision_id),
            ).fetchone()
        return row is not None

    def evaluation_decisions(self) -> tuple[EvaluationDecision, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM audit_records
                WHERE record_type = ?
                ORDER BY occurred_at, record_id
                """,
                (AuditRecordType.EVALUATION_DECISION.value,),
            ).fetchall()
        return tuple(
            self._evaluation_decision_from_payload(json.loads(row["payload_json"]))
            for row in rows
        )

    def _append(
        self,
        record_type: AuditRecordType,
        record_id: str,
        occurred_at: datetime,
        payload: object,
    ) -> bool:
        if not record_id:
            raise ValueError("record_id is required")
        occurred_at = require_aware(occurred_at, "occurred_at")
        payload_json = canonical_json(payload)
        digest = sha256_digest(json.loads(payload_json))
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT digest FROM audit_records WHERE record_type = ? AND record_id = ?",
                (record_type.value, record_id),
            ).fetchone()
            if existing:
                if existing["digest"] != digest:
                    raise AuditConflictError(
                        f"{record_type.value} {record_id} already exists with different contents"
                    )
                return False
            connection.execute(
                """
                INSERT INTO audit_records (
                    record_type, record_id, occurred_at, recorded_at, payload_json, digest
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record_type.value,
                    record_id,
                    occurred_at.isoformat(),
                    utc_now().isoformat(),
                    payload_json,
                    digest,
                ),
            )
        return True

    def counts(self) -> dict[AuditRecordType, int]:
        result = {record_type: 0 for record_type in AuditRecordType}
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT record_type, COUNT(*) AS count FROM audit_records GROUP BY record_type"
            ).fetchall()
        for row in rows:
            result[AuditRecordType(row["record_type"])] = row["count"]
        return result

    def forecasts(self) -> tuple[Forecast, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM audit_records
                WHERE record_type = ?
                ORDER BY occurred_at, record_id
                """,
                (AuditRecordType.FORECAST.value,),
            ).fetchall()
        return tuple(self._forecast_from_payload(json.loads(row["payload_json"])) for row in rows)

    def scored_forecast_ids(self) -> frozenset[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM audit_records WHERE record_type = ?",
                (AuditRecordType.FORECAST_SCORE.value,),
            ).fetchall()
        return frozenset(
            str(payload["forecast_id"])
            for row in rows
            if (payload := json.loads(row["payload_json"])).get("forecast_id")
        )

    def forecast_scores(self) -> tuple[ForecastScore, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM audit_records
                WHERE record_type = ?
                ORDER BY occurred_at, record_id
                """,
                (AuditRecordType.FORECAST_SCORE.value,),
            ).fetchall()
        return tuple(
            self._forecast_score_from_payload(json.loads(row["payload_json"]))
            for row in rows
        )

    def execution_receipts(self) -> tuple[ExecutionReceipt, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM audit_records
                WHERE record_type = ?
                ORDER BY occurred_at, record_id
                """,
                (AuditRecordType.EXECUTION_RECEIPT.value,),
            ).fetchall()
        return tuple(
            self._execution_receipt_from_payload(json.loads(row["payload_json"]))
            for row in rows
        )

    @staticmethod
    def _forecast_from_payload(payload: dict[str, object]) -> Forecast:
        return Forecast(
            forecast_id=str(payload["forecast_id"]),
            specialist_id=str(payload["specialist_id"]),
            model_version=str(payload["model_version"]),
            instrument_id=str(payload["instrument_id"]),
            kind=ForecastKind(str(payload["kind"])),
            generated_at=parse_datetime(str(payload["generated_at"])),
            valid_until=parse_datetime(str(payload["valid_until"])),
            values=dict(payload["values"]),
            confidence=float(payload["confidence"]),
            uncertainty={
                str(key): float(value)
                for key, value in dict(payload["uncertainty"]).items()
            },
            evidence_event_ids=tuple(str(item) for item in payload["evidence_event_ids"]),
            invalidation_conditions=tuple(
                str(item) for item in payload["invalidation_conditions"]
            ),
        )

    @staticmethod
    def _forecast_score_from_payload(payload: dict[str, object]) -> ForecastScore:
        return ForecastScore(
            score_id=str(payload["score_id"]),
            forecast_id=str(payload["forecast_id"]),
            specialist_id=str(payload["specialist_id"]),
            kind=ScoreKind(str(payload["kind"])),
            scored_at=parse_datetime(str(payload["scored_at"])),
            target_time=parse_datetime(str(payload["target_time"])),
            predicted=float(payload["predicted"]),
            actual=float(payload["actual"]),
            benchmark=float(payload["benchmark"]),
            loss=float(payload["loss"]),
            benchmark_loss=float(payload["benchmark_loss"]),
            metrics={
                str(key): float(value)
                for key, value in dict(payload["metrics"]).items()
            },
        )

    @staticmethod
    def _evaluation_decision_from_payload(
        payload: dict[str, object],
    ) -> EvaluationDecision:
        def optional_float(key: str) -> float | None:
            value = payload.get(key)
            return None if value is None else float(value)

        return EvaluationDecision(
            specialist_id=str(payload["specialist_id"]),
            kind=ScoreKind(str(payload["kind"])),
            scope=str(payload["scope"]),
            boundary=int(payload["boundary"]),
            status=EdgeStatus(str(payload["status"])),
            independent_outcomes=int(payload["independent_outcomes"]),
            unique_instruments=int(payload["unique_instruments"]),
            mean_improvement=optional_float("mean_improvement"),
            lower_confidence_bound=optional_float("lower_confidence_bound"),
            win_rate=optional_float("win_rate"),
            reasons=tuple(str(item) for item in payload["reasons"]),
            decided_at=parse_datetime(str(payload["decided_at"])),
        )

    @staticmethod
    def _execution_receipt_from_payload(payload: dict[str, object]) -> ExecutionReceipt:
        average = payload.get("average_fill_price")
        return ExecutionReceipt(
            intent_id=str(payload["intent_id"]),
            environment=ExecutionEnvironment(str(payload["environment"])),
            status=str(payload["status"]),
            executed_at=parse_datetime(str(payload["executed_at"])),
            venue_order_id=(
                str(payload["venue_order_id"]) if payload.get("venue_order_id") else None
            ),
            client_order_id=(
                str(payload["client_order_id"]) if payload.get("client_order_id") else None
            ),
            filled_quantity=float(payload.get("filled_quantity", 0.0)),
            average_fill_price=float(average) if average is not None else None,
        )

    def verify_integrity(self) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT record_type, record_id, payload_json, digest FROM audit_records"
            ).fetchall()
        for row in rows:
            expected = sha256_digest(json.loads(row["payload_json"]))
            if expected != row["digest"]:
                raise AuditIntegrityError(
                    f"digest mismatch for {row['record_type']} {row['record_id']}"
                )
        return len(rows)
