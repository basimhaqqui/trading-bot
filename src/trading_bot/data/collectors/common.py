from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from trading_bot.core.serialization import parse_datetime, require_aware, sha256_digest
from trading_bot.data.schemas import (
    DataQualityDiagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
)


class CollectorPayloadError(ValueError):
    pass


def require_object(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CollectorPayloadError(f"{field_name} must be an object")
    return value


def require_list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CollectorPayloadError(f"{field_name} must be a list")
    return value


def require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CollectorPayloadError(f"{field_name} must be a non-empty string")
    return value


def parse_time(value: object, field_name: str) -> datetime:
    return parse_datetime(require_string(value, field_name))


def observed_event_time(
    source_time: datetime,
    collected_at: datetime,
    *,
    instrument_id: str,
    diagnostics: list[DataQualityDiagnostic],
) -> datetime:
    source_time = require_aware(source_time, "source_time")
    collected_at = require_aware(collected_at, "collected_at")
    if source_time <= collected_at:
        return source_time
    diagnostics.append(
        DataQualityDiagnostic(
            code=DiagnosticCode.SOURCE_CLOCK_AHEAD,
            severity=DiagnosticSeverity.WARNING,
            message=(
                f"source timestamp {source_time.isoformat()} was after collection time; "
                "event time was clamped to the observation boundary"
            ),
            instrument_id=instrument_id,
        )
    )
    return collected_at


def stable_event_id(prefix: str, identity: object) -> str:
    return f"{prefix}:{sha256_digest(identity)[:24]}"
