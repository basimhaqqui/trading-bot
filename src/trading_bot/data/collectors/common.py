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


_BASE58_CHARACTERS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_ALPHABET = frozenset(_BASE58_CHARACTERS)
_BASE58_VALUE = {character: index for index, character in enumerate(_BASE58_CHARACTERS)}


def is_valid_solana_public_key(value: object) -> bool:
    """Return whether ``value`` is exactly one 32-byte base58 public key.

    Discovery metadata is untrusted. Checking the encoded byte length locally
    keeps malformed identifiers from becoming instruments or consuming the
    bounded read-only RPC follow-up budget.
    """
    if (
        not isinstance(value, str)
        or not 32 <= len(value) <= 44
        or any(character not in _BASE58_ALPHABET for character in value)
    ):
        return False
    decoded = 0
    for character in value:
        decoded = decoded * 58 + _BASE58_VALUE[character]
    encoded_bytes = (
        decoded.to_bytes((decoded.bit_length() + 7) // 8, "big") if decoded else b""
    )
    leading_zero_bytes = len(value) - len(value.lstrip("1"))
    return leading_zero_bytes + len(encoded_bytes) == 32


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
