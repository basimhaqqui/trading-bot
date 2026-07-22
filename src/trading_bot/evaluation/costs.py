from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from trading_bot.evaluation.scoring import ScoreKind


class CostBasis(StrEnum):
    STATIC_BPS = "static_bps"
    FORECAST_EXECUTION_BOUND = "forecast_execution_bound"
    BINARY_CONTRACT = "binary_contract"


@dataclass(frozen=True)
class EconomicCostModel:
    model_id: str
    specialist_id: str
    kind: ScoreKind
    basis: CostBasis
    source_url: str
    effective_date: date
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    latency_bps: float = 0.0
    binary_fee_coefficient: float = 0.0
    binary_fee_increment: float = 0.01

    def __post_init__(self) -> None:
        if not self.model_id or not self.specialist_id:
            raise ValueError("cost model identity is required")
        parsed = urlsplit(self.source_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("cost model source must be an absolute HTTPS URL")
        bps_values = (
            self.fee_bps,
            self.spread_bps,
            self.slippage_bps,
            self.latency_bps,
        )
        if any(value < 0 or value > 10_000 for value in bps_values):
            raise ValueError("cost assumptions must be between 0 and 10,000 bps")
        if self.basis is CostBasis.BINARY_CONTRACT:
            if self.kind is not ScoreKind.BINARY:
                raise ValueError("binary contract costs require binary forecasts")
            if self.binary_fee_coefficient <= 0 or not 0 < self.binary_fee_increment <= 1:
                raise ValueError("binary fee coefficient and increment must be positive")
        elif self.binary_fee_coefficient != 0:
            raise ValueError("binary fee coefficient is only valid for binary contract costs")
        if (
            self.basis is CostBasis.FORECAST_EXECUTION_BOUND
            and self.kind is not ScoreKind.FUNDING
        ):
            raise ValueError("forecast execution bounds require funding forecasts")


@dataclass(frozen=True)
class EconomicCostRegistry:
    version: str
    models: tuple[EconomicCostModel, ...]

    def __post_init__(self) -> None:
        if not self.version or not self.models:
            raise ValueError("cost registry version and models are required")
        keys = [(item.specialist_id, item.kind) for item in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("cost registry contains duplicate specialist/kind models")


def load_cost_registry(path: str | Path) -> EconomicCostRegistry:
    config_path = Path(path)
    if config_path.stat().st_size > 1_000_000:
        raise ValueError("economic cost config exceeds the 1 MB safety limit")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"version", "models"}:
        raise ValueError("cost config must contain only version and models")
    if not isinstance(payload["version"], str) or not isinstance(payload["models"], list):
        raise ValueError("cost config version must be text and models must be a list")
    allowed = {
        "model_id",
        "specialist_id",
        "kind",
        "basis",
        "source_url",
        "effective_date",
        "fee_bps",
        "spread_bps",
        "slippage_bps",
        "latency_bps",
        "binary_fee_coefficient",
        "binary_fee_increment",
    }
    models: list[EconomicCostModel] = []
    for raw in payload["models"]:
        if not isinstance(raw, dict):
            raise ValueError("every cost model must be an object")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown cost model fields: {sorted(unknown)}")
        values = dict(raw)
        try:
            values["kind"] = ScoreKind(values["kind"])
            values["basis"] = CostBasis(values["basis"])
            values["effective_date"] = date.fromisoformat(values["effective_date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cost model has an invalid kind, basis, or effective date") from exc
        models.append(EconomicCostModel(**values))
    return EconomicCostRegistry(payload["version"], tuple(models))
