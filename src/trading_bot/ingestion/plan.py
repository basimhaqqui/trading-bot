from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ALLOWED_DATASETS = {
    "kalshi": {"markets", "forecast_outcomes", "trades", "book"},
    "coinbase": {"products", "book", "candles"},
    "alpaca": {"chain", "bars", "quotes"},
}
SENSITIVE_FRAGMENTS = (
    "secret",
    "password",
    "credential",
    "authorization",
    "private_key",
    "api_key",
    "token",
)
ACTIVATION_PROFILES = {
    "alpaca_market_data": (
        "ALPACA_MARKET_DATA_KEY_ID",
        "ALPACA_MARKET_DATA_SECRET_KEY",
    ),
}
CURSOR_MODES = {"resume", "restart"}


@dataclass(frozen=True)
class ObservationJob:
    job_id: str
    venue: str
    dataset: str
    symbol: str | None = None
    limit: int = 100
    status: str = "open"
    feed: str = "indicative"
    product_type: str | None = None
    stock_feed: str = "iex"
    lookback_days: int = 45
    granularity: str = "ONE_HOUR"
    enabled: bool = True
    activation_profile: str | None = None
    cursor_mode: str = "resume"
    mve_filter: str | None = None
    close_lookahead_hours: int | None = None
    expiration_lookahead_days: int | None = None
    strike_band_pct: float | None = None
    updated_since_minutes: int | None = None

    def __post_init__(self) -> None:
        if not self.job_id or not self.job_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("job_id must contain only letters, numbers, hyphens, or underscores")
        if self.dataset not in ALLOWED_DATASETS.get(self.venue, set()):
            raise ValueError(f"unsupported observation job: {self.venue}/{self.dataset}")
        if not 1 <= self.limit <= 1000:
            raise ValueError("job limit must be between 1 and 1000")
        if self.dataset in {"book", "chain", "bars", "candles", "quotes"} and not self.symbol:
            raise ValueError(f"{self.dataset} jobs require a symbol")
        if self.dataset == "candles" and self.limit > 350:
            raise ValueError("Coinbase candle limit cannot exceed 350")
        if self.feed not in {"opra", "indicative"}:
            raise ValueError("feed must be opra or indicative")
        if self.venue == "kalshi" and self.status not in {
            "unopened",
            "open",
            "paused",
            "closed",
            "settled",
        }:
            raise ValueError("invalid Kalshi status")
        if self.product_type not in {None, "SPOT", "FUTURE"}:
            raise ValueError("Coinbase product_type must be SPOT or FUTURE")
        if self.product_type is not None and not (
            self.venue == "coinbase" and self.dataset == "products"
        ):
            raise ValueError("product_type is only valid for Coinbase product jobs")
        if self.dataset in {"markets", "forecast_outcomes", "products"} and self.symbol is not None:
            raise ValueError(f"{self.dataset} jobs do not accept a symbol")
        if self.stock_feed not in {"iex", "sip", "delayed_sip"}:
            raise ValueError("stock_feed must be iex, sip, or delayed_sip")
        if not 2 <= self.lookback_days <= 3660:
            raise ValueError("lookback_days must be between 2 and 3660")
        if self.granularity not in {
            "ONE_MINUTE",
            "FIVE_MINUTE",
            "FIFTEEN_MINUTE",
            "THIRTY_MINUTE",
            "ONE_HOUR",
            "TWO_HOUR",
            "FOUR_HOUR",
            "SIX_HOUR",
            "ONE_DAY",
        }:
            raise ValueError("unsupported candle granularity")
        if self.granularity != "ONE_HOUR" and not (
            self.venue == "coinbase" and self.dataset == "candles"
        ):
            raise ValueError("granularity is only valid for Coinbase candle jobs")
        if self.activation_profile not in {None, *ACTIVATION_PROFILES}:
            raise ValueError("unsupported activation profile")
        if self.activation_profile == "alpaca_market_data" and self.venue != "alpaca":
            raise ValueError("alpaca_market_data activation is only valid for Alpaca jobs")
        if self.cursor_mode not in CURSOR_MODES:
            raise ValueError("cursor_mode must be resume or restart")
        restartable = {
            ("alpaca", "chain"),
            ("kalshi", "markets"),
            ("kalshi", "forecast_outcomes"),
        }
        if self.cursor_mode == "restart" and (self.venue, self.dataset) not in restartable:
            raise ValueError("restart cursor mode is not valid for this observation job")
        if self.mve_filter not in {None, "only", "exclude"}:
            raise ValueError("mve_filter must be only or exclude")
        if self.mve_filter is not None and not (
            self.venue == "kalshi" and self.dataset == "markets"
        ):
            raise ValueError("mve_filter is only valid for Kalshi market jobs")
        if self.close_lookahead_hours is not None:
            if isinstance(self.close_lookahead_hours, bool) or not (
                1 <= self.close_lookahead_hours <= 168
            ):
                raise ValueError("close_lookahead_hours must be between 1 and 168")
            if not (self.venue == "kalshi" and self.dataset == "markets"):
                raise ValueError(
                    "close_lookahead_hours is only valid for Kalshi market jobs"
                )
            if self.status != "open":
                raise ValueError("close lookahead jobs must target open markets")
        option_filter_values = (
            self.expiration_lookahead_days,
            self.strike_band_pct,
            self.updated_since_minutes,
        )
        if any(value is not None for value in option_filter_values) and not (
            self.venue == "alpaca" and self.dataset == "chain"
        ):
            raise ValueError("option cohort filters are only valid for Alpaca chain jobs")
        if self.expiration_lookahead_days is not None and (
            isinstance(self.expiration_lookahead_days, bool)
            or not 1 <= self.expiration_lookahead_days <= 60
        ):
            raise ValueError("expiration lookahead days must be between 1 and 60")
        if self.strike_band_pct is not None and (
            isinstance(self.strike_band_pct, bool)
            or not 0 < self.strike_band_pct <= 0.5
        ):
            raise ValueError("strike band percent must be between zero and 0.5")
        if self.updated_since_minutes is not None and (
            isinstance(self.updated_since_minutes, bool)
            or not 1 <= self.updated_since_minutes <= 1440
        ):
            raise ValueError("updated-since minutes must be between 1 and 1440")
        if (
            any(value is not None for value in option_filter_values)
            and self.cursor_mode != "restart"
        ):
            raise ValueError("filtered option cohort jobs must restart pagination")

    def missing_activation_environment(
        self, environment: Mapping[str, str] | None = None
    ) -> tuple[str, ...]:
        if self.activation_profile is None:
            return ()
        values = os.environ if environment is None else environment
        return tuple(
            name
            for name in ACTIVATION_PROFILES[self.activation_profile]
            if not values.get(name, "").strip()
        )

    def is_active(self, environment: Mapping[str, str] | None = None) -> bool:
        return self.enabled and not self.missing_activation_environment(environment)


@dataclass(frozen=True)
class ShadowIngestionPlan:
    name: str
    jobs: tuple[ObservationJob, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.jobs:
            raise ValueError("plan name and at least one job are required")
        if not any(job.enabled for job in self.jobs):
            raise ValueError("plan must contain at least one enabled job")
        job_ids = [job.job_id for job in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("job IDs must be unique within a plan")


def load_plan(path: str | Path) -> ShadowIngestionPlan:
    plan_path = Path(path)
    if plan_path.stat().st_size > 1_000_000:
        raise ValueError("ingestion plan exceeds the 1 MB safety limit")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ingestion plan must be a JSON object")
    _reject_sensitive_fields(payload)
    allowed_plan_fields = {"name", "jobs"}
    unknown_plan_fields = set(payload) - allowed_plan_fields
    if unknown_plan_fields:
        raise ValueError(f"unknown plan fields: {sorted(unknown_plan_fields)}")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("plan jobs must be a list")
    allowed_job_fields = {
        "job_id",
        "venue",
        "dataset",
        "symbol",
        "limit",
        "status",
        "feed",
        "product_type",
        "stock_feed",
        "lookback_days",
        "granularity",
        "enabled",
        "activation_profile",
        "cursor_mode",
        "mve_filter",
        "close_lookahead_hours",
        "expiration_lookahead_days",
        "strike_band_pct",
        "updated_since_minutes",
    }
    jobs: list[ObservationJob] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise ValueError("every ingestion job must be an object")
        unknown = set(raw_job) - allowed_job_fields
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        jobs.append(ObservationJob(**raw_job))
    name = payload.get("name")
    if not isinstance(name, str):
        raise ValueError("plan name must be a string")
    return ShadowIngestionPlan(name, tuple(jobs))


def _reject_sensitive_fields(value: object, path: str = "plan") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in SENSITIVE_FRAGMENTS):
                raise ValueError(f"credentials are forbidden in ingestion plans: {path}.{key}")
            _reject_sensitive_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_fields(item, f"{path}[{index}]")
