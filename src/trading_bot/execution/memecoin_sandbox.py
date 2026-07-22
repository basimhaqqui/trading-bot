from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping
from unittest.mock import patch
from urllib.parse import urlsplit

from trading_bot.core.serialization import canonical_json, require_aware, sha256_digest


SANDBOX_TIME = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


@dataclass(frozen=True)
class MemecoinSandboxConfig:
    version: str = "memecoin-safety-sandbox-v1"
    min_liquidity_usd: float = 100_000
    min_pool_age_hours: float = 168
    max_top10_holder_pct: float = 50
    max_largest_holder_pct: float = 10
    max_transfer_fee_bps: float = 200
    max_price_impact_bps: float = 300
    max_oracle_deviation_bps: float = 500
    min_sell_recovery_pct: float = 90
    max_snapshot_age: timedelta = timedelta(seconds=60)
    max_shadow_position_usd: float = 500
    token_authority_source_url: str = "https://solana.com/docs/tokens/basics"
    token_risk_source_url: str = (
        "https://docs.raydium.io/security/oracle-and-token-risks"
    )

    def __post_init__(self) -> None:
        numeric = (
            self.min_liquidity_usd,
            self.min_pool_age_hours,
            self.max_top10_holder_pct,
            self.max_largest_holder_pct,
            self.max_transfer_fee_bps,
            self.max_price_impact_bps,
            self.max_oracle_deviation_bps,
            self.min_sell_recovery_pct,
            self.max_shadow_position_usd,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("memecoin policy values must be finite")
        if not self.version or self.min_liquidity_usd <= 0:
            raise ValueError("memecoin policy version and liquidity floor are required")
        if self.min_pool_age_hours <= 0 or self.max_snapshot_age <= timedelta(0):
            raise ValueError("memecoin pool and snapshot ages must be positive")
        if not 0 <= self.max_largest_holder_pct <= self.max_top10_holder_pct <= 100:
            raise ValueError("memecoin holder limits are invalid")
        if not 0 <= self.max_transfer_fee_bps <= 10_000:
            raise ValueError("memecoin transfer-fee limit is invalid")
        if not 0 <= self.max_price_impact_bps <= 10_000:
            raise ValueError("memecoin price-impact limit is invalid")
        if not 0 <= self.max_oracle_deviation_bps <= 10_000:
            raise ValueError("memecoin oracle-deviation limit is invalid")
        if not 0 < self.min_sell_recovery_pct <= 100:
            raise ValueError("memecoin sell-recovery floor is invalid")
        if self.max_shadow_position_usd <= 0:
            raise ValueError("memecoin shadow-position cap must be positive")
        for source in (self.token_authority_source_url, self.token_risk_source_url):
            parsed = urlsplit(source)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("memecoin policy sources must be absolute HTTPS URLs")


def load_memecoin_sandbox_config(path: str | Path) -> MemecoinSandboxConfig:
    config_path = Path(path)
    if config_path.stat().st_size > 1_000_000:
        raise ValueError("memecoin sandbox config exceeds the 1 MB safety limit")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "version",
        "min_liquidity_usd",
        "min_pool_age_hours",
        "max_top10_holder_pct",
        "max_largest_holder_pct",
        "max_transfer_fee_bps",
        "max_price_impact_bps",
        "max_oracle_deviation_bps",
        "min_sell_recovery_pct",
        "max_snapshot_age_seconds",
        "max_shadow_position_usd",
        "token_authority_source_url",
        "token_risk_source_url",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            "memecoin sandbox config keys mismatch: "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )
    return MemecoinSandboxConfig(
        version=str(raw["version"]),
        min_liquidity_usd=float(raw["min_liquidity_usd"]),
        min_pool_age_hours=float(raw["min_pool_age_hours"]),
        max_top10_holder_pct=float(raw["max_top10_holder_pct"]),
        max_largest_holder_pct=float(raw["max_largest_holder_pct"]),
        max_transfer_fee_bps=float(raw["max_transfer_fee_bps"]),
        max_price_impact_bps=float(raw["max_price_impact_bps"]),
        max_oracle_deviation_bps=float(raw["max_oracle_deviation_bps"]),
        min_sell_recovery_pct=float(raw["min_sell_recovery_pct"]),
        max_snapshot_age=timedelta(seconds=float(raw["max_snapshot_age_seconds"])),
        max_shadow_position_usd=float(raw["max_shadow_position_usd"]),
        token_authority_source_url=str(raw["token_authority_source_url"]),
        token_risk_source_url=str(raw["token_risk_source_url"]),
    )


@dataclass(frozen=True)
class MemecoinRiskSnapshot:
    token_id: str
    chain: str
    observed_at: datetime
    liquidity_usd: float
    pool_age_hours: float
    top10_holder_pct: float
    largest_holder_pct: float
    transfer_fee_bps: float
    price_impact_bps: float
    oracle_deviation_bps: float
    expected_sell_value: float
    simulated_sell_value: float
    mint_authority_active: bool
    freeze_authority_active: bool
    permanent_delegate_active: bool
    transfer_hook_active: bool
    pausable_active: bool
    source_verified: bool
    buy_simulation_passed: bool
    sell_simulation_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", require_aware(self.observed_at, "observed_at")
        )
        boolean_fields = (
            "mint_authority_active",
            "freeze_authority_active",
            "permanent_delegate_active",
            "transfer_hook_active",
            "pausable_active",
            "source_verified",
            "buy_simulation_passed",
            "sell_simulation_passed",
        )
        if any(type(getattr(self, field)) is not bool for field in boolean_fields):
            raise ValueError("memecoin risk flags must be boolean")
        numeric = (
            self.liquidity_usd,
            self.pool_age_hours,
            self.top10_holder_pct,
            self.largest_holder_pct,
            self.transfer_fee_bps,
            self.price_impact_bps,
            self.oracle_deviation_bps,
            self.expected_sell_value,
            self.simulated_sell_value,
        )
        if not self.token_id or not self.chain:
            raise ValueError("memecoin snapshot identity is required")
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("memecoin snapshot values must be finite")
        if min(self.liquidity_usd, self.pool_age_hours, self.expected_sell_value) <= 0:
            raise ValueError("memecoin liquidity, age, and expected sell value must be positive")
        if self.simulated_sell_value < 0:
            raise ValueError("simulated sell value cannot be negative")
        if not 0 <= self.largest_holder_pct <= self.top10_holder_pct <= 100:
            raise ValueError("memecoin holder concentration is invalid")
        if min(self.transfer_fee_bps, self.price_impact_bps, self.oracle_deviation_bps) < 0:
            raise ValueError("memecoin costs and deviations cannot be negative")

    @property
    def sell_recovery_pct(self) -> float:
        return self.simulated_sell_value / self.expected_sell_value * 100


class MemecoinRiskStatus(StrEnum):
    BLOCKED = "blocked"
    SANDBOX_ELIGIBLE = "sandbox_eligible"


@dataclass(frozen=True)
class MemecoinRiskDecision:
    token_id: str
    status: MemecoinRiskStatus
    evaluated_at: datetime
    risk_score: int
    reasons: tuple[str, ...]
    snapshot_digest: str


def evaluate_memecoin_risk(
    snapshot: MemecoinRiskSnapshot,
    config: MemecoinSandboxConfig,
    *,
    now: datetime,
) -> MemecoinRiskDecision:
    now = require_aware(now, "now")
    reasons: list[str] = []
    if snapshot.observed_at > now:
        reasons.append("snapshot is from the future")
    elif now - snapshot.observed_at > config.max_snapshot_age:
        reasons.append("snapshot is stale")
    if not snapshot.source_verified:
        reasons.append("token source or program is unverified")
    if snapshot.mint_authority_active:
        reasons.append("mint authority can inflate supply")
    if snapshot.freeze_authority_active:
        reasons.append("freeze authority can block token accounts")
    if snapshot.permanent_delegate_active:
        reasons.append("permanent delegate can transfer or burn holder tokens")
    if snapshot.transfer_hook_active:
        reasons.append("transfer hook can alter transfer behavior")
    if snapshot.pausable_active:
        reasons.append("token transfers can be paused")
    if snapshot.liquidity_usd < config.min_liquidity_usd:
        reasons.append("pool liquidity is below the safety floor")
    if snapshot.pool_age_hours < config.min_pool_age_hours:
        reasons.append("pool is too new")
    if snapshot.top10_holder_pct > config.max_top10_holder_pct:
        reasons.append("top-ten holder concentration exceeds the limit")
    if snapshot.largest_holder_pct > config.max_largest_holder_pct:
        reasons.append("largest holder concentration exceeds the limit")
    if snapshot.transfer_fee_bps > config.max_transfer_fee_bps:
        reasons.append("token transfer fee exceeds the limit")
    if snapshot.price_impact_bps > config.max_price_impact_bps:
        reasons.append("estimated price impact exceeds the limit")
    if snapshot.oracle_deviation_bps > config.max_oracle_deviation_bps:
        reasons.append("pool price diverges from the reference price")
    if not snapshot.buy_simulation_passed:
        reasons.append("buy simulation failed")
    if not snapshot.sell_simulation_passed:
        reasons.append("sell simulation failed")
    if snapshot.sell_recovery_pct < config.min_sell_recovery_pct:
        reasons.append("sell simulation recovery is below the floor")
    risk_score = min(100, len(reasons) * 12)
    return MemecoinRiskDecision(
        snapshot.token_id,
        MemecoinRiskStatus.BLOCKED if reasons else MemecoinRiskStatus.SANDBOX_ELIGIBLE,
        now,
        risk_score,
        tuple(reasons),
        sha256_digest(snapshot),
    )


@dataclass(frozen=True)
class MemecoinShadowIntent:
    intent_id: str
    token_id: str
    strategy_id: str
    notional_usd: float
    created_at: datetime
    snapshot_digest: str
    environment: str = "shadow"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "created_at", require_aware(self.created_at, "created_at")
        )
        if not self.intent_id or not self.token_id or not self.strategy_id:
            raise ValueError("memecoin intent identity is required")
        if not math.isfinite(self.notional_usd) or self.notional_usd <= 0:
            raise ValueError("memecoin intent notional must be finite and positive")
        if self.environment != "shadow":
            raise ValueError("memecoin sandbox intents must remain shadow-only")


@dataclass(frozen=True)
class MemecoinAuditEvent:
    event_id: str
    event_type: str
    occurred_at: datetime
    payload_json: str
    digest: str


class MemecoinSafetySandbox:
    def __init__(
        self,
        config: MemecoinSandboxConfig,
        *,
        enabled: bool = False,
        strategy_eligible: bool = False,
    ) -> None:
        self.config = config
        self.enabled = enabled
        self.strategy_eligible = strategy_eligible
        self._events: list[MemecoinAuditEvent] = []

    @property
    def events(self) -> tuple[MemecoinAuditEvent, ...]:
        return tuple(self._events)

    def propose(
        self,
        snapshot: MemecoinRiskSnapshot,
        *,
        notional_usd: float,
        now: datetime,
    ) -> MemecoinShadowIntent:
        if not self.enabled:
            raise PermissionError("memecoin safety sandbox is disabled")
        if not self.strategy_eligible:
            raise PermissionError("memecoin strategy has not passed evidence gates")
        if not math.isfinite(notional_usd) or notional_usd <= 0:
            raise ValueError("memecoin shadow notional must be finite and positive")
        if notional_usd > self.config.max_shadow_position_usd:
            raise PermissionError("memecoin shadow notional exceeds the policy cap")
        decision = evaluate_memecoin_risk(snapshot, self.config, now=now)
        self._append_event("risk_decision", now, {"decision": decision})
        if decision.status is not MemecoinRiskStatus.SANDBOX_ELIGIBLE:
            raise PermissionError("memecoin token failed safety screening")
        intent_id = sha256_digest(
            {
                "token_id": snapshot.token_id,
                "snapshot_digest": decision.snapshot_digest,
                "notional_usd": notional_usd,
            }
        )
        intent = MemecoinShadowIntent(
            intent_id,
            snapshot.token_id,
            "memecoin-safety-baseline",
            notional_usd,
            require_aware(now, "now"),
            decision.snapshot_digest,
        )
        self._append_event("shadow_intent", now, {"intent": intent})
        return intent

    def verify_integrity(self) -> int:
        for ordinal, event in enumerate(self._events):
            payload = json.loads(event.payload_json)
            if sha256_digest(payload) != event.digest:
                raise RuntimeError(f"memecoin event digest mismatch: {event.event_id}")
            expected = sha256_digest({"ordinal": ordinal, "payload": payload})
            if event.event_id != expected:
                raise RuntimeError(f"memecoin event identity mismatch: {event.event_id}")
        return len(self._events)

    def _append_event(
        self, event_type: str, occurred_at: datetime, payload: Mapping[str, object]
    ) -> None:
        occurred_at = require_aware(occurred_at, "occurred_at")
        body = {"event_type": event_type, "occurred_at": occurred_at, **payload}
        payload_json = canonical_json(body)
        normalized = json.loads(payload_json)
        event_id = sha256_digest({"ordinal": len(self._events), "payload": normalized})
        self._events.append(
            MemecoinAuditEvent(
                event_id,
                event_type,
                occurred_at,
                payload_json,
                sha256_digest(normalized),
            )
        )


class MemecoinScenarioStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class MemecoinScenarioResult:
    scenario_id: str
    label: str
    status: MemecoinScenarioStatus
    checks: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class MemecoinSandboxReport:
    generated_at: datetime
    config_version: str
    scenarios: tuple[MemecoinScenarioResult, ...]
    network_access: bool = False
    wallet_credentials_used: bool = False
    real_transactions_signed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generated_at", require_aware(self.generated_at, "generated_at")
        )
        if self.network_access or self.wallet_credentials_used or self.real_transactions_signed:
            raise ValueError("memecoin sandbox report must remain isolated")

    @property
    def passed(self) -> int:
        return sum(item.status is MemecoinScenarioStatus.PASSED for item in self.scenarios)

    @property
    def failed(self) -> int:
        return len(self.scenarios) - self.passed

    @property
    def successful(self) -> bool:
        return self.failed == 0


def memecoin_scenario_names() -> tuple[str, ...]:
    return tuple(_MEMECOIN_SCENARIOS)


def run_memecoin_sandbox_scenarios(
    scenario: str = "all",
    *,
    config: MemecoinSandboxConfig | None = None,
    generated_at: datetime | None = None,
) -> MemecoinSandboxReport:
    config = config or MemecoinSandboxConfig()
    selected = tuple(_MEMECOIN_SCENARIOS) if scenario == "all" else (scenario,)
    if any(item not in _MEMECOIN_SCENARIOS for item in selected):
        raise ValueError(f"unknown memecoin sandbox scenario: {selected[0]}")
    results = tuple(
        _run_safely(item, _MEMECOIN_SCENARIOS[item], config) for item in selected
    )
    return MemecoinSandboxReport(
        generated_at or datetime.now(timezone.utc), config.version, results
    )


def render_memecoin_sandbox_report(
    report: MemecoinSandboxReport, output_format: str = "text"
) -> str:
    if output_format == "json":
        payload = asdict(report)
        payload["generated_at"] = report.generated_at.isoformat()
        payload.update(
            passed=report.passed, failed=report.failed, successful=report.successful
        )
        return json.dumps(payload, sort_keys=True, indent=2)
    if output_format == "markdown":
        lines = [
            "## Memecoin safety sandbox",
            "",
            f"**{'PASS' if report.successful else 'FAIL'}** · "
            f"{report.passed}/{len(report.scenarios)} scenarios passed · "
            f"policy `{report.config_version}` · network disabled · wallet unused · transactions 0",
            "",
            "| Scenario | Status | Verified behavior |",
            "|---|---:|---|",
        ]
        for item in report.scenarios:
            checks = "; ".join(item.checks).replace("|", "\\|")
            lines.append(f"| {item.label} | {item.status.value.upper()} | {checks} |")
        return "\n".join(lines)
    if output_format != "text":
        raise ValueError("memecoin sandbox format must be text, json, or markdown")
    lines = [
        f"Memecoin safety sandbox: {'PASS' if report.successful else 'FAIL'} "
        f"passed={report.passed} failed={report.failed} policy={report.config_version} "
        "network=false wallet=false transactions=0"
    ]
    for item in report.scenarios:
        lines.append(f"{item.scenario_id}: {item.status.value} - {item.detail}")
    return "\n".join(lines)


def _run_safely(
    scenario_id: str,
    scenario: tuple[
        str, Callable[[MemecoinSandboxConfig], tuple[tuple[str, ...], str]]
    ],
    config: MemecoinSandboxConfig,
) -> MemecoinScenarioResult:
    label, operation = scenario
    try:
        with patch(
            "socket.create_connection",
            side_effect=PermissionError("network disabled in memecoin sandbox"),
        ):
            checks, detail = operation(config)
        return MemecoinScenarioResult(
            scenario_id, label, MemecoinScenarioStatus.PASSED, checks, detail
        )
    except Exception as exc:
        return MemecoinScenarioResult(
            scenario_id,
            label,
            MemecoinScenarioStatus.FAILED,
            (),
            f"{type(exc).__name__}: {exc}",
        )


def _safe_token(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    snapshot = _snapshot()
    decision = evaluate_memecoin_risk(snapshot, config, now=SANDBOX_TIME)
    sandbox = MemecoinSafetySandbox(config, enabled=True, strategy_eligible=True)
    intent = sandbox.propose(snapshot, notional_usd=100, now=SANDBOX_TIME)
    _require(decision.status is MemecoinRiskStatus.SANDBOX_ELIGIBLE, "safe token blocked")
    _require(intent.environment == "shadow", "memecoin intent escaped shadow")
    _require(sandbox.verify_integrity() == 2, "memecoin audit integrity failed")
    return (("strict screen passed", "shadow only", "audit verified"), "fully screened token reached shadow simulation only")


def _mint_authority(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    decision = evaluate_memecoin_risk(
        replace(_snapshot(), mint_authority_active=True), config, now=SANDBOX_TIME
    )
    _require(any("inflate" in reason for reason in decision.reasons), "mint risk missing")
    return (("mint authority blocked",), "retained supply authority failed closed")


def _freeze_delegate(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    decision = evaluate_memecoin_risk(
        replace(
            _snapshot(), freeze_authority_active=True, permanent_delegate_active=True
        ),
        config,
        now=SANDBOX_TIME,
    )
    _require(len(decision.reasons) == 2, "authority risks were not independently reported")
    return (("freeze blocked", "permanent delegate blocked"), "holder-control authorities failed screening")


def _token_extensions(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    decision = evaluate_memecoin_risk(
        replace(
            _snapshot(),
            transfer_hook_active=True,
            pausable_active=True,
            transfer_fee_bps=config.max_transfer_fee_bps + 1,
        ),
        config,
        now=SANDBOX_TIME,
    )
    _require(len(decision.reasons) == 3, "dangerous extensions were not blocked")
    return (("transfer hook", "pause authority", "transfer fee"), "dangerous token extensions failed screening")


def _liquidity_concentration(
    config: MemecoinSandboxConfig,
) -> tuple[tuple[str, ...], str]:
    decision = evaluate_memecoin_risk(
        replace(
            _snapshot(),
            liquidity_usd=config.min_liquidity_usd - 1,
            pool_age_hours=config.min_pool_age_hours - 1,
            top10_holder_pct=config.max_top10_holder_pct + 1,
            largest_holder_pct=config.max_largest_holder_pct + 1,
        ),
        config,
        now=SANDBOX_TIME,
    )
    _require(len(decision.reasons) == 4, "structure risks were not all blocked")
    return (("liquidity floor", "pool age", "holder concentration"), "fragile pool structure failed screening")


def _market_integrity(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    decision = evaluate_memecoin_risk(
        replace(
            _snapshot(),
            price_impact_bps=config.max_price_impact_bps + 1,
            oracle_deviation_bps=config.max_oracle_deviation_bps + 1,
            observed_at=SANDBOX_TIME - config.max_snapshot_age - timedelta(seconds=1),
        ),
        config,
        now=SANDBOX_TIME,
    )
    _require(len(decision.reasons) == 3, "market integrity risks were not blocked")
    return (("freshness", "price impact", "oracle divergence"), "unreliable market state failed screening")


def _sell_simulation(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    decision = evaluate_memecoin_risk(
        replace(
            _snapshot(),
            sell_simulation_passed=False,
            simulated_sell_value=0,
        ),
        config,
        now=SANDBOX_TIME,
    )
    _require(any("sell simulation failed" == reason for reason in decision.reasons), "sell failure missing")
    _require(any("recovery" in reason for reason in decision.reasons), "recovery failure missing")
    return (("sellability", "round-trip recovery"), "honeypot-like sell behavior failed closed")


def _runtime_limits(config: MemecoinSandboxConfig) -> tuple[tuple[str, ...], str]:
    snapshot = _snapshot()
    disabled = MemecoinSafetySandbox(config, enabled=False, strategy_eligible=True)
    ineligible = MemecoinSafetySandbox(config, enabled=True, strategy_eligible=False)
    for sandbox in (disabled, ineligible):
        try:
            sandbox.propose(snapshot, notional_usd=100, now=SANDBOX_TIME)
        except PermissionError:
            pass
        else:
            raise AssertionError("closed memecoin gate accepted an intent")
    enabled = MemecoinSafetySandbox(config, enabled=True, strategy_eligible=True)
    try:
        enabled.propose(
            snapshot,
            notional_usd=config.max_shadow_position_usd + 1,
            now=SANDBOX_TIME,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("memecoin position cap was bypassed")
    _require(enabled.events == (), "rejected notional mutated memecoin audit")
    return (("runtime gate", "evidence gate", "position cap"), "memecoin controls remained closed by default")


def _snapshot() -> MemecoinRiskSnapshot:
    return MemecoinRiskSnapshot(
        "sandbox:solana:MEME",
        "solana",
        SANDBOX_TIME,
        500_000,
        720,
        35,
        7,
        0,
        100,
        100,
        100,
        98,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_MEMECOIN_SCENARIOS: dict[
    str, tuple[str, Callable[[MemecoinSandboxConfig], tuple[tuple[str, ...], str]]]
] = {
    "safe-token": ("Fully screened token", _safe_token),
    "mint-authority": ("Retained mint authority", _mint_authority),
    "freeze-delegate": ("Freeze and delegate authority", _freeze_delegate),
    "token-extensions": ("Dangerous token extensions", _token_extensions),
    "liquidity-concentration": ("Liquidity and concentration", _liquidity_concentration),
    "market-integrity": ("Market-state integrity", _market_integrity),
    "sell-simulation": ("Sell simulation and recovery", _sell_simulation),
    "runtime-limits": ("Runtime and position limits", _runtime_limits),
}
