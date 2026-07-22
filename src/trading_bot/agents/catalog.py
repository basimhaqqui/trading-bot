from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    purpose: str
    permissions: tuple[str, ...]


SKILLS: tuple[SkillDefinition, ...] = (
    SkillDefinition(
        "market-data-point-in-time",
        "Ingest replayable data with event and availability timestamps",
        ("read_external_data", "write_immutable_data"),
    ),
    SkillDefinition(
        "research-hypothesis-and-falsification",
        "Register hypotheses, track trials, and try to disprove results",
        ("read_research_data", "write_experiments"),
    ),
    SkillDefinition("equity-specialist", "Research point-in-time equity signals", ("read_research_data",)),
    SkillDefinition("options-volatility-specialist", "Model option surfaces and volatility", ("read_research_data",)),
    SkillDefinition("futures-term-structure-specialist", "Model futures curves, carry, and rolls", ("read_research_data",)),
    SkillDefinition("crypto-market-structure-specialist", "Research fragmented spot crypto markets", ("read_research_data",)),
    SkillDefinition("perpetual-funding-and-liquidation-specialist", "Model funding, basis, and liquidation", ("read_research_data",)),
    SkillDefinition("onchain-memecoin-forensics", "Classify token and execution safety", ("read_chain_data", "simulate_transactions")),
    SkillDefinition("prediction-market-forecaster", "Produce calibrated forecasts tied to settlement rules", ("read_research_data", "read_primary_sources")),
    SkillDefinition("execution-and-market-microstructure", "Execute signed intents without choosing exposure", ("submit_approved_orders", "cancel_orders")),
    SkillDefinition("portfolio-and-margin-risk-governor", "Reject or reduce unsafe exposure", ("read_all_positions", "approve_reduced_risk", "cancel_orders")),
    SkillDefinition("strategy-validation-red-team", "Independently reproduce and attack strategy claims", ("read_research_data", "read_experiments", "run_isolated_replays")),
    SkillDefinition("agent-security-and-operations", "Protect credentials, tools, logs, and recovery paths", ("manage_policy", "read_audit_logs", "trigger_shutdown")),
)


def skill_names() -> tuple[str, ...]:
    return tuple(skill.name for skill in SKILLS)
