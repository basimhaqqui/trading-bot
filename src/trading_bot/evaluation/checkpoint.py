from __future__ import annotations

from datetime import datetime

from trading_bot.core.audit import AuditLedger
from trading_bot.core.serialization import require_aware
from trading_bot.evaluation.reporting import (
    EvaluationDecision,
    EvaluationGateConfig,
    WalkForwardReport,
    build_walk_forward_report,
    new_mature_decisions,
)


def locked_walk_forward_report(
    audit: AuditLedger,
    config: EvaluationGateConfig | None = None,
) -> WalkForwardReport:
    return build_walk_forward_report(
        audit.forecasts(),
        audit.forecast_scores(),
        config or EvaluationGateConfig(),
        locked_decisions=audit.evaluation_decisions(),
    )


def checkpointed_walk_forward_report(
    audit: AuditLedger,
    config: EvaluationGateConfig | None = None,
    *,
    as_of: datetime,
) -> tuple[WalkForwardReport, tuple[EvaluationDecision, ...]]:
    as_of = require_aware(as_of, "as_of")
    config = config or EvaluationGateConfig()
    forecasts = audit.forecasts()
    scores = audit.forecast_scores()
    report = build_walk_forward_report(
        forecasts,
        scores,
        config,
        locked_decisions=audit.evaluation_decisions(),
    )
    recorded = tuple(
        decision
        for decision in new_mature_decisions(report, config=config, as_of=as_of)
        if audit.append_evaluation_decision(decision)
    )
    if recorded:
        report = build_walk_forward_report(
            forecasts,
            scores,
            config,
            locked_decisions=audit.evaluation_decisions(),
        )
    return report, recorded
