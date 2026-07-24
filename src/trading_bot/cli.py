from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.agents.catalog import SKILLS
from trading_bot.agents.demo import DemoRegimeSpecialist
from trading_bot.agents.hypotheses import BASELINE_HYPOTHESES
from trading_bot.core.audit import AuditLedger
from trading_bot.core.experiments import ExperimentRegistry
from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import utc_now
from trading_bot.core.snapshot import create_verified_snapshot
from trading_bot.core.store import PointInTimeStore
from trading_bot.data.collectors import (
    AlpacaOptionsCollector,
    AlpacaStockCollector,
    CoinbaseCollector,
    KalshiCollector,
)
from trading_bot.execution.control import DeterministicExecutor, PaperLedgerAdapter
from trading_bot.execution.alpaca import AlpacaPaperClient
from trading_bot.execution.drills import (
    render_paper_drill_report,
    run_paper_drills,
    scenario_names,
)
from trading_bot.execution.crypto_sandbox import (
    load_crypto_sandbox_config,
    render_crypto_sandbox_report,
    run_crypto_sandbox_scenarios,
    sandbox_scenario_names,
)
from trading_bot.execution.prediction_sandbox import (
    load_prediction_sandbox_config,
    prediction_scenario_names,
    render_prediction_sandbox_report,
    run_prediction_sandbox_scenarios,
)
from trading_bot.execution.options_sandbox import (
    load_options_sandbox_config,
    options_scenario_names,
    render_options_sandbox_report,
    run_options_sandbox_scenarios,
)
from trading_bot.execution.memecoin_sandbox import (
    load_memecoin_sandbox_config,
    memecoin_scenario_names,
    render_memecoin_sandbox_report,
    run_memecoin_sandbox_scenarios,
)
from trading_bot.execution.operations import (
    PaperControlStore,
    PaperExecutionLedger,
    PaperReconciler,
    activate_paper_emergency_stop,
)
from trading_bot.execution.paper import (
    AlpacaPaperAllocator,
    PaperExecutionService,
    candidate_eligibility,
    load_paper_risk_config,
)
from trading_bot.execution.risk import ApprovalSigner, RiskGovernor, RiskLimits
from trading_bot.execution.schemas import (
    ExecutionEnvironment,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
)
from trading_bot.evaluation.readiness import data_readiness
from trading_bot.evaluation.costs import load_cost_registry
from trading_bot.evaluation.economics import EconomicGateConfig, build_economic_report
from trading_bot.evaluation.checkpoint import (
    checkpointed_walk_forward_report,
    locked_walk_forward_report,
)
from trading_bot.evaluation.reporting import (
    EvaluationDecision,
    EvaluationGateConfig,
    WalkForwardReport,
)
from trading_bot.evaluation.shadow import ShadowResearchResult, ShadowResearchRunner
from trading_bot.evaluation.scorecard import (
    ScorecardStatus,
    build_daily_scorecard,
    render_github_alerts,
    render_scorecard,
)
from trading_bot.evaluation.launch_readiness import (
    LaunchReadinessStatus,
    build_launch_readiness_report,
    render_launch_readiness_report,
)
from trading_bot.ingestion.plan import load_plan
from trading_bot.ingestion.health import ingestion_health, render_health
from trading_bot.ingestion.runner import (
    IngestionRunLedger,
    IngestionRunStatus,
    ShadowIngestionRunner,
)
from trading_bot.replay import ReplayEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-market research and controlled-execution foundation"
    )
    parser.add_argument(
        "--db",
        default=os.getenv("TRADING_DB_PATH", "var/trading.db"),
        help="SQLite research database path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="initialize the local point-in-time database")
    subparsers.add_parser("skills", help="list specialist and control skills")
    subparsers.add_parser("demo", help="run a safe end-to-end replay and shadow order")
    subparsers.add_parser("doctor", help="check the database and safety defaults")
    subparsers.add_parser(
        "research-init", help="register the versioned baseline research hypotheses"
    )
    readiness = subparsers.add_parser(
        "readiness", help="show which research specialists have enough shadow data"
    )
    readiness.add_argument("--perpetual", default="BIP-20DEC30-CDE")
    readiness.add_argument("--spot", default="BTC-USD")
    shadow_cycle = subparsers.add_parser(
        "shadow-cycle", help="run one read-only scheduled observation cycle"
    )
    shadow_cycle.add_argument("--plan", required=True, help="validated ingestion plan JSON")
    shadow_cycle.add_argument(
        "--validate-only", action="store_true", help="validate and print jobs without network calls"
    )
    subparsers.add_parser(
        "shadow-research",
        help="generate point-in-time shadow forecasts and score available outcomes",
    )
    shadow_report = subparsers.add_parser(
        "shadow-report",
        help="report benchmark-relative walk-forward evidence and controls",
    )
    shadow_report.add_argument("--min-outcomes", type=int, default=30)
    shadow_health = subparsers.add_parser(
        "shadow-health", help="check freshness and failures for every enabled ingestion job"
    )
    shadow_health.add_argument("--plan", required=True, help="validated ingestion plan JSON")
    shadow_health.add_argument("--max-age-minutes", type=int, default=90)
    shadow_health.add_argument("--max-consecutive-failures", type=int, default=0)
    shadow_health.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    economic_report = subparsers.add_parser(
        "economic-report",
        help="run candidate-only net-return replay with doubled-cost stress",
    )
    economic_report.add_argument(
        "--costs", default="config/economic-costs.json", help="cost registry JSON"
    )
    economic_report.add_argument("--min-outcomes", type=int, default=30)
    economic_report.add_argument("--min-trades", type=int, default=30)
    scorecard = subparsers.add_parser(
        "daily-scorecard",
        help="combine ingestion, market coverage, forecast, and economic evidence",
    )
    scorecard.add_argument("--plan", required=True, help="validated ingestion plan JSON")
    scorecard.add_argument(
        "--costs", default="config/economic-costs.json", help="cost registry JSON"
    )
    scorecard.add_argument("--min-outcomes", type=int, default=30)
    scorecard.add_argument("--min-trades", type=int, default=30)
    scorecard.add_argument("--max-age-minutes", type=int, default=90)
    scorecard.add_argument("--max-consecutive-failures", type=int, default=0)
    scorecard.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    scorecard.add_argument("--output", help="optional rendered scorecard path")
    scorecard.add_argument("--json-output", help="optional machine-readable scorecard path")
    scorecard.add_argument("--emit-github-alerts", action="store_true")
    scorecard.add_argument("--fail-on-critical", action="store_true")
    paper_status = subparsers.add_parser(
        "paper-status", help="read Alpaca paper account state without placing orders"
    )
    paper_status.add_argument(
        "--costs", default="config/economic-costs.json", help="cost registry JSON"
    )
    paper_status.add_argument(
        "--policy", default="config/paper-execution.json", help="paper policy JSON"
    )
    subparsers.add_parser(
        "paper-reconcile",
        help="record paper account, positions, and order state without placing orders",
    )
    paper_control = subparsers.add_parser(
        "paper-control", help="manage the persistent paper execution interlock"
    )
    paper_control.add_argument(
        "action", choices=("status", "enable", "disable", "kill", "release")
    )
    paper_control.add_argument("--confirm", default="")
    paper_control.add_argument("--reason", default="operator request")
    paper_control.add_argument("--cancel-open-orders", action="store_true")
    paper_plan = subparsers.add_parser(
        "paper-plan", help="preview eligibility-gated Alpaca paper allocations"
    )
    paper_plan.add_argument(
        "--costs", default="config/economic-costs.json", help="cost registry JSON"
    )
    paper_plan.add_argument(
        "--policy", default="config/paper-execution.json", help="paper policy JSON"
    )
    paper_cycle = subparsers.add_parser(
        "paper-cycle", help="preview or execute one eligibility-gated paper cycle"
    )
    paper_cycle.add_argument(
        "--costs", default="config/economic-costs.json", help="cost registry JSON"
    )
    paper_cycle.add_argument(
        "--policy", default="config/paper-execution.json", help="paper policy JSON"
    )
    paper_cycle.add_argument("--execute", action="store_true")
    paper_cycle.add_argument("--confirm", default="")
    paper_drill = subparsers.add_parser(
        "paper-drill",
        help="run isolated paper incident drills without network or broker credentials",
    )
    paper_drill.add_argument(
        "--scenario", choices=("all", *scenario_names()), default="all"
    )
    paper_drill.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    paper_drill.add_argument("--output", help="optional drill report path")
    crypto_sandbox = subparsers.add_parser(
        "crypto-sandbox",
        help="run credential-free spot crypto and perpetual execution scenarios",
    )
    crypto_sandbox.add_argument(
        "--scenario", choices=("all", *sandbox_scenario_names()), default="all"
    )
    crypto_sandbox.add_argument(
        "--policy",
        default="config/crypto-sandbox.json",
        help="crypto sandbox policy JSON",
    )
    crypto_sandbox.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    crypto_sandbox.add_argument("--output", help="optional sandbox report path")
    prediction_sandbox = subparsers.add_parser(
        "prediction-sandbox",
        help="run credential-free prediction execution and settlement scenarios",
    )
    prediction_sandbox.add_argument(
        "--scenario", choices=("all", *prediction_scenario_names()), default="all"
    )
    prediction_sandbox.add_argument(
        "--policy",
        default="config/prediction-sandbox.json",
        help="prediction sandbox policy JSON",
    )
    prediction_sandbox.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    prediction_sandbox.add_argument("--output", help="optional sandbox report path")
    options_sandbox = subparsers.add_parser(
        "options-sandbox",
        help="run credential-free option lifecycle and delta-hedging scenarios",
    )
    options_sandbox.add_argument(
        "--scenario", choices=("all", *options_scenario_names()), default="all"
    )
    options_sandbox.add_argument(
        "--policy",
        default="config/options-sandbox.json",
        help="options sandbox policy JSON",
    )
    options_sandbox.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    options_sandbox.add_argument("--output", help="optional sandbox report path")
    memecoin_sandbox = subparsers.add_parser(
        "memecoin-sandbox",
        help="run credential-free token and liquidity safety scenarios",
    )
    memecoin_sandbox.add_argument(
        "--scenario", choices=("all", *memecoin_scenario_names()), default="all"
    )
    memecoin_sandbox.add_argument(
        "--policy",
        default="config/memecoin-sandbox.json",
        help="memecoin sandbox policy JSON",
    )
    memecoin_sandbox.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    memecoin_sandbox.add_argument("--output", help="optional sandbox report path")
    launch_readiness = subparsers.add_parser(
        "launch-readiness",
        help="evaluate the complete roadmap without authorizing live execution",
    )
    launch_readiness.add_argument(
        "--plan", default="config/shadow-ingestion.json", help="ingestion plan JSON"
    )
    launch_readiness.add_argument(
        "--costs", default="config/economic-costs.json", help="cost registry JSON"
    )
    launch_readiness.add_argument(
        "--policy",
        default="config/launch-readiness.json",
        help="launch readiness policy JSON",
    )
    launch_readiness.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    launch_readiness.add_argument("--output", help="optional readiness report path")
    launch_readiness.add_argument(
        "--require-paper-review",
        action="store_true",
        help="return a failure unless every paper-review gate passes",
    )
    snapshot = subparsers.add_parser(
        "snapshot", help="create an atomic, integrity-checked database snapshot"
    )
    snapshot.add_argument("--output", required=True, help="snapshot database output path")
    collect = subparsers.add_parser("collect", help="collect and store read-only venue data")
    collect.add_argument("venue", choices=("kalshi", "coinbase", "alpaca"))
    collect.add_argument(
        "dataset",
        choices=("markets", "trades", "book", "products", "candles", "chain", "bars"),
    )
    collect.add_argument("--symbol", help="market ticker, product ID, or underlying symbol")
    collect.add_argument("--limit", type=int, default=100)
    collect.add_argument("--cursor")
    collect.add_argument("--status", default="open")
    collect.add_argument("--feed", choices=("opra", "indicative"), default="indicative")
    collect.add_argument(
        "--stock-feed", choices=("iex", "sip", "delayed_sip"), default="iex"
    )
    collect.add_argument("--lookback-days", type=int, default=45)
    collect.add_argument(
        "--granularity",
        choices=tuple(CoinbaseCollector.CANDLE_GRANULARITIES),
        default="ONE_HOUR",
    )
    return parser


def _initialize(path: Path) -> tuple[PointInTimeStore, ExperimentRegistry, AuditLedger]:
    store = PointInTimeStore(path)
    store.initialize()
    registry = ExperimentRegistry(path)
    registry.initialize()
    audit = AuditLedger(path)
    audit.initialize()
    IngestionRunLedger(path).initialize()
    PaperControlStore(path).initialize()
    PaperExecutionLedger(path).initialize()
    return store, registry, audit


def _risk_signer() -> ApprovalSigner:
    key = os.getenv("RISK_SIGNING_KEY", "development-only-change-me").encode("utf-8")
    return ApprovalSigner(key)


def _paper_client() -> AlpacaPaperClient:
    key_id = os.getenv("ALPACA_PAPER_KEY_ID") or os.getenv("ALPACA_MARKET_DATA_KEY_ID", "")
    secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv(
        "ALPACA_MARKET_DATA_SECRET_KEY", ""
    )
    return AlpacaPaperClient(key_id, secret_key)


def _paper_signer() -> ApprovalSigner:
    key = os.getenv("RISK_SIGNING_KEY", "")
    if not key or key == "development-only-change-me":
        raise PermissionError("paper execution requires a non-default RISK_SIGNING_KEY")
    return ApprovalSigner(key.encode("utf-8"), key_id="paper-risk-v1")


def _paper_risk_limits() -> RiskLimits:
    return RiskLimits(
        max_gross_notional=float(os.getenv("MAX_GROSS_NOTIONAL", "100000")),
        max_instrument_notional=float(os.getenv("MAX_INSTRUMENT_NOTIONAL", "10000")),
        max_venue_notional=float(os.getenv("MAX_VENUE_NOTIONAL", "30000")),
        asset_class_caps={
            AssetClass.EQUITY: float(os.getenv("MAX_EQUITY_NOTIONAL", "30000")),
            AssetClass.OPTION: float(os.getenv("MAX_OPTION_NOTIONAL", "10000")),
            AssetClass.MEMECOIN: 0,
            AssetClass.PREDICTION: float(os.getenv("MAX_PREDICTION_NOTIONAL", "5000")),
        },
        allow_live=False,
    )


def _demo(path: Path) -> int:
    store, _, audit = _initialize(path)
    instrument = Instrument(
        instrument_id="demo:equity:SPY",
        venue="demo",
        symbol="SPY",
        asset_class=AssetClass.EQUITY,
        quote_currency="USD",
    )
    store.register_instrument(instrument)
    base = datetime(2026, 1, 1, 20, tzinfo=timezone.utc)
    for index, close in enumerate((100.0, 102.0, 101.0), start=1):
        event_time = base + timedelta(days=index - 1)
        available_at = event_time + timedelta(minutes=1)
        store.append_event(
            MarketEvent(
                event_id=f"demo-bar-{index}",
                event_type=MarketEventType.BAR,
                venue="demo",
                instrument_id=instrument.instrument_id,
                event_time=event_time,
                available_at=available_at,
                source="deterministic-demo",
                payload={"close": close},
                sequence=index,
                ingested_at=available_at,
            )
        )

    decision_times = (
        base + timedelta(days=2),
        base + timedelta(days=3),
    )
    replay = ReplayEngine(store).run(
        DemoRegimeSpecialist(),
        instrument_id=instrument.instrument_id,
        decision_times=decision_times,
    )
    for forecast in replay.forecasts:
        audit.append_forecast(forecast)
    print(f"Replay: {replay.decision_times} decisions, {len(replay.forecasts)} forecasts")
    for forecast in replay.forecasts:
        print(
            f"  {forecast.generated_at.isoformat()} "
            f"regime={forecast.values['regime']} "
            f"last_available_close={forecast.values['last_available_close']} "
            f"evidence={','.join(forecast.evidence_event_ids)}"
        )

    now = base + timedelta(days=4)
    intent = OrderIntent(
        intent_id="demo-shadow-intent",
        strategy_id="demo-only",
        model_version="demo-v1",
        instrument_id=instrument.instrument_id,
        venue=instrument.venue,
        asset_class=instrument.asset_class,
        side=OrderSide.BUY,
        notional=5_000,
        environment=ExecutionEnvironment.SHADOW,
        allowed_order_types=(OrderType.LIMIT,),
        expires_at=now + timedelta(minutes=1),
        max_price=101.5,
        created_at=now,
    )
    portfolio = PortfolioSnapshot(
        snapshot_at=now,
        equity=100_000,
        available_cash=100_000,
    )
    signer = _risk_signer()
    governor = RiskGovernor(
        RiskLimits(
            max_gross_notional=100_000,
            max_instrument_notional=10_000,
            max_venue_notional=30_000,
            asset_class_caps={AssetClass.MEMECOIN: 0, AssetClass.PREDICTION: 5_000},
            allow_live=False,
        ),
        signer,
    )
    approval = governor.approve(intent, instrument=instrument, portfolio=portfolio, now=now)
    adapter = PaperLedgerAdapter(ExecutionEnvironment.SHADOW)
    receipt = DeterministicExecutor(signer, adapter).execute(approval, now=now)
    audit.append_order_intent(intent)
    audit.append_risk_decision(approval.decision)
    audit.append_approval(approval)
    audit.append_execution_receipt(receipt)
    print(
        f"Shadow execution: intent={receipt.intent_id} status={receipt.status} "
        f"signed={signer.verify(approval)}"
    )
    print("No venue connection was used and no order was placed.")
    return 0


def _collect(path: Path, args: argparse.Namespace) -> int:
    store, _, _ = _initialize(path)
    collected_at = None
    if args.venue == "kalshi":
        collector = KalshiCollector()
        if args.dataset == "markets":
            batch = collector.collect_markets(
                collected_at=collected_at,
                status=args.status,
                limit=args.limit,
                cursor=args.cursor,
            )
        elif args.dataset == "trades":
            batch = collector.collect_trades(
                collected_at=collected_at,
                ticker=args.symbol,
                limit=args.limit,
                cursor=args.cursor,
            )
        elif args.dataset == "book" and args.symbol:
            batch = collector.collect_orderbook(
                args.symbol, collected_at=collected_at, depth=min(args.limit, 100)
            )
        else:
            raise ValueError("Kalshi supports markets, trades, or book with --symbol")
    elif args.venue == "coinbase":
        collector = CoinbaseCollector()
        if args.dataset == "products":
            batch = collector.collect_products(
                collected_at=collected_at, limit=args.limit, cursor=args.cursor
            )
        elif args.dataset == "book" and args.symbol:
            batch = collector.collect_product_book(
                args.symbol, collected_at=collected_at, limit=args.limit
            )
        elif args.dataset == "candles" and args.symbol:
            batch = collector.collect_candles(
                args.symbol,
                collected_at=collected_at,
                granularity=args.granularity,
                limit=args.limit,
            )
        else:
            raise ValueError("Coinbase supports products, book, or candles with --symbol")
    else:
        if args.dataset not in {"chain", "bars"} or not args.symbol:
            raise ValueError("Alpaca supports chain or bars with --symbol")
        key_id = os.getenv("ALPACA_MARKET_DATA_KEY_ID", "")
        secret_key = os.getenv("ALPACA_MARKET_DATA_SECRET_KEY", "")
        if args.dataset == "chain":
            collector = AlpacaOptionsCollector(key_id, secret_key)
            batch = collector.collect_chain(
                args.symbol,
                collected_at=collected_at,
                feed=args.feed,
                limit=args.limit,
                page_token=args.cursor,
            )
        else:
            stock_collector = AlpacaStockCollector(key_id, secret_key)
            batch = stock_collector.collect_daily_bars(
                args.symbol,
                collected_at=collected_at,
                feed=args.stock_feed,
                lookback_days=args.lookback_days,
                limit=args.limit,
                page_token=args.cursor,
            )
    instrument_count, event_count = store.append_batch(batch)
    receipt_time = (
        max(event.available_at for event in batch.events) if batch.events else utc_now()
    )
    print(
        f"Collected {instrument_count} instruments and {event_count} new events "
        f"from {batch.venue} at {receipt_time.isoformat()}"
    )
    if batch.cursor:
        print(f"Next cursor: {batch.cursor}")
    for diagnostic in batch.diagnostics:
        print(f"{diagnostic.severity.value}: {diagnostic.code.value}: {diagnostic.message}")
    return 0


def _shadow_cycle(path: Path, plan_path: Path, *, validate_only: bool = False) -> int:
    plan = load_plan(plan_path)
    if validate_only:
        for job in plan.jobs:
            if not job.enabled:
                state = "disabled"
            elif job.missing_activation_environment():
                state = f"waiting for {job.activation_profile}"
            else:
                state = "enabled"
            print(f"{job.job_id}: {job.venue}/{job.dataset} {state}")
        return 0
    store, _, audit = _initialize(path)
    ledger = IngestionRunLedger(path)
    records = ShadowIngestionRunner(store, ledger, audit=audit).run_plan(plan)
    for record in records:
        print(
            f"{record.job_id}: {record.status.value} "
            f"instruments={record.instruments_seen} events={record.events_inserted} "
            f"diagnostics={len(record.diagnostics)}"
        )
        if record.error_type:
            print(f"  {record.error_type}: {record.error_message}")
        if record.request_cursor:
            print(f"  request_cursor={record.request_cursor}")
        if record.next_cursor:
            print(f"  next_cursor={record.next_cursor}")
    research = ShadowResearchRunner(store, audit).run(as_of=utc_now())
    _print_shadow_research(research)
    report, locked = checkpointed_walk_forward_report(audit, as_of=utc_now())
    _print_locked_decisions(locked)
    _print_shadow_report(report, min_outcomes=30)
    failed = any(record.status is IngestionRunStatus.FAILED for record in records)
    return 1 if failed or research.generation.errors or research.scoring.errors else 0


def _shadow_research(path: Path) -> int:
    store, _, audit = _initialize(path)
    result = ShadowResearchRunner(store, audit).run(as_of=utc_now())
    _print_shadow_research(result)
    report, locked = checkpointed_walk_forward_report(audit, as_of=utc_now())
    _print_locked_decisions(locked)
    _print_shadow_report(report, min_outcomes=30)
    return 1 if result.generation.errors or result.scoring.errors else 0


def _print_locked_decisions(decisions: tuple[EvaluationDecision, ...]) -> None:
    for decision in decisions:
        print(
            f"decision locked: {decision.specialist_id}/{decision.kind.value} "
            f"scope={decision.scope} status={decision.status.value} "
            f"outcomes={decision.independent_outcomes}/{decision.boundary}"
        )


def _print_shadow_research(result: ShadowResearchResult) -> None:
    generation = result.generation
    scoring = result.scoring
    print(
        "shadow forecasts: "
        f"candidates={generation.candidates} new={generation.appended} "
        f"existing={generation.existing} skipped={generation.skipped}"
    )
    print(
        "shadow scores: "
        f"unscored={scoring.unscored} not_due={scoring.not_due} "
        f"due_unmatched={scoring.due_unmatched} quarantined={scoring.quarantined} "
        f"matched={scoring.matched} "
        f"new={scoring.appended} existing={scoring.existing}"
    )
    if scoring.next_due_at is not None:
        print(f"  next_due_at={scoring.next_due_at.isoformat()}")
    if scoring.oldest_due_at is not None:
        print(f"  oldest_due_at={scoring.oldest_due_at.isoformat()}")
    for error in (*scoring.errors, *generation.errors):
        print(f"  research error: {error}")


def _shadow_report(path: Path, *, min_outcomes: int) -> int:
    _, _, audit = _initialize(path)
    report = locked_walk_forward_report(
        audit,
        EvaluationGateConfig(min_independent_outcomes=min_outcomes),
    )
    _print_shadow_report(report, min_outcomes=min_outcomes)
    return 0


def _print_shadow_report(report: WalkForwardReport, *, min_outcomes: int) -> None:
    print(
        f"walk-forward gate: min_outcomes={min_outcomes} "
        f"familywise_alpha={report.familywise_alpha:.3f} "
        f"tests={report.confidence_tests}"
    )
    if not report.groups:
        print("no scoreable forecasts recorded")
        return
    for group in report.groups:
        print(
            f"{group.specialist_id}/{group.kind.value}: {group.status.value} "
            f"outcomes={group.independent_outcomes}/{min_outcomes} "
            f"raw_scores={group.raw_scores} forecasts={group.forecasts} "
            f"instruments={group.unique_instruments}"
        )
        if group.locked_status is not None:
            print(
                f"  decision: locked {group.locked_status.value} at "
                f"{group.locked_outcomes} outcomes on "
                f"{group.locked_at.isoformat()}; "
                f"monitoring={group.monitoring_status.value}"
            )
        if group.mean_loss is not None:
            print(
                "  paired: "
                f"loss={group.mean_loss:.8g} "
                f"benchmark={group.mean_benchmark_loss:.8g} "
                f"improvement={group.mean_improvement:.8g} "
                f"lower_bound={group.lower_confidence_bound:.8g} "
                f"win_rate={group.win_rate:.1%}"
            )
            print(
                "  controls: "
                f"delayed={_optional_number(group.delayed_control_improvement)} "
                f"shuffled={_optional_number(group.shuffled_control_improvement)}"
            )
        for reason in group.reasons:
            print(f"  gate: {reason}")
        for cohort in report.cohorts:
            evaluation = cohort.evaluation
            if (
                evaluation.specialist_id != group.specialist_id
                or evaluation.kind is not group.kind
            ):
                continue
            print(
                f"  cohort {cohort.dimension.value}={cohort.label}: "
                f"{evaluation.status.value} "
                f"outcomes={evaluation.independent_outcomes}/{min_outcomes} "
                f"raw_scores={evaluation.raw_scores} forecasts={evaluation.forecasts}"
            )
            if evaluation.locked_status is not None:
                print(
                    f"    decision: locked {evaluation.locked_status.value} at "
                    f"{evaluation.locked_outcomes} outcomes on "
                    f"{evaluation.locked_at.isoformat()}; "
                    f"monitoring={evaluation.monitoring_status.value}"
                )
            if evaluation.mean_improvement is not None:
                print(
                    "    paired: "
                    f"improvement={evaluation.mean_improvement:.8g} "
                    f"lower_bound={evaluation.lower_confidence_bound:.8g} "
                    f"win_rate={evaluation.win_rate:.1%}"
                )
            for reason in evaluation.reasons:
                print(f"    gate: {reason}")


def _optional_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8g}"


def _shadow_health(path: Path, args: argparse.Namespace) -> int:
    _initialize(path)
    report = ingestion_health(
        path,
        load_plan(args.plan),
        as_of=utc_now(),
        max_age=timedelta(minutes=args.max_age_minutes),
        max_consecutive_failures=args.max_consecutive_failures,
    )
    print(render_health(report, args.format))
    return 0 if report.healthy else 1


def _snapshot(path: Path, output: Path) -> int:
    summary = create_verified_snapshot(path, output)
    print(f"Snapshot: {summary.output_path}")
    print(f"Bytes: {summary.bytes_written}")
    print(f"SHA-256: {summary.sha256}")
    print(
        f"Verified: events={summary.events} audit_records={summary.audit_records} "
        f"ingestion_runs={summary.ingestion_runs} paper_records={summary.paper_records} "
        f"paper_ready={str(summary.paper_control_ready).lower()}"
    )
    return 0


def _economic_report(path: Path, args: argparse.Namespace) -> int:
    _, _, audit = _initialize(path)
    forecasts = audit.forecasts()
    scores = audit.forecast_scores()
    forecast_report = locked_walk_forward_report(
        audit,
        EvaluationGateConfig(min_independent_outcomes=args.min_outcomes),
    )
    report = build_economic_report(
        forecasts,
        scores,
        forecast_report,
        load_cost_registry(args.costs),
        EconomicGateConfig(min_trades=args.min_trades),
    )
    print(
        f"economic gate: registry={report.cost_registry_version} "
        f"digest={report.cost_registry_digest} "
        f"familywise_alpha={report.familywise_alpha:.3f} "
        f"tests={report.confidence_tests}"
    )
    for item in report.evaluations:
        print(
            f"{item.specialist_id}/{item.kind.value}: {item.status.value} "
            f"trades={item.trades}/{args.min_trades} "
            f"eligible_forecasts={item.eligible_forecasts} "
            f"skipped={item.skipped_signals} "
            f"cost_model={item.cost_model_id or 'none'}"
        )
        if item.mean_net_return is not None:
            print(
                "  base: "
                f"gross={item.mean_gross_return:.8g} "
                f"cost={item.mean_assumed_cost:.8g} "
                f"net={item.mean_net_return:.8g} "
                f"lower_bound={_optional_number(item.net_lower_confidence_bound)} "
                f"win_rate={item.win_rate:.1%} "
                f"drawdown={item.max_full_notional_drawdown:.1%}"
            )
            print(
                "  doubled-cost: "
                f"net={item.doubled_cost_mean_return:.8g} "
                f"lower_bound={_optional_number(item.doubled_cost_lower_confidence_bound)} "
                f"win_rate={item.doubled_cost_win_rate:.1%} "
                f"drawdown={item.doubled_cost_max_full_notional_drawdown:.1%}"
            )
        for reason in item.reasons:
            print(f"  gate: {reason}")
    return 0


def _daily_scorecard(path: Path, args: argparse.Namespace) -> int:
    _initialize(path)
    scorecard = build_daily_scorecard(
        path,
        load_plan(args.plan),
        load_cost_registry(args.costs),
        as_of=utc_now(),
        min_outcomes=args.min_outcomes,
        min_trades=args.min_trades,
        max_age=timedelta(minutes=args.max_age_minutes),
        max_consecutive_failures=args.max_consecutive_failures,
    )
    rendered = render_scorecard(scorecard, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(render_scorecard(scorecard, "json") + "\n", encoding="utf-8")
    if args.emit_github_alerts:
        alerts = render_github_alerts(scorecard)
        if alerts:
            print(alerts)
    if args.fail_on_critical and scorecard.status is ScorecardStatus.CRITICAL:
        return 1
    return 0


def _paper_status(path: Path, args: argparse.Namespace) -> int:
    store, _, audit = _initialize(path)
    client = _paper_client()
    account = client.account()
    positions = client.positions()
    orders = client.orders(status="open")
    control = PaperControlStore(path).status()
    policy = load_paper_risk_config(args.policy)
    eligibility = candidate_eligibility(audit, load_cost_registry(args.costs), policy)
    print(f"Paper account: {account.account_id}")
    print(
        f"Status: {account.status} can_trade={str(account.can_trade).lower()} "
        f"equity={account.equity:.2f} buying_power={account.buying_power:.2f} "
        f"daily_return={account.daily_return:.2%}"
    )
    print(
        f"Control: enabled={str(control.enabled).lower()} "
        f"kill_switch={str(control.kill_switch_active).lower()} ready={str(control.ready).lower()}"
    )
    print(f"Remote positions: {len(positions)} open_orders: {len(orders)}")
    print(f"Eligible strategy families: {len(eligibility.candidates)}")
    print(f"Paper policy: {policy.version}")
    print(f"Known Alpaca instruments: {len([item for item in store.instruments() if item.venue == 'alpaca'])}")
    print("Live trading: structurally unavailable")
    return 0


def _paper_reconcile(path: Path) -> int:
    _, _, audit = _initialize(path)
    result = PaperReconciler(
        _paper_client(), PaperExecutionLedger(path), audit
    ).run()
    print(
        f"Reconciled: remote_orders={result.remote_orders} open_orders={result.open_orders} "
        f"new_order_events={result.order_events_added} "
        f"account_snapshot_added={str(result.account_snapshot_added).lower()}"
    )
    print(
        f"Consistency: {'clean' if result.clean else 'mismatch'} "
        f"missing_remote={len(result.missing_remote_client_order_ids)} "
        f"unexpected_remote={len(result.unexpected_remote_client_order_ids)}"
    )
    for item in result.missing_remote_client_order_ids:
        print(f"  missing remote client order: {item}")
    for item in result.unexpected_remote_client_order_ids:
        print(f"  unexpected remote client order: {item}")
    if not result.clean:
        PaperControlStore(path).activate_kill_switch(
            reason="automatic reconciliation mismatch"
        )
        print("Paper kill switch activated because reconciliation was not clean.")
    return 0 if result.clean else 1


def _paper_control(path: Path, args: argparse.Namespace) -> int:
    _initialize(path)
    controls = PaperControlStore(path)
    if args.action == "enable":
        status = controls.enable(confirmation=args.confirm, reason=args.reason)
    elif args.action == "disable":
        status = controls.disable(reason=args.reason)
    elif args.action == "kill":
        cancellation = (
            (lambda: _paper_client().cancel_open_orders())
            if args.cancel_open_orders
            else None
        )
        stopped = activate_paper_emergency_stop(
            controls,
            reason=args.reason,
            cancel_open_orders=cancellation,
        )
        status = stopped.control
        if args.cancel_open_orders:
            print(f"Cancel requests: {stopped.cancellation_requests}")
    elif args.action == "release":
        status = controls.release_kill_switch(
            confirmation=args.confirm, reason=args.reason
        )
    else:
        status = controls.status()
    print(
        f"Paper control: enabled={str(status.enabled).lower()} "
        f"kill_switch={str(status.kill_switch_active).lower()} "
        f"ready={str(status.ready).lower()} reason={status.reason}"
    )
    return 0


def _paper_plan(path: Path, args: argparse.Namespace) -> int:
    store, _, audit = _initialize(path)
    policy = load_paper_risk_config(args.policy)
    plan = AlpacaPaperAllocator(
        store,
        audit,
        _paper_client(),
        PaperControlStore(path),
        load_cost_registry(args.costs),
        policy,
    ).plan()
    print(
        f"Paper plan: intents={len(plan.intents)} equity={plan.account_equity:.2f} "
        f"daily_return={plan.account_daily_return:.2%}"
    )
    for intent in plan.intents:
        print(
            f"  {intent.intent_id} {intent.side.value} {intent.quantity:g} "
            f"{intent.instrument_id} notional={intent.notional:.2f} "
            f"forecast={intent.forecast_id}"
        )
    for reason in plan.skipped:
        print(f"  blocked: {reason}")
    print("Preview only; no orders were placed.")
    return 0


def _paper_cycle(path: Path, args: argparse.Namespace) -> int:
    store, _, audit = _initialize(path)
    submission_enabled = False
    if args.execute:
        if args.confirm != "PAPER-ONLY":
            raise PermissionError("paper execution requires --confirm PAPER-ONLY")
        if os.getenv("ALPACA_PAPER_TRADING_ENABLED", "").lower() != "true":
            raise PermissionError(
                "paper execution requires ALPACA_PAPER_TRADING_ENABLED=true"
            )
        submission_enabled = True
    signer = _paper_signer() if submission_enabled else _risk_signer()
    policy = load_paper_risk_config(args.policy)
    result = PaperExecutionService(
        store,
        audit,
        _paper_client(),
        PaperControlStore(path),
        load_cost_registry(args.costs),
        signer,
        _paper_risk_limits(),
        config=policy,
        submission_enabled=submission_enabled,
    ).run()
    print(
        f"Paper cycle: planned={len(result.plan.intents)} "
        f"submitted={len(result.receipts)} rejected={len(result.rejected)}"
    )
    for receipt in result.receipts:
        print(
            f"  submitted intent={receipt.intent_id} status={receipt.status} "
            f"order_id={receipt.venue_order_id}"
        )
    for reason in result.rejected:
        print(f"  blocked: {reason}")
    if not submission_enabled:
        print("Preview only; no orders were placed.")
    return 0 if not result.rejected or not submission_enabled else 1


def _paper_drill(args: argparse.Namespace) -> int:
    report = run_paper_drills(args.scenario)
    rendered = render_paper_drill_report(report, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.successful else 1


def _crypto_sandbox(args: argparse.Namespace) -> int:
    config = load_crypto_sandbox_config(args.policy)
    report = run_crypto_sandbox_scenarios(args.scenario, config=config)
    rendered = render_crypto_sandbox_report(report, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.successful else 1


def _prediction_sandbox(args: argparse.Namespace) -> int:
    config = load_prediction_sandbox_config(args.policy)
    report = run_prediction_sandbox_scenarios(args.scenario, config=config)
    rendered = render_prediction_sandbox_report(report, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.successful else 1


def _options_sandbox(args: argparse.Namespace) -> int:
    config = load_options_sandbox_config(args.policy)
    report = run_options_sandbox_scenarios(args.scenario, config=config)
    rendered = render_options_sandbox_report(report, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.successful else 1


def _memecoin_sandbox(args: argparse.Namespace) -> int:
    config = load_memecoin_sandbox_config(args.policy)
    report = run_memecoin_sandbox_scenarios(args.scenario, config=config)
    rendered = render_memecoin_sandbox_report(report, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.successful else 1


def _launch_readiness(path: Path, args: argparse.Namespace) -> int:
    report = build_launch_readiness_report(
        path,
        plan_path=args.plan,
        costs_path=args.costs,
        config_path=args.policy,
        as_of=utc_now(),
    )
    rendered = render_launch_readiness_report(report, args.format)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not report.technical_successful:
        return 1
    if args.require_paper_review and report.status is not LaunchReadinessStatus.PAPER_REVIEW:
        return 1
    return 0


def main() -> int:
    args = _parser().parse_args()
    path = Path(args.db)
    try:
        if args.command == "init":
            _initialize(path)
            print(f"Initialized {path.resolve()}")
            return 0
        if args.command == "skills":
            for skill in SKILLS:
                print(f"{skill.name}: {skill.purpose} [{', '.join(skill.permissions)}]")
            return 0
        if args.command == "demo":
            return _demo(path)
        if args.command == "collect":
            return _collect(path, args)
        if args.command == "research-init":
            _, registry, _ = _initialize(path)
            for hypothesis in BASELINE_HYPOTHESES:
                registry.register_hypothesis(hypothesis)
                print(f"Registered {hypothesis.hypothesis_id}")
            return 0
        if args.command == "readiness":
            store, _, _ = _initialize(path)
            for item in data_readiness(
                store,
                as_of=utc_now(),
                perpetual_symbol=args.perpetual,
                spot_symbol=args.spot,
            ):
                state = "ready" if item.ready else "collecting"
                print(
                    f"{item.specialist}: {state} observations={item.observations} "
                    f"needs={item.requirement}"
                )
            return 0
        if args.command == "shadow-cycle":
            return _shadow_cycle(
                path, Path(args.plan), validate_only=args.validate_only
            )
        if args.command == "shadow-research":
            return _shadow_research(path)
        if args.command == "shadow-report":
            return _shadow_report(path, min_outcomes=args.min_outcomes)
        if args.command == "shadow-health":
            return _shadow_health(path, args)
        if args.command == "snapshot":
            return _snapshot(path, Path(args.output))
        if args.command == "economic-report":
            return _economic_report(path, args)
        if args.command == "daily-scorecard":
            return _daily_scorecard(path, args)
        if args.command == "paper-status":
            return _paper_status(path, args)
        if args.command == "paper-reconcile":
            return _paper_reconcile(path)
        if args.command == "paper-control":
            return _paper_control(path, args)
        if args.command == "paper-plan":
            return _paper_plan(path, args)
        if args.command == "paper-cycle":
            return _paper_cycle(path, args)
        if args.command == "paper-drill":
            return _paper_drill(args)
        if args.command == "crypto-sandbox":
            return _crypto_sandbox(args)
        if args.command == "prediction-sandbox":
            return _prediction_sandbox(args)
        if args.command == "options-sandbox":
            return _options_sandbox(args)
        if args.command == "memecoin-sandbox":
            return _memecoin_sandbox(args)
        if args.command == "launch-readiness":
            return _launch_readiness(path, args)
        if args.command == "doctor":
            store, _, audit = _initialize(path)
            with store.connect() as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                event_count = connection.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]
            audit_count = audit.verify_integrity()
            ingestion_count = IngestionRunLedger(path).verify_integrity()
            paper_count = PaperExecutionLedger(path).verify_integrity()
            paper_control = PaperControlStore(path).status()
            print(f"Database integrity: {integrity}")
            print(f"Stored events: {event_count}")
            print(f"Verified audit records: {audit_count}")
            print(f"Verified ingestion runs: {ingestion_count}")
            print(f"Verified paper reconciliation records: {paper_count}")
            print(
                f"Paper execution: enabled={str(paper_control.enabled).lower()} "
                f"kill_switch={str(paper_control.kill_switch_active).lower()}"
            )
            print("Live execution: disabled by default")
            print(
                "Observation collectors: Kalshi, Coinbase books/candles, "
                "Alpaca options and stock bars"
            )
            print("External execution adapters: Alpaca paper only (default locked)")
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
