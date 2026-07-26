# Multi-market research system

Foundation for researching stocks, listed options, futures, perpetual futures, spot crypto, memecoins, and prediction markets without giving research agents direct control of money.

Milestones 16–18 complete the engineering roadmap with isolated options lifecycle and delta-hedging scenarios, a fail-closed memecoin token/liquidity safety gate, and one unified launch-readiness report. Completion means the controls are built and tested—not that an edge exists. The strongest launch result is `PAPER_REVIEW`; live trading remains structurally unavailable.

Project build: `[##################] 18/18 (100%)`

## Implemented

- Canonical instrument definitions for seven asset classes.
- Immutable market events with separate event and availability timestamps.
- SQLite point-in-time queries that prevent replay from seeing future information.
- A hypothesis and experiment registry that counts every trial, including rejected work.
- Structured specialist forecasts with evidence IDs and invalidation conditions.
- Replay validation that rejects unavailable evidence or mismatched agent output.
- Portfolio limits and an independent risk governor.
- Short-lived HMAC-signed order approvals.
- An executor that rejects tampering, expiry, duplicates, and environment mismatches.
- Shadow/paper ledger adapter with no network or venue access.
- Project-local foundational agent skills.
- One append-only audit chain for forecasts, intents, risk decisions, signed approvals, and execution receipts.
- Host-pinned, HTTPS-only venue transport that only exposes GET requests and refuses redirects.
- Read-only collectors for Kalshi markets/trades/books/settlements, Coinbase spot/perpetual products/books/completed candles, and Alpaca option chains and stock bars.
- Venue timestamp normalization with explicit `available_at` boundaries.
- Diagnostics for stale data, source clock drift, crossed or empty books, invalid values, sequence gaps, and indicative feeds.
- Versioned, pre-registered hypotheses for perpetual funding/basis, options implied-volatility state, one-to-eight-hour and fast-settling prediction-market calibration, hourly crypto range breakouts, and fifteen-minute crypto momentum.
- Related-instrument replay for basis and historical calibration without weakening evidence checks.
- Deterministic research specialists that emit forecasts only and refuse insufficient or stale inputs.
- Benchmark-relative Brier, log-loss, funding-rate, volatility, and forward-return scoring.
- Validated shadow-ingestion plans, non-overlapping run locks, and append-only job records.
- Append-only cursor checkpoints that advance only after successful storage and preserve the requested and returned page tokens for audit.
- Per-job health gates for missing, stale, future-dated, and consecutively failing scheduled runs.
- Atomic SQLite online backups verified against database, event, forecast-audit, and ingestion-run integrity before publication.
- A deployment workflow with serialized runs, state continuity, opt-in seven-day recovery artifacts, manual restoration, and run summaries.
- Public Kalshi market listings normalized into executable top-of-book snapshots and finalized outcome labels.
- Coinbase FCM products classified by funding mechanism and contract behavior, including long-dated U.S. perpetuals.
- Raw, feed-tagged Alpaca equity bars for point-in-time realized-volatility measurement.
- Readiness reporting that refuses repeated polls as independent funding observations and rejects post-settlement books as calibration labels.
- Automatic, capped shadow candidate selection for perpetual funding, option volatility, and prediction-market calibration.
- Deterministic forecast deduplication and outcome matching that cannot score a label before it became available.
- Next-distinct-period funding scores, full-horizon option volatility scores, and public-settlement prediction scores.
- Outcome-clustered walk-forward reports that keep repeated forecasts for one result from inflating sample size.
- Benchmark-relative loss, paired win rate, serial-correlation-aware confidence bounds, and family-wise error control.
- Delayed and deterministically shuffled forecast controls; a baseline cannot become an edge candidate unless it beats both.
- Fixed forecast and outcome horizon cohorts (`<=1h`, `1h-8h`, `8h-1d`, `1d-7d`, `7d-30d`, and `>30d`) included in family-wise error control.
- Aggregate candidate gating that remains collecting for any observed underpowered duration cohort and rejects when any mature duration cohort fails.
- A versioned economic cost registry with dated source URLs and strict specialist/score-family binding.
- Candidate-only payoff replay for spot/crypto returns, prediction contracts, and hedged perpetual funding.
- Base and doubled-cost net returns with family-wise confidence bounds, win-rate gates, and full-notional drawdown diagnostics.
- Fail-closed handling for missing cost models, missing executable fields, and forecast types without defensible payoff mappings.
- A fixed-parameter twenty-bar breakout specialist that only emits on completed bars with adequate volume and predicts one source bar ahead.
- A fixed-parameter fifteen-minute crypto momentum specialist that uses eight completed bars, predicts exactly one bar ahead, and clusters every asset sharing a target time into one market-wide outcome.
- Semantic candle deduplication that preserves first receipt time while storing later venue revisions as new immutable versions.
- A host-pinned Alpaca paper client that cannot address the live Trading API host.
- Read-only paper account, buying-power, position, and order synchronization.
- Restart-safe client order IDs and remote idempotency checks before submission.
- Persistent default-locked paper controls with an independent kill switch and append-only control history.
- Append-only paper account snapshots and order-event reconciliation with mismatch detection.
- Candidate-only allocation linked to an exact forecast, model version, instrument, and point-in-time evidence set.
- Daily-loss, stale-data, per-trade, per-plan, instrument, venue, and asset-class limits.
- Three execution interlocks: persistent control state, an environment flag, and explicit CLI confirmation.
- Eight deterministic paper incident scenarios that exercise the real adapter, executor, control, reconciliation, and snapshot interfaces using isolated fakes.
- Text, Markdown, and JSON drill reports with scenario-level verified behaviors.
- A credential-free manual GitHub workflow that publishes the complete drill report as a run summary and retained artifact.
- A deterministic spot crypto and perpetual ledger with fees, realized and unrealized P&L, leverage, initial and maintenance margin, and funding settlement.
- Fail-closed reduce-only, post-only, stale-market, maximum-leverage, minimum-notional, signed-notional, and strategy-eligibility controls.
- Deterministic cross-margin liquidation and an integrity-checked append-only event trail with restart-safe intent identity checks.
- Eight credential-free sandbox scenarios and a dedicated cloud workflow that verify execution behavior without contacting a venue.
- A Decimal-based prediction ledger with complementary Yes/No executable prices, long-only inventory, realized P&L, and conservative open-cost limits.
- Versioned July 7, 2026 taker and maker fee formulas with centicent rounding and explicit per-series fee multipliers.
- Determined, disputed, and amended markets that remain nonpayable until finalization; binary and rule-defined scalar settlements pay with cent rounding and zero settlement fee.
- Signed, expiring prediction approvals, duplicate-order and conflicting-settlement protection, and an integrity-checked event trail.
- Eight offline prediction settlement scenarios plus a dedicated cloud workflow with credentials, network access, and real orders fixed at zero.
- A standard American equity-option lifecycle ledger covering premium fills, delta hedges, exercise, do-not-exercise instructions, expiry, and buying-power risk sellouts.
- Eight credential-free options scenarios with fees, slippage, stale-data gates, position caps, and an integrity-checked event trail.
- A memecoin safety evaluator that blocks dangerous token authorities and extensions, unverified programs, concentrated holders, young or thin pools, excessive costs, price divergence, and failed round-trip simulations.
- Shadow-only memecoin intents capped at $500, with two independent gates and no wallet, transaction-signing, or venue code path.
- A unified 18-milestone readiness report combining incident drills, every market sandbox, database integrity, ingestion health, evidence, after-cost eligibility, locked paper controls, and a live-intent rejection probe.
- A final cloud workflow that publishes all three safety reports while keeping credentials, network use, signed transactions, and real orders at zero.

## Safety boundary

```text
Untrusted external data
        ↓
Point-in-time store
        ↓
Research specialists (no credentials)
        ↓ structured forecasts
Independent validation
        ↓ approved target
Risk governor (may reject or reduce)
        ↓ signed, expiring intent
Deterministic executor
        ↓
Shadow ledger or eligibility-gated paper adapter
```

Research agents cannot call execution adapters. Execution adapters cannot browse, interpret news, select strategies, or change risk limits.

## Alpaca paper operations

Paper support is installed but locked by default. The Trading API client is pinned to
`paper-api.alpaca.markets`; constructing it with the live host is rejected. The existing
market-data credentials can be reused locally, or explicit `ALPACA_PAPER_KEY_ID` and
`ALPACA_PAPER_SECRET_KEY` aliases can be set.

Read account state and reconcile remote orders without submitting anything:

```bash
trading-bot --db var/trading.db paper-status
trading-bot --db var/trading.db paper-reconcile
trading-bot --db var/trading.db paper-plan
trading-bot --db var/trading.db paper-cycle
```

The checked-in policy at `config/paper-execution.json` risks 0.25% of paper equity per
eligible trade, caps a complete plan at 5%, stops after a 1% daily paper loss, rejects
quotes older than 20 minutes and stock bars older than two days, and retains the 30
independent-outcome and 30 after-cost-trade gates.

Do not unlock submission while a strategy is collecting or rejected. When a strategy
eventually clears both gates, paper submission still requires all of the following:

```bash
export ALPACA_PAPER_TRADING_ENABLED=true
export RISK_SIGNING_KEY="a-long-random-secret-from-your-secret-store"
trading-bot --db var/trading.db paper-control release \
  --confirm PAPER-ONLY --reason "approved paper trial"
trading-bot --db var/trading.db paper-control enable \
  --confirm PAPER-ONLY --reason "approved paper trial"
trading-bot --db var/trading.db paper-cycle --execute --confirm PAPER-ONLY
```

Emergency stop first locks future submissions, then optionally requests cancellation of
all open paper orders:

```bash
trading-bot --db var/trading.db paper-control kill \
  --reason "operator emergency stop" --cancel-open-orders
```

The scheduled shadow workflow performs read-only paper reconciliation but never calls the
paper submission command and does not set the paper enable flag.

## Paper incident drills

Run every scenario locally without loading an Alpaca key or contacting any venue:

```bash
trading-bot paper-drill --scenario all
trading-bot paper-drill --scenario all \
  --format markdown --output var/reports/paper-drills.md
trading-bot paper-drill --scenario ambiguous-timeout --format json
```

The complete suite verifies restart-safe duplicate handling, ambiguous acceptance recovery,
partial-fill preservation, terminal rejection idempotency, stale-data rejection,
reconciliation shutdown, daily-loss shutdown, emergency cancellation, and locked snapshot
recovery. The `Paper incident drills` GitHub workflow runs the same credential-free suite on
demand and retains its report for 30 days.

## Crypto and perpetual sandbox

Run the complete simulator without an exchange account, API key, or network connection:

```bash
trading-bot crypto-sandbox --scenario all
trading-bot crypto-sandbox --scenario all \
  --format markdown --output var/reports/crypto-sandbox.md
trading-bot crypto-sandbox --scenario liquidation --format json
```

The policy in `config/crypto-sandbox.json` starts with $100,000 of simulated cash,
defaults perpetual positions to 3x leverage, caps leverage at 5x, charges 10 bps per
fill plus 5 bps of slippage, rejects market state older than 60 seconds, and liquidates
cross-margin perpetual exposure at the configured maintenance threshold. Spot positions
are unlevered and cannot be short.

The adapter is disabled by default and separately requires a strategy-eligibility flag.
The included scenarios open those gates only inside an isolated deterministic test. They
verify spot fills, perpetual margin, reduce-only behavior, funding idempotency, liquidation,
stale-data rejection, independent gates, and post-only protection. This is an in-memory
execution simulator, not a venue connection or evidence that a strategy has an edge.

## Prediction settlement sandbox

Run the complete prediction lifecycle without an exchange account, API key, or network:

```bash
trading-bot prediction-sandbox --scenario all
trading-bot prediction-sandbox --scenario all \
  --format markdown --output var/reports/prediction-sandbox.md
trading-bot prediction-sandbox --scenario lifecycle --format json
```

The policy at `config/prediction-sandbox.json` pins its fee schedule to July 7, 2026
and links the official fee and settlement sources. It models Yes buys at the observed
Yes ask, No buys at the complement of the Yes bid, matched-order fees rounded up to a
centicent, and settlement payouts rounded to cents. General taker fees default on while
maker fees require the applicable series multiplier.

Orders require signed, expiring approvals and the adapter has separate runtime and
strategy-evidence gates that default closed. Trading is accepted only for fresh active
markets. Determined, disputed, and amended states cannot pay; settlement requires a
finalized market plus an identified public source event. Standard Yes/No settlement pays
the winner $1 per contract. Nonbinary outcomes must arrive as an explicit rule-defined
scalar value—the simulator does not invent a generic void or refund rule. If eligibility
closes after a simulated fill, new orders stop but a finalized settlement can still close
the existing position rather than trapping it.

The eight scenarios verify Yes and No payouts, scalar rounding, lifecycle disputes,
restart idempotency, conflicting-result rejection, maker/taker fees, stale and closed
markets, independent gates, and naked-short rejection. This remains an in-memory
simulator and does not establish that a prediction strategy has an edge.

## Options lifecycle sandbox

Run the option lifecycle and delta-hedging simulator without brokerage credentials or
network access:

```bash
trading-bot options-sandbox --scenario all
trading-bot options-sandbox --scenario all \
  --format markdown --output var/reports/options-sandbox.md
trading-bot options-sandbox --scenario delta-hedge --format json
```

The policy accepts only standard 100-share American equity options. It models
conservative option and stock slippage, premium and hedge-notional limits,
exercise-by-exception at $0.01 in the money, do-not-exercise, worthless expiry, and
simulated risk sellout when exercise resources are insufficient. This is a payoff and
operations sandbox, not an option strategy or proof of profitable volatility forecasting.

## Memecoin safety sandbox

Run the token and pool safety evaluator without a wallet, RPC connection, or signed
transaction:

```bash
trading-bot memecoin-sandbox --scenario all
trading-bot memecoin-sandbox --scenario all \
  --format markdown --output var/reports/memecoin-sandbox.md
trading-bot memecoin-sandbox --scenario sell-simulation --format json
```

The policy requires verified source metadata, revoked dangerous authorities and token
extensions, at least $100,000 of pool liquidity, a seven-day pool history, bounded
holder concentration, costs and price divergence, plus successful buy and sell
simulations recovering at least 90% of expected value. Passing creates only a capped
shadow intent. It never creates a wallet instruction and is not evidence of an edge.

## Unified launch readiness

Evaluate the complete roadmap against the current research database:

```bash
trading-bot --db var/trading.db launch-readiness --format markdown
trading-bot --db var/trading.db launch-readiness \
  --require-paper-review --output var/reports/launch-readiness.txt
```

The report distinguishes engineering completion from trading readiness. A fresh or
underpowered database reports `NO_GO` even when every deterministic safety suite passes.
`PAPER_REVIEW` requires current ingestion, qualifying forecast and after-cost candidates,
clean ledgers, locked paper controls, and every incident and sandbox suite passing. No
status enables live trading or places an order.

## Run locally

Python 3.11 or newer is required. The current foundation has no third-party runtime dependencies.

```bash
cd trading-bot
python -m venv .venv
source .venv/bin/activate
pip install -e .
trading-bot init
trading-bot skills
trading-bot demo
trading-bot doctor
trading-bot research-init
trading-bot --db var/trading.db readiness
trading-bot --db var/trading.db shadow-research
trading-bot --db var/trading.db shadow-report
trading-bot --db var/trading.db economic-report --costs config/economic-costs.json
trading-bot --db var/trading.db shadow-health --plan config/shadow-ingestion.json
trading-bot --db var/trading.db daily-scorecard --plan config/shadow-ingestion.json --format markdown
trading-bot --db var/trading.db snapshot --output var/snapshots/trading.db
trading-bot paper-drill --scenario all
trading-bot crypto-sandbox --scenario all
trading-bot prediction-sandbox --scenario all
trading-bot options-sandbox --scenario all
trading-bot memecoin-sandbox --scenario all
trading-bot --db var/trading.db launch-readiness --format markdown
```

The demo creates deterministic delayed market events, replays two decision times, records the complete audit chain, signs one approved shadow intent, and records it without contacting a venue.

## Read-only collection

Kalshi and Coinbase use public market-data endpoints:

```bash
trading-bot collect kalshi markets --status open --limit 100
trading-bot collect kalshi trades --symbol KXTICKER --limit 100
trading-bot collect kalshi book --symbol KXTICKER --limit 100
trading-bot collect coinbase products --limit 100
trading-bot collect coinbase book --symbol BTC-USD --limit 100
trading-bot collect coinbase candles --symbol BTC-USD --granularity ONE_HOUR --limit 30
```

Alpaca options data requires market-data credentials. The collector sends them only to `data.alpaca.markets` and still has no order method:

```bash
export ALPACA_MARKET_DATA_KEY_ID="..."
export ALPACA_MARKET_DATA_SECRET_KEY="..."
trading-bot collect alpaca chain --symbol AAPL --feed indicative --limit 100
trading-bot collect alpaca bars --symbol AAPL --stock-feed iex --lookback-days 45
```

For GitHub Actions, store the same values as repository secrets named
`ALPACA_MARKET_DATA_KEY_ID` and `ALPACA_MARKET_DATA_SECRET_KEY`. The checked-in
SPY, QQQ, AAPL, and NVDA stock/options jobs automatically activate only when
both secrets are present. Secret values never enter the observation plan,
database, logs, artifacts, or scorecard.

Use `--cursor` to continue a one-off manual collection. Scheduled shadow jobs manage pagination automatically. The options plan pairs each broad, cursor-resuming chain crawl with a bounded rolling cohort collected every cycle. The cohort uses the latest point-in-time underlying close to select strikes within ten percent, expirations within fourteen days, and snapshots updated within two hours. It restarts pagination because its date and price bounds change over time. This preserves universe discovery while concentrating repeat observations in more relevant contracts; unchanged source quotes remain deduplicated, and the option specialist still requires three distinct observations. Every returned event is stored with collection time as its availability boundary. Venue text, including news and settlement rules, remains untrusted data.

## Shadow observation cycle

The checked-in public plan collects active Kalshi binary markets closing within 48 hours, public settlements and trades, Coinbase product definitions and funding, executable BTC and ETH spot/perpetual books, and hourly candles for twenty online USD markets: BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, LINK, LTC, BCH, HYPE, ZEC, XLM, ONDO, HBAR, NEAR, SUI, UNI, TAO, and PUMP. It also collects up to 350 completed fifteen-minute candles for BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, LINK, HYPE, and PUMP. This liquid intraday cohort can produce one independently clustered target block every fifteen minutes instead of waiting for daily outcomes; correlated assets at the same target time still count as only one outcome. The original breakout baseline is explicitly pinned to hourly candles, so adding faster data cannot alter its locked evaluation.

Stablecoin pairs are excluded. Multivariate Kalshi combinations are excluded because their scalar settlement does not match the pre-registered binary target. A bounded outcome job uses Kalshi's public exact-ticker filter to revisit every timing-valid unscored binary forecast, prioritizing due targets before future targets so publicly finalized early outcomes are captured promptly; already scored markets, combinations, and contracts without a trustworthy lifecycle time are skipped. Candidate selection covers unforecasted Kalshi events before revisiting any, chooses the tightest executable contract within each event, and rejects spreads above the pre-declared ten-cent limit. The frozen v3 cohort requires a parseable occurrence more than one and no more than eight hours after the point-in-time book; its existing forecasts remain scoreable. The preregistered v4 successor uses the same fixed inputs and parameters but emits only a genuine cohort adjustment backed by at least five independent resolved events, so benchmark-identical warm-up observations cannot dilute the adjusted strategy. Its bounded history allocates at most one slot per independent event and chooses the eligible contract closest to the current target probability, preventing large strike families from crowding out unrelated evidence. The forecast validity boundary stores occurrence time separately, and post-occurrence forecasts are never scored. Public settlement time controls when a label becomes available, while the pre-registered occurrence remains the scored target and evaluation horizon even when publication is delayed. Historical calibration books must satisfy the same 1–8 hour pre-occurrence window and spread limit, preventing post-start prices from leaking results into the cohort. This keeps every new prediction observation in one fixed `1h-8h` cohort.

The separate fast-settling baseline is not an adjustment and cannot reuse the frozen cohort. It accepts only active, binary Kalshi markets with `can_close_early=false`, a settlement timer of at most fifteen minutes, an executable spread no wider than ten cents, and a point-in-time `expected_expiration_time` more than 20 minutes and no more than two hours ahead. Kalshi documents expected expiration as the forecasted outcome time and warns that close times may change; finalization can also be delayed. We therefore store the observed expected expiration as a fixed forecast boundary, retain public settlement availability separately, and never infer prompt labels. Every event ticker supplies at most one candidate and one scored outcome, regardless of how many correlated strikes it contains. See Kalshi's [market lifecycle documentation](https://docs.kalshi.com/getting_started/market_lifecycle).

```bash
trading-bot shadow-cycle --plan config/shadow-ingestion.json --validate-only
trading-bot --db var/trading.db research-init
trading-bot --db var/trading.db shadow-cycle --plan config/shadow-ingestion.json
trading-bot --db var/trading.db readiness
```

The command runs exactly one cycle and exits, so an external scheduler controls frequency. After collection it scores any newly observable outcomes and emits fresh forecasts from eligible data. For paginated jobs, each successful or degraded run records both the requested cursor and the venue's next cursor. Resume-mode jobs advance to that page; a failed page is retried, and a terminal response with no next cursor restarts the crawl on page one. The bounded Alpaca option cohort jobs use explicit restart mode, so they request page one each cycle while still recording the returned next cursor for audit. Cursor checkpoints are recovered from the immutable run ledger rather than hidden mutable state and are capped at 4,096 characters.

Plans cannot contain credentials, job types are allowlisted, concurrent cycles are rejected, failures are isolated per job, and every result is recorded append-only. Rerunning against the same information set is idempotent.

`shadow-research` runs only the local scoring and forecast stages, with no network collection. Prediction-market generation is capped at 25 fresh candidates per cycle; invalid or non-executable books are skipped. Funding forecasts require distinct funding periods plus fresh spot and perpetual books. Option forecasts require enough recent contract observations.

`shadow-report` evaluates scored forecasts against their explicit benchmarks. It uses only the latest forecast for each instrument/outcome pair, requires 30 outcome clusters by default, widens uncertainty for serial dependence, and requires the model to beat delayed and shuffled controls. Crypto and perpetual candidates additionally require outcomes from at least two instruments with no single instrument contributing more than 80% of the independent sample, enforcing their preregistered cross-asset invalidation condition. Results are split by specialist/score family, forecast validity horizon, and realized target horizon using the fixed pre-declared buckets `<=1h`, `1h-8h`, `8h-1d`, `1d-7d`, `7d-30d`, and `>30d`. All populated overall and cohort tests share the family-wise confidence correction.

An aggregate cannot become a candidate while any observed horizon cohort has fewer than 30 independent outcomes. Once mature, every observed cohort must independently clear the benchmark, confidence, win-rate, delayed-control, and shuffled-control gates. The strongest label is `candidate`, never `proven`. Use `--min-outcomes` only for diagnostics; lowering it does not authorize paper or live allocation.

`economic-report` is a second, stricter gate and does no work for forecast families that are still collecting or rejected. Return forecasts use signed realized returns. Prediction forecasts buy the executable yes or no side, include the observed half-spread, and apply the general Kalshi taker-fee formula. Funding forecasts only map to a hedged carry trade when basis and funding agree and the forecast already clears its dynamic execution bound. Options volatility remains unsupported because an implied-volatility score alone cannot reconstruct a delta-hedged option P&L path.

The checked-in `config/economic-costs.json` uses a conservative 120 bps Coinbase retail taker round trip before additional spread, slippage, and latency; prediction fees use Kalshi's July 7, 2026 general schedule; perpetual forecasts use their point-in-time spot/perpetual execution bound. Every surviving trade is rerun at twice the complete assumed cost. Results are normalized research returns, not an allocation or portfolio-sizing recommendation.

Alpaca jobs use the `alpaca_market_data` activation profile. Without both read-only environment values they appear as `waiting_credentials`, remain health-neutral, and never construct a network collector. With both values present they become required health-gated jobs and collect paired option-chain snapshots and underlying stock bars.

The deployment-ready workflow is installed at `.github/workflows/shadow-ingestion.yml` and runs at minutes 7, 22, 37, and 52 of every hour once this project is in a GitHub repository. This leaves a seven-minute collection buffer after each fifteen-minute candle close. It restores the most recent cache, serializes overlapping runs, validates the plan, collects and researches, and publishes a combined operational scorecard. The scorecard reports market coverage, ingestion health, forecast evidence, per-strategy pending and overdue outcomes, prediction-calibration cohort readiness, cost-adjusted eligibility, and execution-audit counts. GitHub annotations flag failures, credentials waiting, and newly qualified candidates. Approximately once every 48 runs it retains the Markdown and JSON scorecards for 90 days and creates a verified SQLite snapshot retained for seven days. A manual run creates those artifacts only when `create_recovery_checkpoint` is selected. GitHub marks the workflow failed when any active job is missing, more than 90 minutes stale, or has a latest failed run.

For recovery, manually dispatch the workflow and enter the workflow run ID that owns the desired `shadow-database` artifact in `restore_run_id`. The downloaded snapshot is integrity-checked before collection continues. Select `create_recovery_checkpoint` only when that run should publish a new recovery snapshot; normal manual verification continues through the cache without duplicating a large artifact. The cache is continuity storage; the immutable run artifacts are the recovery path. For longer-lived or higher-volume research, move the same append-only model to managed durable storage before relying on it operationally.

## Research baselines

- Perpetual funding/basis: forecasts the next funding-rate state, calculates executable spot/perpetual basis, and places a conservative spread-and-fee bound around carry.
- Options volatility: forecasts short-horizon implied-volatility state from point-in-time contract quotes; indicative feeds receive a hard confidence cap.
- Prediction markets: starts from the executable yes/no midpoint and applies a shrunk calibration adjustment only when enough already-resolved related contracts exist.
- Crypto breakouts: detects closes outside the previous twenty completed hourly bars, requires nontrivial volume, and forecasts the next bar using a fixed 25% shrinkage of the range break. No parameter search is performed.
- Crypto intraday momentum: measures the average log return over eight completed fifteen-minute bars, applies a fixed 25% shrinkage and 1% forecast cap, and forecasts the next completed fifteen-minute bar. No parameter search is performed.

These are competence baselines, not proven edges. Each is compared with a simple benchmark and remains in research/shadow mode until held-out, net-cost validation passes.

## Test

```bash
python -m unittest discover -s tests -v
```

The tests deliberately attempt look-ahead evidence, premature outcome scoring, event and audit mutation, hypothesis redefinition, embedded plan credentials, venue symbol confusion, future source clocks, crossed books, privileged HTTP headers, host escape, stale specialist inputs, risk-limit violations, live execution, signature tampering, duplicate execution, altered intent reuse, conflicting prediction settlement, premature disputed-market payout, nonstandard option contracts, unsafe token authorities, non-finite sandbox values, ambiguous remote acceptance, cancellation failure, and locked snapshot recovery.

## Operational path after the roadmap

1. Accumulate at least 30 scored outcome clusters in every observed horizon cohort.
2. Keep every paper and sandbox strategy gate locked until its family passes the evidence and after-cost gates.
3. Require fresh passing incident-drill and sandbox reports before every execution-control change.
4. Use `launch-readiness --require-paper-review` as the fail-closed paper review gate.

Live venue adapters remain out of scope. Any future live phase requires separate operator design and approval after sustained paper validation; the current system cannot authorize it.
