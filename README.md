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
- Bounded public Solana token-profile discovery with raw-payload provenance; every discovered token remains safety-blocked until independent on-chain authority, holder, transfer-behavior, and round-trip evidence is observed.
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
- A separately preregistered fast-settlement prediction lane that clusters related Kalshi strikes by documented `event_ticker` and scores only public finalization timestamps inside each forecast's immutable label window.
- Fast-settlement prediction v11 requires an `active` binary market with an executable book and lifecycle-rule snapshot no more than fifteen minutes apart, recorded `close_time` 20 minutes to two hours ahead, a settlement timer of at most fifteen minutes, and a maximum ten-cent spread. It anchors the label window to the recorded `close_time`, not expected or latest expiration: same-event finalization may count from that close through the settlement timer plus one hour. An earlier finalization may count only if `can_close_early=true` was recorded at forecast time and the final response records a close time strictly after forecast generation and strictly before both finalization and the registered close. Delayed, missing, policy-inconsistent, and uncorroborated labels remain unscored and are never recategorized as evidence. The rules follow Kalshi's [market lifecycle documentation](https://docs.kalshi.com/getting_started/market_lifecycle) and [settlement documentation](https://docs.kalshi.com/getting_started/market_settlement).
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
- A separately preregistered v2 of that intraday lane that assigns each target-time cluster to one fixed Coinbase symbol with SHA-256 before signal evaluation, so no result can depend on which instrument arrived last; it never substitutes another symbol when the assigned signal is absent.
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

The scheduled shadow plan collects at most 25 public Solana token-profile discoveries per
cycle from Dexscreener. For that bounded discovery set it also makes one documented,
read-only batch request for the most-liquid public pool snapshot per token and at most 25
finalized mint-control account reads. Holder-concentration reads use a separate, conservative
batch of at most 10 mints per cycle (20 read-only RPC requests) to respect shared public RPC
capacity. The
reader structurally records legacy mint/freeze authorities and Token-2022 transfer controls
(including permanent delegates, transfer hooks, pausable/default-frozen, non-transferable, and
unknown extensions). It is restricted to Solana's documented `getMultipleAccounts`,
`getTokenLargestAccounts`, `getTokenSupply`, and `getSignaturesForAddress` methods, and rejects
transaction or signing methods. A holder concentration is recorded only when the two independently
returned finalized slots match; it retains counts and basis-point shares, never holder addresses.
Separately, a bounded holder-activity read samples public finalized transaction references for at
most two large token accounts per mint and retains aggregate counts only. It is negative evidence
only: it never proves transferability or clears a safety gate. These
observations cannot create a safety snapshot, forecast, shadow intent, wallet, signed
transaction, or venue order. Every observation is explicitly recorded as
`blocked_unverified` until separate, point-in-time authority, holder-concentration,
transfer-behavior, and simulated round-trip evidence exists. The policy's authority and
extension gates follow Solana's primary token documentation.

The shared public Solana endpoint is deliberately not used by the scheduled research plan:
it is rate-limited and cannot provide reliable prospective evidence. The two Solana safety-read
jobs plus the bounded holder-activity job use the `solana_read_only_rpc` activation profile and remain health-neutral
`waiting_credentials` jobs until an operator adds a dedicated **read-only** mainnet endpoint as
the GitHub Actions secret `SOLANA_READ_ONLY_RPC_URL`. The endpoint must be HTTPS and may include
the provider's path or query credential; its value is never rendered in logs. The transport
remains pinned to that endpoint's host, refuses redirects and userinfo, and permits only
`getMultipleAccounts`, `getTokenLargestAccounts`, `getTokenSupply`, and `getSignaturesForAddress`.
Configuring it does not
change any safety gate or grant wallet, signing, transaction, forecast, or order authority.

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

The checked-in public plan collects active Kalshi binary markets closing within 48 hours, public settlements and trades, Coinbase product definitions and funding, executable BTC and ETH spot/perpetual books, and hourly candles for twenty online USD markets: BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, LINK, LTC, BCH, HYPE, ZEC, XLM, ONDO, HBAR, NEAR, SUI, UNI, TAO, and PUMP. Its hourly archival pass also collects up to 350 completed fifteen-minute candles for BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, LINK, HYPE, and PUMP. The separate fifteen-minute rapid pass reads a fixed 32 completed bars for that same cohort: the preregistered specialist needs eight, so this provides a fixed operational buffer without repeatedly transferring the archival window. This liquid intraday cohort can produce one independently clustered target block every fifteen minutes instead of waiting for daily outcomes; correlated assets at the same target time still count as only one outcome. The original breakout baseline is explicitly pinned to hourly candles, so adding faster data cannot alter its locked evaluation.

Stablecoin pairs are excluded. Multivariate Kalshi combinations are excluded because their scalar settlement does not match the pre-registered binary target. A bounded outcome job uses Kalshi's public exact-ticker filter to revisit every timing-valid unscored binary forecast, prioritizing due targets before future targets so publicly finalized early outcomes are captured promptly; already scored markets, combinations, and contracts without a trustworthy lifecycle time are skipped. Candidate selection covers unforecasted Kalshi events before revisiting any, chooses the tightest executable contract within each event, and rejects spreads above the pre-declared ten-cent limit. The frozen v3 cohort requires a parseable occurrence more than one and no more than eight hours after the point-in-time book; its existing forecasts remain scoreable. The preregistered v4 successor uses the same fixed inputs and parameters but emits only a genuine cohort adjustment backed by at least five independent resolved events, so benchmark-identical warm-up observations cannot dilute the adjusted strategy. Its bounded history allocates at most one slot per independent event and chooses the eligible contract closest to the current target probability, preventing large strike families from crowding out unrelated evidence. The forecast validity boundary stores occurrence time separately, and post-occurrence forecasts are never scored. Public settlement time controls when a label becomes available, while the pre-registered occurrence remains the scored target and evaluation horizon even when publication is delayed. Historical calibration books must satisfy the same 1–8 hour pre-occurrence window and spread limit, preventing post-start prices from leaking results into the cohort. This keeps every new prediction observation in one fixed `1h-8h` cohort.

The separate fast-settling baseline is not an adjustment and cannot reuse the frozen cohort. It accepts only active, binary Kalshi markets with a documented boolean `can_close_early` policy, a settlement timer of at most fifteen minutes, an executable spread no wider than ten cents, and a point-in-time `expected_expiration_time` more than 20 minutes and no more than two hours ahead. The v9 lane records both true and false policies rather than silently treating an undocumented policy as safe: Kalshi says `can_close_early=true` allows `close_time` to move earlier, while expected expiration remains the forecasted outcome time and latest expiration is only the latest possible expiry. A finalization before expected expiration is therefore scoreable only when the forecast recorded `can_close_early=true` and the later finalization response's `close_time` is strictly between forecast generation and finalization; equal timestamps remain ambiguous and unscored. The recorded expected expiration is a fixed evaluation boundary, not a guarantee of prompt closure; eligible finalization is bounded by the pre-registered settlement-timer-plus-one-hour deadline, and late or missing labels remain unscored. The public list endpoint filters only on `close_time`, while expected expiration may precede close time, so the dedicated rapid job reads a bounded public page of 250 active non-MVE markets without a close-time filter; it resumes the documented cursor on the next cycle before the immutable local eligibility funnel applies the actual expected-expiration window. This fixed operational page budget keeps the fifteen-minute evidence cycle bounded without silently fixing the universe to its first page. Every event ticker supplies at most one candidate and one scored outcome, regardless of how many correlated strikes it contains. See Kalshi's [market lifecycle documentation](https://docs.kalshi.com/getting_started/market_lifecycle) and [public market-list reference](https://docs.kalshi.com/api-reference/market/get-markets).

```bash
trading-bot shadow-cycle --plan config/shadow-ingestion.json --validate-only
trading-bot --db var/trading.db research-init
trading-bot --db var/trading.db shadow-cycle --plan config/shadow-ingestion.json
trading-bot --db var/trading.db readiness
```

The command runs exactly one cycle and exits, so an external scheduler controls frequency. After collection it scores any newly observable outcomes and emits fresh forecasts from eligible data. Every ingestion record carries an observation origin: local and manually dispatched cycles are `manual`, while the production workflow marks only its `schedule` event cycles as `scheduled`. The rapid crypto and fast-prediction continuity gates count only `scheduled` records; manual/recovery and legacy records remain auditable but cannot fill a prospective evidence gap. For paginated jobs, each successful or degraded run records both the requested cursor and the venue's next cursor. Resume-mode jobs advance to that page; a failed page is retried, and a terminal response with no next cursor restarts the crawl on page one. The bounded Alpaca option cohort jobs use explicit restart mode, so they request page one each cycle while still recording the returned next cursor for audit. Cursor checkpoints are recovered from the immutable run ledger rather than hidden mutable state and are capped at 4,096 characters.

Plans cannot contain credentials, job types are allowlisted, concurrent cycles are rejected, failures are isolated per job, and every result is recorded append-only. Rerunning against the same information set is idempotent.

`shadow-research` runs only the local scoring and forecast stages, with no network collection. Prediction-market generation is capped at 25 fresh candidates per cycle; invalid or non-executable books are skipped. Funding forecasts require distinct funding periods plus fresh spot and perpetual books. Option forecasts require enough recent contract observations.

`shadow-report` evaluates scored forecasts against their explicit benchmarks. It uses only the latest forecast for each instrument/outcome pair, requires 30 outcome clusters by default, widens uncertainty for serial dependence, and requires the model to beat delayed and shuffled controls. Crypto and perpetual candidates additionally require outcomes from at least two instruments with no single instrument contributing more than 80% of the independent sample, enforcing their preregistered cross-asset invalidation condition. Results are split by specialist/score family, forecast validity horizon, and realized target horizon using the fixed pre-declared buckets `<=1h`, `1h-8h`, `8h-1d`, `1d-7d`, `7d-30d`, and `>30d`. All populated overall and cohort tests share the family-wise confidence correction.

An aggregate cannot become a candidate while any observed horizon cohort has fewer than 30 independent outcomes. Once mature, every observed cohort must independently clear the benchmark, confidence, win-rate, delayed-control, and shuffled-control gates. The strongest label is `candidate`, never `proven`. Use `--min-outcomes` only for diagnostics; lowering it does not authorize paper or live allocation.

`economic-report` is a second, stricter gate and does no work for forecast families that are still collecting or rejected. Return forecasts use signed realized returns. Prediction forecasts buy the executable yes or no side, include the observed half-spread, and apply the general Kalshi taker-fee formula. Funding forecasts only map to a hedged carry trade when basis and funding agree and the forecast already clears its dynamic execution bound. Options volatility remains unsupported because an implied-volatility score alone cannot reconstruct a delta-hedged option P&L path.

The checked-in `config/economic-costs.json` uses a conservative 120 bps Coinbase retail taker round trip before additional spread, slippage, and latency; prediction fees use Kalshi's July 7, 2026 general schedule; perpetual forecasts use their point-in-time spot/perpetual execution bound. Every surviving trade is rerun at twice the complete assumed cost. Results are normalized research returns, not an allocation or portfolio-sizing recommendation.

Alpaca jobs use the `alpaca_market_data` activation profile, while Solana memecoin safety reads use `solana_read_only_rpc`. Without their read-only environment values they appear as `waiting_credentials`, remain health-neutral, and never construct a network collector. With values present they become required health-gated jobs; Alpaca collects paired option-chain snapshots and underlying stock bars, and Solana collects the bounded on-chain safety observations.

The deployment-ready full-market workflow is installed at `.github/workflows/shadow-ingestion.yml` and runs at minute 7 of every hour. The separate `.github/workflows/rapid-shadow-ingestion.yml` runs the fixed fifteen-minute Coinbase candles, cursor-resuming fast Kalshi market page, and bounded prediction-outcome polling at minutes 22, 37, and 52; together they preserve the seven-minute post-close buffer. Both workflows serialize on the same lock and use the `SHADOW_DATABASE_URL` repository secret as a TLS-required pooled Neon PostgreSQL database. Durable evidence is isolated in the `shadow_evidence_v2` schema through a transaction-local search path compatible with the pooled endpoint, so cutover diagnostics in the default and v1 schemas cannot rewrite or contaminate the imported point-in-time history. Before any external collection, each workflow verifies that this durable store is reachable; it fails closed if it is not, so unpersisted observations cannot become evidence. The workflows refuse to run if the secret is absent; GitHub Actions caches and SQLite recovery artifacts are not used for prospective evidence continuity. The one-time manual `Migrate shadow persistence` workflow imports the last retained SQLite checkpoint into the locked evidence schema. Existing identical records are verified exactly, target-only observations are preserved, conflicts are refused, and every final immutable table count is reconciled. Once it passes, all prospective observation uses Neon only.

### Evidence retention and restore

The hot Neon tier is budgeted below 450 MB. It may eventually retain a rolling raw-data
window plus the aggregates required by every evidence gate, but hot-tier eviction is
disabled until the archive and restore proof below have succeeded. Audit records,
including `evaluation_decision` records, are permanent in Postgres and are never eligible
for eviction.

The manual `Archive immutable shadow evidence` workflow restores the retained SQLite
checkpoint, creates a consistent SQLite backup, verifies database and ledger integrity,
and writes a versioned JSON manifest. It compresses the complete snapshot with Zstandard,
splits assets at 1.9 GB if necessary, and records SHA-256 checksums. Only its final release
job receives `contents: write`; all preparation jobs remain read-only. Repository release
immutability must be enabled before dispatch. The job creates a draft, attaches every
asset, verifies the asset count, publishes the release, and fails unless GitHub reports
the published release as immutable.

Restore an archive with the manual `Verify shadow evidence restore` workflow and its
immutable `shadow-evidence-*` tag. It downloads release assets with `contents: read`,
verifies every checksum, rejoins split parts, verifies the decompressed snapshot SHA-256,
rebuilds a new SQLite database through the backup API, runs the complete database doctor,
and publishes a 90-day restore report. The equivalent local path is:

```bash
gh release download <shadow-evidence-tag> --dir archive
(cd archive && sha256sum --check SHA256SUMS)
cat archive/shadow-evidence.sqlite.zst.part-* > shadow-evidence.sqlite.zst  # split archive only
zstd --decompress shadow-evidence.sqlite.zst -o restored.sqlite
trading-bot --db restored.sqlite snapshot \
  --output rebuilt.sqlite \
  --json-output rebuilt-manifest.json
trading-bot --db rebuilt.sqlite doctor
```

Nothing may leave the hot tier until it is present in a published immutable release and
the restore workflow has passed against that exact tag. Cold archives are append-only;
they are never edited, replaced, or deleted. No evidence is deleted anywhere.

The rapid plan deliberately shares the full plan identity, so its scheduled observations count toward the existing 30-minute rapid-lane continuity gates without allowing manual cycles to fill a gap. The full workflow validates the complete plan, publishes the combined operational scorecard, and fails when either rapid lane lacks a full observed 24-hour in-bound window. The scorecard reports market coverage, ingestion health, forecast evidence, per-strategy pending and overdue outcomes, prediction-calibration cohort readiness, cost-adjusted eligibility, and execution-audit counts. Its memecoin summary treats hard-gate observations outside the active health window as missing, so archived authority, holder, transfer, or simulation records cannot make a current token appear sandbox-eligible. GitHub annotations flag failures, credentials waiting, and newly qualified candidates. Markdown and JSON scorecards are retained for 90 days; the Neon provider owns database durability and recovery. GitHub marks the full workflow failed when any active job is missing, more than 90 minutes stale, or has a latest failed run.

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
