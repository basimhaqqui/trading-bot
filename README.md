# Multi-market research system

Foundation for researching stocks, listed options, futures, perpetual futures, spot crypto, memecoins, and prediction markets without giving research agents direct control of money.

Milestone 11 adds a candidate-only economic validation gate. Forecast families must first pass every statistical and duration cohort, then survive executable-side payoffs, explicit fees, spread, slippage, latency haircuts, and doubled-cost stress before receiving an economic candidate label. It still has no broker, wallet, order-entry, or live execution adapter.

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
- Versioned, pre-registered hypotheses for perpetual funding/basis, options implied-volatility state, prediction-market calibration, and crypto range breakouts.
- Related-instrument replay for basis and historical calibration without weakening evidence checks.
- Deterministic research specialists that emit forecasts only and refuse insufficient or stale inputs.
- Benchmark-relative Brier, log-loss, funding-rate, volatility, and forward-return scoring.
- Validated shadow-ingestion plans, non-overlapping run locks, and append-only job records.
- Append-only cursor checkpoints that advance only after successful storage and preserve the requested and returned page tokens for audit.
- Per-job health gates for missing, stale, future-dated, and consecutively failing scheduled runs.
- Atomic SQLite online backups verified against database, event, forecast-audit, and ingestion-run integrity before publication.
- A deployment workflow with serialized runs, state continuity, 30-day recovery artifacts, manual restoration, and run summaries.
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
- Semantic candle deduplication that preserves first receipt time while storing later venue revisions as new immutable versions.

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
Shadow/paper adapter only
```

Research agents cannot call execution adapters. Execution adapters cannot browse, interpret news, select strategies, or change risk limits.

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

Use `--cursor` to continue a one-off manual collection. Scheduled shadow jobs manage pagination automatically. Every returned event is stored with collection time as its availability boundary. Venue text, including news and settlement rules, remains untrusted data.

## Shadow observation cycle

The checked-in public plan collects current Kalshi markets, public settlements and trades, Coinbase product definitions and funding, executable BTC spot/perpetual books, and hourly candles for BTC, ETH, SOL, and DOGE:

```bash
trading-bot shadow-cycle --plan config/shadow-ingestion.json --validate-only
trading-bot --db var/trading.db research-init
trading-bot --db var/trading.db shadow-cycle --plan config/shadow-ingestion.json
trading-bot --db var/trading.db readiness
```

The command runs exactly one cycle and exits, so an external scheduler controls frequency. After collection it scores any newly observable outcomes and emits fresh forecasts from eligible data. For paginated jobs, each successful or degraded run records both the requested cursor and the venue's next cursor. The next cycle advances to that page; a failed page is retried, and a terminal response with no next cursor restarts the crawl on page one. Cursor checkpoints are recovered from the immutable run ledger rather than hidden mutable state and are capped at 4,096 characters.

Plans cannot contain credentials, job types are allowlisted, concurrent cycles are rejected, failures are isolated per job, and every result is recorded append-only. Rerunning against the same information set is idempotent.

`shadow-research` runs only the local scoring and forecast stages, with no network collection. Prediction-market generation is capped at 25 fresh candidates per cycle; invalid or non-executable books are skipped. Funding forecasts require distinct funding periods plus fresh spot and perpetual books. Option forecasts require enough recent contract observations.

`shadow-report` evaluates scored forecasts against their explicit benchmarks. It uses only the latest forecast for each instrument/outcome pair, requires 30 outcome clusters by default, widens uncertainty for serial dependence, and requires the model to beat delayed and shuffled controls. Results are split by specialist/score family, forecast validity horizon, and realized target horizon using the fixed pre-declared buckets `<=1h`, `1h-8h`, `8h-1d`, `1d-7d`, `7d-30d`, and `>30d`. All populated overall and cohort tests share the family-wise confidence correction.

An aggregate cannot become a candidate while any observed horizon cohort has fewer than 30 independent outcomes. Once mature, every observed cohort must independently clear the benchmark, confidence, win-rate, delayed-control, and shuffled-control gates. The strongest label is `candidate`, never `proven`. Use `--min-outcomes` only for diagnostics; lowering it does not authorize paper or live allocation.

`economic-report` is a second, stricter gate and does no work for forecast families that are still collecting or rejected. Return forecasts use signed realized returns. Prediction forecasts buy the executable yes or no side, include the observed half-spread, and apply the general Kalshi taker-fee formula. Funding forecasts only map to a hedged carry trade when basis and funding agree and the forecast already clears its dynamic execution bound. Options volatility remains unsupported because an implied-volatility score alone cannot reconstruct a delta-hedged option P&L path.

The checked-in `config/economic-costs.json` uses a conservative 120 bps Coinbase retail taker round trip before additional spread, slippage, and latency; prediction fees use Kalshi's July 7, 2026 general schedule; perpetual forecasts use their point-in-time spot/perpetual execution bound. Every surviving trade is rerun at twice the complete assumed cost. Results are normalized research returns, not an allocation or portfolio-sizing recommendation.

Alpaca jobs use the `alpaca_market_data` activation profile. Without both read-only environment values they appear as `waiting_credentials`, remain health-neutral, and never construct a network collector. With both values present they become required health-gated jobs and collect paired option-chain snapshots and underlying stock bars.

The deployment-ready workflow is installed at `.github/workflows/shadow-ingestion.yml` and runs at minutes 7 and 37 of every hour once this project is in a GitHub repository. It restores the most recent cache, serializes overlapping runs, validates the plan, collects and researches, and publishes a combined operational scorecard. The scorecard reports market coverage, ingestion health, forecast evidence, cost-adjusted eligibility, and execution-audit counts. GitHub annotations flag failures, credentials waiting, and newly qualified candidates. Approximately once every 48 runs, plus every manual run, it retains the Markdown and JSON scorecards for 90 days and creates a verified SQLite snapshot retained for 30 days. GitHub marks the workflow failed when any active job is missing, more than 90 minutes stale, or has a latest failed run.

For recovery, manually dispatch the workflow and enter the workflow run ID that owns the desired `shadow-database` artifact in `restore_run_id`. The downloaded snapshot is integrity-checked before collection continues. The cache is continuity storage; the immutable run artifacts are the recovery path. For longer-lived or higher-volume research, move the same append-only model to managed durable storage before relying on it operationally.

## Research baselines

- Perpetual funding/basis: forecasts the next funding-rate state, calculates executable spot/perpetual basis, and places a conservative spread-and-fee bound around carry.
- Options volatility: forecasts short-horizon implied-volatility state from point-in-time contract quotes; indicative feeds receive a hard confidence cap.
- Prediction markets: starts from the executable yes/no midpoint and applies a shrunk calibration adjustment only when enough already-resolved related contracts exist.
- Crypto breakouts: detects closes outside the previous twenty completed hourly bars, requires nontrivial volume, and forecasts the next bar using a fixed 25% shrinkage of the range break. No parameter search is performed.

These are competence baselines, not proven edges. Each is compared with a simple benchmark and remains in research/shadow mode until held-out, net-cost validation passes.

## Test

```bash
python -m unittest discover -s tests -v
```

The tests deliberately attempt look-ahead evidence, premature outcome scoring, event and audit mutation, hypothesis redefinition, embedded plan credentials, venue symbol confusion, future source clocks, crossed books, privileged HTTP headers, host escape, stale specialist inputs, risk-limit violations, live execution, signature tampering, and duplicate execution.

## Next milestone

1. Accumulate at least 30 scored outcome clusters in every observed horizon cohort.
2. Add the two read-only Alpaca repository secrets to activate option/underlying collection.
3. Build a paper portfolio allocator only for baselines that survive both forecast and economic gates out of sample.

Live venue adapters remain out of scope until their data, eligibility, margin, settlement, and recovery behavior have been validated in replay and paper environments.
