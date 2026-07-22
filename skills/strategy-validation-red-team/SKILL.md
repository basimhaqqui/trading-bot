---
name: strategy-validation-red-team
description: Independently attack a trading strategy, forecast, backtest, or promotion claim for data leakage, selection bias, unrealistic fills, hidden common risks, operational failures, and irreproducibility. Use before shadow or live promotion, after suspicious performance, or when reviewing research produced by another agent.
---

# Strategy Validation Red Team

Assume the result is wrong until reproduced from raw point-in-time data.

## Validation sequence

1. Reconstruct the claim from its hypothesis, experiment record, code version, and data cutoff. Do not accept screenshots or summary metrics as evidence.
2. Re-run from raw events and compare positions, fills, P&L, and metrics exactly.
3. Search for future availability, survivor-only universes, revisions, stale quotes, last-price fills, and contract-roll errors.
4. Count all related trials and challenge parameter, feature, universe, and period selection.
5. Replace the signal with shuffled, delayed, and null versions.
6. Double realistic costs and latency; reduce assumed liquidity; inject partial fills and rejected orders.
7. Attribute returns to beta, momentum, carry, short volatility, liquidity, leverage, venue, and event concentration.
8. Replay outages, stale feeds, duplicate events, option assignment, futures expiry, liquidation, chain congestion, settlement disputes, and margin increases as applicable.
9. Compare backtest, replay, shadow, paper, and live distributions. Investigate every unexplained degradation.

## Verdict

Return `pass`, `conditional`, or `block`, with reproducible evidence and the smallest required remediation. A red-team agent cannot approve its own strategy or place orders.
