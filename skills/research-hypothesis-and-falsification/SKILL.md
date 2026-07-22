---
name: research-hypothesis-and-falsification
description: Design auditable trading experiments by registering mechanisms and invalidation conditions before testing, tracking every trial, using point-in-time walk-forward evaluation, and correcting for false discovery. Use when proposing, backtesting, tuning, comparing, rejecting, or promoting any strategy or predictive feature.
---

# Research Hypothesis and Falsification

Try to disprove a mechanism before optimizing it.

## Workflow

1. Write a `Hypothesis` before inspecting final-period results. State mechanism, target, horizon, information set, and invalidation conditions.
2. Register it with `ExperimentRegistry` in `src/trading_bot/core/experiments.py`.
3. Record the experiment family, full configuration, code version, and immutable data cutoff.
4. Choose walk-forward splits that respect time. Purge overlapping label windows and reserve an untouched final period.
5. Model actual spreads, fees, funding, borrow, gas, impact, latency, rejects, partial fills, and contract lifecycle.
6. Compare against simple benchmarks, shuffled/null signals, and known common-risk exposures.
7. Stress start dates, universes, venues, parameters, costs, missing data, and market regimes.
8. Record failed and rejected trials. Never delete them or restart a family to reset its trial count.
9. Promote only through the documented replay, shadow, and paper gates.

## Report

Return the mechanism, experiment ID, information cutoff, all trials attempted, validation design, costs, sensitivity results, failure modes, and a `reject`, `continue research`, or `shadow` recommendation.

Never use a research result to submit an order directly.
