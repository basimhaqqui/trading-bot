from datetime import datetime, timezone

from trading_bot.core.schemas import AssetClass, Hypothesis


BASELINE_PROPOSED_AT = datetime(2026, 7, 21, tzinfo=timezone.utc)
PREDICTION_V3_PROPOSED_AT = datetime(2026, 7, 23, tzinfo=timezone.utc)
PREDICTION_ADJUSTED_V1_PROPOSED_AT = datetime(2026, 7, 29, 1, tzinfo=timezone.utc)
CRYPTO_INTRADAY_PROPOSED_AT = datetime(2026, 7, 26, 11, 35, tzinfo=timezone.utc)
CRYPTO_INTRADAY_V2_PROPOSED_AT = datetime(2026, 7, 27, 5, 57, tzinfo=timezone.utc)
PREDICTION_FAST_V1_PROPOSED_AT = datetime(2026, 7, 26, 13, 10, tzinfo=timezone.utc)
PREDICTION_FAST_V2_PROPOSED_AT = datetime(2026, 7, 27, tzinfo=timezone.utc)
PREDICTION_FAST_V3_PROPOSED_AT = datetime(2026, 7, 27, 6, 25, tzinfo=timezone.utc)
PREDICTION_FAST_V4_PROPOSED_AT = datetime(2026, 7, 28, 7, 5, tzinfo=timezone.utc)
PREDICTION_FAST_V5_PROPOSED_AT = datetime(2026, 7, 30, 5, 0, tzinfo=timezone.utc)
PREDICTION_FAST_V6_PROPOSED_AT = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)
PREDICTION_FAST_V7_PROPOSED_AT = datetime(2026, 7, 31, 19, 30, tzinfo=timezone.utc)
PREDICTION_FAST_V8_PROPOSED_AT = datetime(2026, 8, 1, 9, 25, tzinfo=timezone.utc)


PERPETUAL_FUNDING_HYPOTHESIS = Hypothesis(
    hypothesis_id="perpetual-funding-basis-baseline-v1",
    family="perpetual-funding-basis",
    market=AssetClass.PERPETUAL,
    mechanism=(
        "Funding is persistent over short horizons when leveraged positioning and the "
        "perpetual-versus-spot basis remain same-signed, but only carry above executable "
        "spread and fee bounds is economically relevant."
    ),
    target="next published funding rate and current perpetual-versus-spot basis",
    horizon="one funding interval",
    information_set=(
        "point-in-time funding observations",
        "perpetual executable book",
        "related spot executable book",
        "open interest when available",
    ),
    invalidation_conditions=(
        "funding persistence is no better than the latest-rate benchmark out of sample",
        "estimated carry does not survive doubled fees and spreads",
        "results depend on one venue, regime, or parameter choice",
    ),
    proposed_at=BASELINE_PROPOSED_AT,
)


OPTIONS_VOLATILITY_HYPOTHESIS = Hypothesis(
    hypothesis_id="options-implied-volatility-state-baseline-v1",
    family="options-implied-volatility-state",
    market=AssetClass.OPTION,
    mechanism=(
        "Contract-level implied volatility is persistent over short horizons, while large "
        "deviations from its trailing point-in-time median identify a volatility state that "
        "can later be tested against underlying realized volatility and executable costs."
    ),
    target="next-session implied volatility state",
    horizon="one trading session",
    information_set=(
        "point-in-time option quotes",
        "snapshot implied volatility",
        "bid-ask spread",
        "feed provenance",
    ),
    invalidation_conditions=(
        "the trailing state does not predict next-session implied volatility",
        "results disappear on OPRA quotes or after executable spreads",
        "adjusted, expiring, or illiquid contracts drive the result",
    ),
    proposed_at=BASELINE_PROPOSED_AT,
)


PREDICTION_CALIBRATION_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-calibration-baseline-v3",
    family="prediction-market-calibration",
    market=AssetClass.PREDICTION,
    mechanism=(
        "Executable market probabilities are the primary prior, but resolved, independent "
        "underlying occurrences may reveal stable probability-bucket calibration errors after "
        "conditioning on time horizon, spread, and exact settlement rules. Correlated props "
        "and strikes from one occurrence contribute at most one calibration sample."
    ),
    target="binary settlement probability",
    horizon="one to eight hours until the underlying occurrence",
    information_set=(
        "current yes and no executable bids",
        "contract settlement rules",
        "point-in-time books from resolved related markets",
        "resolved outcomes available before forecast time",
        "a pre-declared maximum executable spread of ten cents",
        "a pre-declared occurrence horizon greater than one and at most eight hours",
    ),
    invalidation_conditions=(
        "calibration adjustment does not improve held-out Brier or log loss",
        "improvement disappears by resolution cohort or time-to-event bucket",
        "spread, contract wording, or selection bias explains the apparent adjustment",
    ),
    proposed_at=PREDICTION_V3_PROPOSED_AT,
)


PREDICTION_CALIBRATION_ADJUSTED_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-calibration-adjusted-v1",
    family="prediction-market-calibration",
    market=AssetClass.PREDICTION,
    mechanism=(
        "Executable market probabilities are adjusted only when at least five resolved, "
        "independent Kalshi events in the same fixed ten-cent probability bucket reveal a "
        "calibration difference. The empirical rate is shrunk toward the current market "
        "probability with twenty prior observations, and benchmark-identical forecasts are "
        "not emitted."
    ),
    target="binary settlement probability",
    horizon="one to eight hours until the underlying occurrence",
    information_set=(
        "current yes and no executable bids",
        "contract settlement rules and stable event ticker",
        "point-in-time books observed one to eight hours before resolved occurrences",
        "resolved outcomes available before forecast time",
        "a fixed ten-cent probability bucket radius",
        "a fixed minimum of five independent resolved events",
        "a fixed shrinkage prior of twenty observations",
        "a fixed maximum executable spread of ten cents",
    ),
    invalidation_conditions=(
        "the adjusted forecast does not improve held-out Brier or log loss",
        "improvement disappears by resolution cohort or time-to-event bucket",
        "spread, contract wording, or selection bias explains the apparent adjustment",
        "the adjustment does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_ADJUSTED_V1_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v1",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for its finalized settlement. This lane is intentionally unadjusted: "
        "it collects prospective calibration evidence without borrowing from the longer "
        "one-to-eight-hour cohort."
    ),
    target="binary finalized settlement probability",
    horizon="20 minutes to two hours until Kalshi expected expiration",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker and expected expiration time",
        "active market status with early closing disabled",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "expected-expiration changes or delayed finalization explain apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V1_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v2",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for its finalized settlement. This lane is intentionally unadjusted: "
        "it collects prospective calibration evidence without borrowing from the longer "
        "one-to-eight-hour cohort."
    ),
    target="binary finalized settlement probability",
    horizon="20 minutes to two hours until Kalshi expected expiration",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker and expected expiration time",
        "active market status with early closing disabled",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "the pre-recorded expected-expiration plus settlement-timer deadline",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "expected-expiration changes or finalization after the pre-recorded settlement deadline explain apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V2_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v3",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for its finalized settlement. This lane is intentionally unadjusted: "
        "it collects prospective calibration evidence without borrowing from the longer "
        "one-to-eight-hour cohort. A settlement is eligible only when its recorded "
        "finalization timestamp falls between the forecast's pre-recorded expected "
        "expiration and its pre-recorded settlement deadline."
    ),
    target="binary finalized settlement probability",
    horizon="20 minutes to two hours until Kalshi expected expiration",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker and expected expiration time",
        "active market status with early closing disabled",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "the pre-recorded expected-expiration plus settlement-timer label window",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "the required settlement label window leaves insufficient independent outcomes",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V3_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v4",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for its finalized settlement. This lane is intentionally unadjusted: "
        "it collects prospective calibration evidence without borrowing from the longer "
        "one-to-eight-hour cohort. Each observation cycle reads one public Kalshi market "
        "page and advances the documented cursor, so API pagination cannot silently make "
        "the selection universe a permanently fixed first page. A settlement is eligible "
        "only when its recorded finalization timestamp falls between the forecast's "
        "pre-recorded expected expiration and its pre-recorded settlement deadline."
    ),
    target="binary finalized settlement probability",
    horizon="20 minutes to two hours until Kalshi expected expiration",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker and expected expiration time",
        "active market status with early closing disabled",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "one public market-list page per cycle with the returned cursor resumed next cycle",
        "the pre-recorded expected-expiration plus settlement-timer label window",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "the required settlement label window leaves insufficient independent outcomes",
        "cursor order or page coverage explains the apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V4_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V5_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v5",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for its finalized settlement. This lane is intentionally unadjusted. "
        "Kalshi documents that can_close_early may move close_time earlier, so the "
        "recorded expected expiration is not treated as a fixed lower label bound. "
        "Instead, each forecast records the venue's latest_expiration_time and accepts "
        "only its own event's public finalization from forecast creation through the "
        "pre-recorded latest-expiration plus settlement-timer deadline."
    ),
    target="binary finalized settlement probability",
    horizon="20 minutes to two hours until expected expiration and no more than two hours until latest expiration",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker, expected expiration time, and latest expiration time",
        "active market status and documented boolean can_close_early policy",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "one public market-list page per cycle with the returned cursor resumed next cycle",
        "the pre-recorded latest-expiration plus settlement-timer label deadline",
        "Kalshi Market Lifecycle documentation: https://docs.kalshi.com/getting_started/market_lifecycle",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "the bounded latest-expiration label window leaves insufficient independent outcomes",
        "cursor order or page coverage explains the apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V5_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V6_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v6",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for a fast finalization. This lane is intentionally unadjusted. "
        "Kalshi documents latest_expiration_time as the latest possible expiration, not "
        "a short-horizon promise, so it is recorded for audit but does not exclude a "
        "market whose expected expiration is 20 minutes to two hours away. Each forecast "
        "accepts only its own event's public finalization from forecast creation through "
        "the pre-recorded expected-expiration plus settlement-timer plus one-hour "
        "fast-finalization deadline. Late or absent labels remain unscored and cannot be "
        "reclassified into evidence."
    ),
    target="binary finalized settlement probability for a pre-registered fast-finalization window",
    horizon="20 minutes to two hours until expected expiration; finalization no later than one hour plus the recorded settlement timer afterward",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker, expected expiration time, and latest expiration time",
        "active market status and documented boolean can_close_early policy",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "one public market-list page per cycle with the returned cursor resumed next cycle",
        "the pre-recorded expected-expiration plus settlement-timer plus one-hour label deadline",
        "Kalshi Market Lifecycle documentation: https://docs.kalshi.com/getting_started/market_lifecycle",
        "Kalshi Market Settlement documentation: https://docs.kalshi.com/getting_started/market_settlement",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "the fixed fast-finalization window leaves insufficient independent outcomes",
        "late or missing label rates indicate that the lane is not operationally representative",
        "cursor order or page coverage explains the apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V6_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V7_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v7",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for a fast finalization. This lane is intentionally unadjusted. "
        "Kalshi documents that close_time may move earlier only when can_close_early is "
        "true, so an early finalization is accepted only when that boolean policy was "
        "recorded as true at forecast generation. Each forecast otherwise accepts only "
        "its own event's public finalization from expected expiration through the "
        "pre-recorded expected-expiration plus settlement-timer plus one-hour deadline. "
        "Late, absent, and policy-inconsistent labels remain unscored and cannot be "
        "reclassified into evidence."
    ),
    target="binary finalized settlement probability for a pre-registered fast-finalization window",
    horizon="20 minutes to two hours until expected expiration; finalization no later than one hour plus the recorded settlement timer afterward",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker, expected expiration time, and latest expiration time",
        "active market status and documented boolean can_close_early policy",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "one public market-list page per cycle with the returned cursor resumed next cycle",
        "the pre-recorded expected-expiration plus settlement-timer plus one-hour label deadline",
        "Kalshi Market Lifecycle documentation: https://docs.kalshi.com/getting_started/market_lifecycle",
        "Kalshi Market Settlement documentation: https://docs.kalshi.com/getting_started/market_settlement",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "the fixed fast-finalization window leaves insufficient independent outcomes",
        "late, missing, or policy-inconsistent label rates indicate that the lane is not operationally representative",
        "cursor order or page coverage explains the apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V7_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_V8_HYPOTHESIS = Hypothesis(
    hypothesis_id="prediction-market-fast-settlement-baseline-v8",
    family="prediction-market-fast-settlement",
    market=AssetClass.PREDICTION,
    mechanism=(
        "The executable probability of a short-dated, active binary market is a fixed "
        "baseline for a fast finalization. This lane is intentionally unadjusted. "
        "Kalshi documents that an active market reaches closed when close_time passes, "
        "and that close_time may move earlier only when can_close_early is true. An "
        "early finalization therefore counts only when the forecast recorded that "
        "boolean as true and the later public finalization response records a close_time "
        "after forecast generation but before both settlement and expected expiration. "
        "Each forecast otherwise accepts only its own event's public finalization from "
        "expected expiration through the pre-recorded expected-expiration plus "
        "settlement-timer plus one-hour deadline. Late, absent, and insufficiently "
        "corroborated labels remain unscored and cannot be reclassified into evidence."
    ),
    target="binary finalized settlement probability for a pre-registered fast-finalization window",
    horizon="20 minutes to two hours until expected expiration; finalization no later than one hour plus the recorded settlement timer afterward",
    information_set=(
        "current executable yes and no bids",
        "Kalshi event ticker, expected expiration time, and latest expiration time",
        "active market status and documented boolean can_close_early policy",
        "a public finalization response's recorded close_time for any early label",
        "a settlement timer no longer than fifteen minutes",
        "a pre-declared maximum executable spread of ten cents",
        "one public market-list page per cycle with the returned cursor resumed next cycle",
        "the pre-recorded expected-expiration plus settlement-timer plus one-hour label deadline",
        "Kalshi Market Lifecycle documentation: https://docs.kalshi.com/getting_started/market_lifecycle",
        "Kalshi Market Settlement documentation: https://docs.kalshi.com/getting_started/market_settlement",
    ),
    invalidation_conditions=(
        "market probabilities do not beat the fixed neutral benchmark out of sample",
        "results disappear when one event ticker or resolution period is removed",
        "the fixed fast-finalization window leaves insufficient independent outcomes",
        "late, missing, policy-inconsistent, or uncorroborated early label rates indicate that the lane is not operationally representative",
        "cursor order or page coverage explains the apparent performance",
        "the baseline does not survive fees, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=PREDICTION_FAST_V8_PROPOSED_AT,
)


CRYPTO_BREAKOUT_HYPOTHESIS = Hypothesis(
    hypothesis_id="crypto-range-breakout-continuation-baseline-v1",
    family="crypto-range-breakout-continuation",
    market=AssetClass.CRYPTO,
    mechanism=(
        "A close beyond the prior twenty completed bars' range, accompanied by nontrivial "
        "volume, may contain short-horizon continuation information. The forecast magnitude "
        "is a fixed shrinkage of the observed range break rather than an optimized parameter."
    ),
    target="next-bar close-to-close return after a confirmed range breakout",
    horizon="one source bar",
    information_set=(
        "point-in-time completed OHLCV bars",
        "prior twenty-bar high and low",
        "current breakout close and volume ratio",
        "source granularity and receipt timestamp",
    ),
    invalidation_conditions=(
        "breakout forecasts do not beat a zero-return benchmark out of sample",
        "results fail delayed or shuffled-prediction controls",
        "results disappear after fees, slippage, latency, and spread stress",
        "one asset, direction, volatility regime, or parameter choice drives the result",
    ),
    proposed_at=BASELINE_PROPOSED_AT,
)


CRYPTO_INTRADAY_MOMENTUM_HYPOTHESIS = Hypothesis(
    hypothesis_id="crypto-intraday-momentum-baseline-v1",
    family="crypto-intraday-momentum",
    market=AssetClass.CRYPTO,
    mechanism=(
        "A fixed fraction of the average log return across the previous eight completed "
        "fifteen-minute bars may persist into the next bar in continuously traded, liquid "
        "crypto markets. Every parameter and the market-wide outcome cluster are fixed "
        "before this hypothesis begins collecting evidence."
    ),
    target="next completed fifteen-minute close-to-close return",
    horizon="one fifteen-minute bar",
    information_set=(
        "the previous eight completed Coinbase fifteen-minute OHLCV bars",
        "their original exchange timestamps and first observed availability",
        "a fixed 0.25 shrinkage of average trailing log return",
        "a fixed one percent absolute forecast cap",
    ),
    invalidation_conditions=(
        "momentum forecasts do not beat a zero-return benchmark out of sample",
        "results fail delayed or shuffled-prediction controls",
        "market-wide blocks, serial dependence, or one instrument explain the result",
        "returns do not survive fees, spread, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=CRYPTO_INTRADAY_PROPOSED_AT,
)


CRYPTO_INTRADAY_MOMENTUM_V2_HYPOTHESIS = Hypothesis(
    hypothesis_id="crypto-intraday-momentum-baseline-v2",
    family="crypto-intraday-momentum",
    market=AssetClass.CRYPTO,
    mechanism=(
        "A fixed fraction of the average log return across the previous eight completed "
        "fifteen-minute bars may persist into the next bar in continuously traded, liquid "
        "crypto markets. To keep a market-wide outcome cluster independent while avoiding "
        "ingestion-order selection, each target time is assigned before evaluation to one "
        "fixed Coinbase symbol by a deterministic SHA-256 rule; an absent signal is not "
        "replaced by another symbol."
    ),
    target="next completed fifteen-minute close-to-close return",
    horizon="one fifteen-minute bar",
    information_set=(
        "the previous eight completed Coinbase fifteen-minute OHLCV bars",
        "their original exchange timestamps and first observed availability",
        "a fixed 0.25 shrinkage of average trailing log return",
        "a fixed one percent absolute forecast cap",
        "a fixed ten-symbol Coinbase universe",
        "a target-time SHA-256 assignment to one universe symbol before signal evaluation",
    ),
    invalidation_conditions=(
        "momentum forecasts do not beat a zero-return benchmark out of sample",
        "results fail delayed or shuffled-prediction controls",
        "market-wide blocks, serial dependence, or one instrument explain the result",
        "returns do not survive fees, spread, slippage, latency, and doubled-cost stress",
    ),
    proposed_at=CRYPTO_INTRADAY_V2_PROPOSED_AT,
)


BASELINE_HYPOTHESES = (
    PERPETUAL_FUNDING_HYPOTHESIS,
    OPTIONS_VOLATILITY_HYPOTHESIS,
    PREDICTION_CALIBRATION_HYPOTHESIS,
    PREDICTION_CALIBRATION_ADJUSTED_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V5_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V6_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V7_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_V8_HYPOTHESIS,
    CRYPTO_BREAKOUT_HYPOTHESIS,
    CRYPTO_INTRADAY_MOMENTUM_HYPOTHESIS,
    CRYPTO_INTRADAY_MOMENTUM_V2_HYPOTHESIS,
)


# Forecast records retain their stable specialist IDs while hypotheses carry
# versioned preregistration IDs. Keep this relationship explicit so reporting
# cannot silently omit prospective evidence when the two names differ.
BASELINE_HYPOTHESIS_SPECIALIST_IDS = {
    PERPETUAL_FUNDING_HYPOTHESIS.hypothesis_id: (
        "perpetual-funding-basis-baseline",
    ),
    OPTIONS_VOLATILITY_HYPOTHESIS.hypothesis_id: (
        "options-implied-volatility-state-baseline",
    ),
    PREDICTION_CALIBRATION_HYPOTHESIS.hypothesis_id: (
        "prediction-market-calibration-baseline-v3",
    ),
    PREDICTION_CALIBRATION_ADJUSTED_HYPOTHESIS.hypothesis_id: (
        "prediction-market-calibration-adjusted-v1",
    ),
    PREDICTION_FAST_SETTLEMENT_V1_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v1",
    ),
    PREDICTION_FAST_SETTLEMENT_V2_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v2",
    ),
    PREDICTION_FAST_SETTLEMENT_V3_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v3",
    ),
    PREDICTION_FAST_SETTLEMENT_V4_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v4",
    ),
    PREDICTION_FAST_SETTLEMENT_V5_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v5",
    ),
    PREDICTION_FAST_SETTLEMENT_V6_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v6",
    ),
    PREDICTION_FAST_SETTLEMENT_V7_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v7",
    ),
    PREDICTION_FAST_SETTLEMENT_V8_HYPOTHESIS.hypothesis_id: (
        "prediction-market-fast-settlement-baseline-v8",
    ),
    CRYPTO_BREAKOUT_HYPOTHESIS.hypothesis_id: (
        "crypto-range-breakout-continuation-baseline",
    ),
    CRYPTO_INTRADAY_MOMENTUM_HYPOTHESIS.hypothesis_id: (
        "crypto-intraday-momentum-baseline",
    ),
    CRYPTO_INTRADAY_MOMENTUM_V2_HYPOTHESIS.hypothesis_id: (
        "crypto-intraday-momentum-baseline-v2",
    ),
}
