from datetime import datetime, timezone

from trading_bot.core.schemas import AssetClass, Hypothesis


BASELINE_PROPOSED_AT = datetime(2026, 7, 21, tzinfo=timezone.utc)
PREDICTION_V3_PROPOSED_AT = datetime(2026, 7, 23, tzinfo=timezone.utc)
PREDICTION_V4_PROPOSED_AT = datetime(2026, 7, 23, 1, 35, tzinfo=timezone.utc)
CRYPTO_INTRADAY_PROPOSED_AT = datetime(2026, 7, 26, 11, 35, tzinfo=timezone.utc)
PREDICTION_FAST_PROPOSED_AT = datetime(2026, 7, 26, 13, 10, tzinfo=timezone.utc)


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
    hypothesis_id="prediction-market-calibration-baseline-v4",
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
    proposed_at=PREDICTION_V4_PROPOSED_AT,
)


PREDICTION_FAST_SETTLEMENT_HYPOTHESIS = Hypothesis(
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
    proposed_at=PREDICTION_FAST_PROPOSED_AT,
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


BASELINE_HYPOTHESES = (
    PERPETUAL_FUNDING_HYPOTHESIS,
    OPTIONS_VOLATILITY_HYPOTHESIS,
    PREDICTION_CALIBRATION_HYPOTHESIS,
    PREDICTION_CALIBRATION_ADJUSTED_HYPOTHESIS,
    PREDICTION_FAST_SETTLEMENT_HYPOTHESIS,
    CRYPTO_BREAKOUT_HYPOTHESIS,
    CRYPTO_INTRADAY_MOMENTUM_HYPOTHESIS,
)
