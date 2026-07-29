import unittest
from datetime import date, datetime, timedelta, timezone

from trading_bot.core.schemas import Forecast, ForecastKind
from trading_bot.evaluation.costs import (
    CostBasis,
    EconomicCostModel,
    EconomicCostRegistry,
    load_cost_registry,
)
from trading_bot.evaluation.economics import (
    EconomicGateConfig,
    EconomicStatus,
    build_economic_report,
)
from trading_bot.evaluation.reporting import EvaluationGateConfig, build_walk_forward_report
from trading_bot.evaluation.scoring import (
    ScoreKind,
    score_binary_forecast,
    score_funding_forecast,
    score_return_forecast,
)


class EconomicReplayTests(unittest.TestCase):
    def setUp(self):
        self.base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def return_observations(self, count=4):
        observations = []
        for index in range(count):
            generated_at = self.base + timedelta(hours=index)
            actual = 0.04 if index % 2 == 0 else -0.04
            forecast = Forecast(
                f"return-{index}",
                "return-specialist",
                "v1",
                f"instrument-{index}",
                ForecastKind.RETURN_DISTRIBUTION,
                generated_at,
                generated_at + timedelta(hours=1),
                {"predicted_return": actual, "benchmark_return": 0.0},
                0.5,
                {"sample_size": 1.0},
                (f"event-{index}",),
                ("fails economic validation",),
            )
            score = score_return_forecast(
                forecast,
                actual_return=actual,
                target_time=forecast.valid_until,
                scored_at=forecast.valid_until,
            )
            observations.append((forecast, score))
        return observations

    def registry(self, cost_bps):
        return EconomicCostRegistry(
            "test-costs-v1",
            (
                EconomicCostModel(
                    "return-costs",
                    "return-specialist",
                    ScoreKind.RETURN,
                    CostBasis.STATIC_BPS,
                    "https://example.com/fees",
                    date(2026, 1, 1),
                    fee_bps=cost_bps,
                ),
            ),
        )

    def report(self, observations, registry, *, min_outcomes=2, min_trades=2):
        forecasts = tuple(item[0] for item in observations)
        scores = tuple(item[1] for item in observations)
        forecast_report = build_walk_forward_report(
            forecasts,
            scores,
            EvaluationGateConfig(min_independent_outcomes=min_outcomes),
        )
        return build_economic_report(
            forecasts,
            scores,
            forecast_report,
            registry,
            EconomicGateConfig(min_trades=min_trades),
        )

    def test_profitable_candidate_survives_base_and_doubled_costs(self):
        report = self.report(self.return_observations(), self.registry(10))
        evaluation = report.evaluations[0]
        self.assertEqual(evaluation.status, EconomicStatus.CANDIDATE)
        self.assertEqual(evaluation.trades, 4)
        self.assertAlmostEqual(evaluation.mean_gross_return, 0.04)
        self.assertAlmostEqual(evaluation.mean_net_return, 0.039)
        self.assertAlmostEqual(evaluation.doubled_cost_mean_return, 0.038)
        self.assertGreater(evaluation.doubled_cost_lower_confidence_bound, 0)
        self.assertEqual(evaluation.max_full_notional_drawdown, 0)

    def test_economic_gate_uses_same_crypto_outcome_clusters(self):
        observations = []
        for cluster_index, actual in enumerate((0.04, -0.04)):
            target_time = self.base + timedelta(hours=cluster_index + 1)
            for instrument_index in range(2):
                generated_at = target_time - timedelta(hours=1)
                if instrument_index == cluster_index:
                    generated_at += timedelta(minutes=1)
                forecast = Forecast(
                    f"crypto-{cluster_index}-{instrument_index}",
                    "crypto-range-breakout-continuation-baseline",
                    "baseline-v1",
                    f"coinbase:product:ASSET-{instrument_index}-USD",
                    ForecastKind.RETURN_DISTRIBUTION,
                    generated_at,
                    target_time,
                    {"predicted_return": actual, "benchmark_return": 0.0},
                    0.4,
                    {"sample_size": 20.0},
                    (f"event-{cluster_index}-{instrument_index}",),
                    ("correlated assets invalidate independence",),
                )
                score = score_return_forecast(
                    forecast,
                    actual_return=actual,
                    target_time=target_time,
                    scored_at=target_time,
                )
                observations.append((forecast, score))

        report = self.report(
            observations,
            EconomicCostRegistry(
                "crypto-costs-v1",
                (
                    EconomicCostModel(
                        "crypto-costs",
                        "crypto-range-breakout-continuation-baseline",
                        ScoreKind.RETURN,
                        CostBasis.STATIC_BPS,
                        "https://example.com/fees",
                        date(2026, 1, 1),
                        fee_bps=10,
                    ),
                ),
            ),
            min_outcomes=2,
            min_trades=2,
        )

        evaluation = report.evaluations[0]
        self.assertEqual(evaluation.eligible_forecasts, 2)
        self.assertEqual(evaluation.trades, 2)
        self.assertEqual(evaluation.status, EconomicStatus.CANDIDATE)

    def test_doubled_cost_failure_rejects_positive_base_case(self):
        report = self.report(self.return_observations(), self.registry(300))
        evaluation = report.evaluations[0]
        self.assertGreater(evaluation.mean_net_return, 0)
        self.assertLess(evaluation.doubled_cost_mean_return, 0)
        self.assertEqual(evaluation.status, EconomicStatus.REJECTED)
        self.assertIn("mean return fails doubled-cost stress", evaluation.reasons)

    def test_non_candidate_forecast_family_is_blocked_before_mapping(self):
        report = self.report(
            self.return_observations(count=1),
            self.registry(10),
            min_outcomes=2,
        )
        evaluation = report.evaluations[0]
        self.assertEqual(evaluation.status, EconomicStatus.BLOCKED)
        self.assertEqual(evaluation.trades, 0)
        self.assertIn("forecast gate is collecting", evaluation.reasons)

    def test_candidate_without_registered_cost_model_is_unsupported(self):
        empty_for_specialist = EconomicCostRegistry(
            "different-model-v1",
            (
                EconomicCostModel(
                    "other-costs",
                    "other-specialist",
                    ScoreKind.RETURN,
                    CostBasis.STATIC_BPS,
                    "https://example.com/fees",
                    date(2026, 1, 1),
                    fee_bps=10,
                ),
            ),
        )
        report = self.report(self.return_observations(), empty_for_specialist)
        evaluation = report.evaluations[0]
        self.assertEqual(evaluation.status, EconomicStatus.UNSUPPORTED)
        self.assertIn("no pre-registered economic cost model", evaluation.reasons)

    def test_checked_in_cost_registry_is_strict_and_versioned(self):
        registry = load_cost_registry("config/economic-costs.json")
        self.assertEqual(registry.version, "public-shadow-costs-v8")
        self.assertEqual(len(registry.models), 10)
        self.assertEqual(registry.models[0].fee_bps, 120)
        self.assertEqual(
            registry.models[2].specialist_id,
            "crypto-intraday-momentum-baseline",
        )
        self.assertEqual(
            registry.models[3].specialist_id,
            "crypto-intraday-momentum-baseline-v2",
        )
        self.assertEqual(
            registry.models[4].specialist_id,
            "prediction-market-calibration-baseline-v3",
        )
        self.assertEqual(
            registry.models[5].specialist_id,
            "prediction-market-calibration-adjusted-v1",
        )
        self.assertEqual(
            registry.models[6].specialist_id,
            "prediction-market-fast-settlement-baseline-v1",
        )
        self.assertEqual(
            registry.models[7].specialist_id,
            "prediction-market-fast-settlement-baseline-v2",
        )
        self.assertEqual(
            registry.models[8].specialist_id,
            "prediction-market-fast-settlement-baseline-v3",
        )
        self.assertEqual(
            registry.models[9].specialist_id,
            "prediction-market-fast-settlement-baseline-v4",
        )

    def test_binary_replay_uses_executable_side_and_rounded_contract_fee(self):
        observations = []
        for index in range(4):
            outcome = index % 2 == 0
            probability = 0.9 if outcome else 0.1
            for strike in range(2):
                generated_at = self.base + timedelta(hours=index, seconds=strike)
                forecast = Forecast(
                    f"binary-{index}-{strike}",
                    "binary-specialist",
                    "v1",
                    f"market-{index}-{strike}",
                    ForecastKind.BINARY_PROBABILITY,
                    generated_at,
                    generated_at + timedelta(hours=1),
                    {
                        "probability": probability,
                        "market_probability": 0.5,
                        "yes_bid": 0.49,
                        "yes_ask": 0.51,
                        "event_ticker": f"EVENT-{index}",
                    },
                    0.5,
                    {"sample_size": 1.0},
                    (f"event-{index}-{strike}",),
                    ("fails economic validation",),
                )
                score = score_binary_forecast(
                    forecast,
                    outcome=outcome,
                    target_time=forecast.valid_until + timedelta(seconds=strike),
                    scored_at=forecast.valid_until + timedelta(seconds=strike),
                )
                observations.append((forecast, score))
        registry = EconomicCostRegistry(
            "binary-costs-v1",
            (
                EconomicCostModel(
                    "binary-fees",
                    "binary-specialist",
                    ScoreKind.BINARY,
                    CostBasis.BINARY_CONTRACT,
                    "https://example.com/fees",
                    date(2026, 1, 1),
                    binary_fee_coefficient=0.07,
                    binary_fee_increment=0.01,
                ),
            ),
        )
        evaluation = self.report(observations, registry).evaluations[0]
        self.assertEqual(evaluation.status, EconomicStatus.CANDIDATE)
        self.assertEqual(evaluation.eligible_forecasts, 4)
        self.assertEqual(evaluation.trades, 4)
        self.assertAlmostEqual(evaluation.mean_gross_return, 0.5)
        self.assertAlmostEqual(evaluation.mean_assumed_cost, 0.03)
        self.assertAlmostEqual(evaluation.mean_net_return, 0.47)

    def test_funding_replay_uses_forecast_execution_bound(self):
        observations = []
        for index in range(4):
            generated_at = self.base + timedelta(hours=index)
            actual = 0.002 if index % 2 == 0 else -0.002
            forecast = Forecast(
                f"funding-{index}",
                "funding-specialist",
                "v1",
                f"perpetual-{index}",
                ForecastKind.FUNDING_RATE,
                generated_at,
                generated_at + timedelta(hours=1),
                {
                    "predicted_funding_rate": actual,
                    "current_funding_rate": 0.0,
                    "execution_bound_bps": 4.0,
                    "funding_and_basis_same_signed": True,
                    "state": "positive_carry" if actual > 0 else "negative_carry",
                },
                0.5,
                {"sample_size": 1.0},
                (f"event-{index}",),
                ("fails economic validation",),
            )
            score = score_funding_forecast(
                forecast,
                actual_rate=actual,
                target_time=forecast.valid_until,
                scored_at=forecast.valid_until,
            )
            observations.append((forecast, score))
        registry = EconomicCostRegistry(
            "funding-costs-v1",
            (
                EconomicCostModel(
                    "dynamic-bound",
                    "funding-specialist",
                    ScoreKind.FUNDING,
                    CostBasis.FORECAST_EXECUTION_BOUND,
                    "https://example.com/fees",
                    date(2026, 1, 1),
                    latency_bps=1.0,
                ),
            ),
        )
        evaluation = self.report(observations, registry).evaluations[0]
        self.assertEqual(evaluation.status, EconomicStatus.CANDIDATE)
        self.assertAlmostEqual(evaluation.mean_assumed_cost, 0.0005)
        self.assertAlmostEqual(evaluation.mean_net_return, 0.0015)
        self.assertAlmostEqual(evaluation.doubled_cost_mean_return, 0.001)


if __name__ == "__main__":
    unittest.main()
