from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trading_bot.core.schemas import Forecast
from trading_bot.core.serialization import sha256_digest
from trading_bot.evaluation.costs import CostBasis, EconomicCostModel, EconomicCostRegistry
from trading_bot.evaluation.reporting import (
    EdgeStatus,
    WalkForwardReport,
    independent_outcome_key,
)
from trading_bot.evaluation.scoring import ForecastScore, ScoreKind


class EconomicStatus(StrEnum):
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"
    COLLECTING = "collecting"
    REJECTED = "rejected"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class EconomicGateConfig:
    min_trades: int = 30
    familywise_alpha: float = 0.05
    minimum_win_rate: float = 0.50

    def __post_init__(self) -> None:
        if self.min_trades < 2:
            raise ValueError("minimum economic trades must be at least two")
        if not 0 < self.familywise_alpha < 0.5:
            raise ValueError("familywise alpha must be between zero and 0.5")
        if not 0 <= self.minimum_win_rate <= 1:
            raise ValueError("minimum win rate must be between zero and one")


@dataclass(frozen=True)
class EconomicTrade:
    forecast_id: str
    target_time: datetime
    direction: str
    gross_return: float
    assumed_cost: float
    net_return: float
    doubled_cost_net_return: float


@dataclass(frozen=True)
class EconomicEvaluation:
    specialist_id: str
    kind: ScoreKind
    status: EconomicStatus
    cost_model_id: str | None
    eligible_forecasts: int
    trades: int
    skipped_signals: int
    mean_gross_return: float | None
    mean_assumed_cost: float | None
    mean_net_return: float | None
    net_lower_confidence_bound: float | None
    win_rate: float | None
    doubled_cost_mean_return: float | None
    doubled_cost_lower_confidence_bound: float | None
    doubled_cost_win_rate: float | None
    max_full_notional_drawdown: float | None
    doubled_cost_max_full_notional_drawdown: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EconomicReport:
    cost_registry_version: str
    cost_registry_digest: str
    familywise_alpha: float
    confidence_tests: int
    evaluations: tuple[EconomicEvaluation, ...]


@dataclass(frozen=True)
class _Draft:
    specialist_id: str
    kind: ScoreKind
    forecast_status: EdgeStatus
    model: EconomicCostModel | None
    eligible_forecasts: int
    trades: tuple[EconomicTrade, ...]
    skipped_signals: int
    errors: tuple[str, ...]


def build_economic_report(
    forecasts: tuple[Forecast, ...],
    scores: tuple[ForecastScore, ...],
    forecast_report: WalkForwardReport,
    cost_registry: EconomicCostRegistry,
    config: EconomicGateConfig | None = None,
) -> EconomicReport:
    config = config or EconomicGateConfig()
    forecasts_by_id = {item.forecast_id: item for item in forecasts}
    models = {(item.specialist_id, item.kind): item for item in cost_registry.models}
    latest_scores = _latest_independent_scores(forecasts_by_id, scores)
    drafts: list[_Draft] = []
    for group in forecast_report.groups:
        key = (group.specialist_id, group.kind)
        model = models.get(key)
        trades: list[EconomicTrade] = []
        errors: list[str] = []
        skipped = 0
        if group.status is EdgeStatus.CANDIDATE and model is not None:
            for forecast, score in latest_scores.get(key, ()):
                try:
                    trade = _map_trade(forecast, score, model)
                except ValueError as exc:
                    errors.append(f"{forecast.forecast_id}: {exc}")
                    continue
                if trade is None:
                    skipped += 1
                else:
                    trades.append(trade)
        drafts.append(
            _Draft(
                group.specialist_id,
                group.kind,
                group.status,
                model,
                group.independent_outcomes,
                tuple(trades),
                skipped,
                tuple(errors),
            )
        )

    tested = sum(1 for item in drafts if item.trades and not item.errors)
    confidence_tests = max(1, tested * 2)
    critical_value = statistics.NormalDist().inv_cdf(
        1 - config.familywise_alpha / confidence_tests
    )
    evaluations = tuple(
        _evaluate_draft(item, config, critical_value) for item in drafts
    )
    digest_payload = {
        "version": cost_registry.version,
        "models": [
            {
                **item.__dict__,
                "kind": item.kind.value,
                "basis": item.basis.value,
                "effective_date": item.effective_date.isoformat(),
            }
            for item in cost_registry.models
        ],
    }
    return EconomicReport(
        cost_registry.version,
        sha256_digest(digest_payload),
        config.familywise_alpha,
        confidence_tests,
        evaluations,
    )


def _latest_independent_scores(
    forecasts_by_id: dict[str, Forecast], scores: tuple[ForecastScore, ...]
) -> dict[tuple[str, ScoreKind], tuple[tuple[Forecast, ForecastScore], ...]]:
    latest: dict[tuple[str, ScoreKind, str], tuple[Forecast, ForecastScore]] = {}
    for score in scores:
        forecast = forecasts_by_id.get(score.forecast_id)
        if forecast is None or forecast.specialist_id != score.specialist_id:
            continue
        key = (
            score.specialist_id,
            score.kind,
            independent_outcome_key(forecast, score.target_time),
        )
        existing = latest.get(key)
        if existing is None or (forecast.generated_at, forecast.forecast_id) > (
            existing[0].generated_at,
            existing[0].forecast_id,
        ):
            latest[key] = (forecast, score)
    grouped: dict[tuple[str, ScoreKind], list[tuple[Forecast, ForecastScore]]] = {}
    for key, item in latest.items():
        grouped.setdefault((key[0], key[1]), []).append(item)
    return {
        key: tuple(
            sorted(
                values,
                key=lambda item: (
                    item[1].target_time,
                    item[0].instrument_id,
                    item[0].forecast_id,
                ),
            )
        )
        for key, values in grouped.items()
    }


def _map_trade(
    forecast: Forecast, score: ForecastScore, model: EconomicCostModel
) -> EconomicTrade | None:
    if score.kind is ScoreKind.RETURN and model.basis is CostBasis.STATIC_BPS:
        predicted = _number(forecast, "predicted_return")
        if predicted == 0:
            return None
        cost = _static_cost(model)
        if abs(predicted) <= cost:
            return None
        direction = "long" if predicted > 0 else "short"
        gross = score.actual if predicted > 0 else -score.actual
        return _trade(forecast, score, direction, gross, cost)

    if score.kind is ScoreKind.FUNDING and model.basis is CostBasis.FORECAST_EXECUTION_BOUND:
        predicted = _number(forecast, "predicted_funding_rate")
        same_signed = forecast.values.get("funding_and_basis_same_signed")
        state = str(forecast.values.get("state", ""))
        if predicted == 0 or same_signed is not True or state == "inside_cost_or_basis_bound":
            return None
        execution_bound = _number(forecast, "execution_bound_bps")
        cost = execution_bound / 10_000 + _static_cost(model)
        if abs(predicted) <= cost:
            return None
        direction = "short_perpetual" if predicted > 0 else "long_perpetual"
        gross = score.actual if predicted > 0 else -score.actual
        return _trade(forecast, score, direction, gross, cost)

    if score.kind is ScoreKind.BINARY and model.basis is CostBasis.BINARY_CONTRACT:
        probability = _number(forecast, "probability")
        market = _number(forecast, "market_probability")
        yes_bid = _number(forecast, "yes_bid")
        yes_ask = _number(forecast, "yes_ask")
        if not 0 <= yes_bid <= yes_ask <= 1:
            raise ValueError("binary executable prices must satisfy 0 <= bid <= ask <= 1")
        if probability == market:
            return None
        if probability > market:
            direction = "buy_yes"
            fair_price = market
            execution_price = yes_ask
            payout = score.actual
        else:
            direction = "buy_no"
            fair_price = 1 - market
            execution_price = 1 - yes_bid
            payout = 1 - score.actual
        spread_cost = execution_price - fair_price
        if spread_cost < 0:
            raise ValueError("binary execution price improves on midpoint unexpectedly")
        fee = _rounded_binary_fee(execution_price, model)
        cost = spread_cost + fee + _static_cost(model)
        if abs(probability - market) <= cost:
            return None
        gross = payout - fair_price
        return _trade(forecast, score, direction, gross, cost)

    raise ValueError(
        f"no economic payoff mapping for {score.kind.value}/{model.basis.value}"
    )


def _trade(
    forecast: Forecast,
    score: ForecastScore,
    direction: str,
    gross: float,
    cost: float,
) -> EconomicTrade:
    if not math.isfinite(gross) or not math.isfinite(cost) or cost < 0:
        raise ValueError("economic returns and costs must be finite and costs nonnegative")
    return EconomicTrade(
        forecast.forecast_id,
        score.target_time,
        direction,
        gross,
        cost,
        gross - cost,
        gross - 2 * cost,
    )


def _static_cost(model: EconomicCostModel) -> float:
    return (
        model.fee_bps + model.spread_bps + model.slippage_bps + model.latency_bps
    ) / 10_000


def _rounded_binary_fee(price: float, model: EconomicCostModel) -> float:
    raw = model.binary_fee_coefficient * price * (1 - price)
    increment = model.binary_fee_increment
    return math.ceil((raw - 1e-12) / increment) * increment


def _number(forecast: Forecast, field: str) -> float:
    value = forecast.values.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"forecast is missing finite economic field {field}")
    return float(value)


def _evaluate_draft(
    draft: _Draft, config: EconomicGateConfig, critical_value: float
) -> EconomicEvaluation:
    if draft.forecast_status is not EdgeStatus.CANDIDATE:
        return _empty_evaluation(
            draft,
            EconomicStatus.BLOCKED,
            (f"forecast gate is {draft.forecast_status.value}",),
        )
    if draft.model is None:
        return _empty_evaluation(
            draft,
            EconomicStatus.UNSUPPORTED,
            ("no pre-registered economic cost model",),
        )
    if draft.errors:
        return _empty_evaluation(
            draft,
            EconomicStatus.UNSUPPORTED,
            ("economic payoff mapping failed", *draft.errors[:5]),
        )
    if not draft.trades:
        return _empty_evaluation(
            draft,
            EconomicStatus.COLLECTING,
            ("no forecasts clear their pre-registered execution cost",),
        )

    net = [item.net_return for item in draft.trades]
    stressed = [item.doubled_cost_net_return for item in draft.trades]
    mean_net = statistics.fmean(net)
    mean_stressed = statistics.fmean(stressed)
    net_lower = _lower_confidence_bound(net, critical_value)
    stressed_lower = _lower_confidence_bound(stressed, critical_value)
    win_rate = sum(value > 0 for value in net) / len(net)
    stressed_win_rate = sum(value > 0 for value in stressed) / len(stressed)
    reasons: list[str] = []
    if len(net) < config.min_trades:
        reasons.append(f"needs {config.min_trades - len(net)} more economic trades")
    if mean_net <= 0:
        reasons.append("mean net return is not positive")
    if mean_stressed <= 0:
        reasons.append("mean return fails doubled-cost stress")
    if net_lower is None or net_lower <= 0:
        reasons.append("family-wise net-return confidence bound is not positive")
    if stressed_lower is None or stressed_lower <= 0:
        reasons.append("doubled-cost confidence bound is not positive")
    if win_rate < config.minimum_win_rate:
        reasons.append("net win rate is below the gate")
    if stressed_win_rate < config.minimum_win_rate:
        reasons.append("doubled-cost win rate is below the gate")
    enough = len(net) >= config.min_trades
    status = (
        EconomicStatus.CANDIDATE
        if enough and not reasons
        else EconomicStatus.REJECTED
        if enough
        else EconomicStatus.COLLECTING
    )
    return EconomicEvaluation(
        draft.specialist_id,
        draft.kind,
        status,
        draft.model.model_id,
        draft.eligible_forecasts,
        len(draft.trades),
        draft.skipped_signals,
        statistics.fmean(item.gross_return for item in draft.trades),
        statistics.fmean(item.assumed_cost for item in draft.trades),
        mean_net,
        net_lower,
        win_rate,
        mean_stressed,
        stressed_lower,
        stressed_win_rate,
        _max_full_notional_drawdown(net),
        _max_full_notional_drawdown(stressed),
        tuple(reasons),
    )


def _empty_evaluation(
    draft: _Draft, status: EconomicStatus, reasons: tuple[str, ...]
) -> EconomicEvaluation:
    return EconomicEvaluation(
        draft.specialist_id,
        draft.kind,
        status,
        draft.model.model_id if draft.model else None,
        draft.eligible_forecasts,
        len(draft.trades),
        draft.skipped_signals,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        reasons,
    )


def _lower_confidence_bound(
    values: list[float], critical_value: float
) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    demeaned = [value - mean for value in values]
    count = len(values)
    bandwidth = min(count - 1, max(1, int(4 * (count / 100) ** (2 / 9))))
    variance = sum(value**2 for value in demeaned) / count
    for lag in range(1, bandwidth + 1):
        covariance = sum(
            demeaned[index] * demeaned[index - lag]
            for index in range(lag, count)
        ) / count
        variance += 2 * (1 - lag / (bandwidth + 1)) * covariance
    standard_error = math.sqrt(max(0.0, variance) / count)
    return mean - critical_value * standard_error


def _max_full_notional_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = equity
    maximum = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak if peak > 0 else math.inf)
    return maximum
