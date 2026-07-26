"""Agent contracts. Research agents have no execution permissions."""

from trading_bot.agents.base import ReplayContext, Specialist

__all__ = ["ReplayContext", "Specialist"]
from trading_bot.agents.option_volatility import OptionVolatilitySpecialist
from trading_bot.agents.breakout import CryptoRangeBreakoutSpecialist
from trading_bot.agents.crypto_momentum import CryptoIntradayMomentumSpecialist
from trading_bot.agents.perpetual import PerpetualFundingBasisSpecialist
from trading_bot.agents.prediction import (
    AdjustedPredictionMarketCalibrationSpecialist,
    FastPredictionSettlementSpecialist,
    PredictionMarketCalibrationSpecialist,
)

__all__ = [
    "OptionVolatilitySpecialist",
    "CryptoRangeBreakoutSpecialist",
    "CryptoIntradayMomentumSpecialist",
    "PerpetualFundingBasisSpecialist",
    "PredictionMarketCalibrationSpecialist",
    "AdjustedPredictionMarketCalibrationSpecialist",
    "FastPredictionSettlementSpecialist",
]
