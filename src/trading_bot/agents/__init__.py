"""Agent contracts. Research agents have no execution permissions."""

from trading_bot.agents.base import ReplayContext, Specialist

__all__ = ["ReplayContext", "Specialist"]
from trading_bot.agents.option_volatility import OptionVolatilitySpecialist
from trading_bot.agents.breakout import CryptoRangeBreakoutSpecialist
from trading_bot.agents.crypto_momentum import (
    CryptoIntradayMomentumSpecialist,
    CryptoIntradayMomentumV2Specialist,
)
from trading_bot.agents.perpetual import PerpetualFundingBasisSpecialist
from trading_bot.agents.prediction import (
    AdjustedPredictionMarketCalibrationSpecialist,
    FastPredictionSettlementSpecialist,
    FastPredictionSettlementV3Specialist,
    PredictionMarketCalibrationSpecialist,
)

__all__ = [
    "OptionVolatilitySpecialist",
    "CryptoRangeBreakoutSpecialist",
    "CryptoIntradayMomentumSpecialist",
    "CryptoIntradayMomentumV2Specialist",
    "PerpetualFundingBasisSpecialist",
    "PredictionMarketCalibrationSpecialist",
    "AdjustedPredictionMarketCalibrationSpecialist",
    "FastPredictionSettlementSpecialist",
    "FastPredictionSettlementV3Specialist",
]
