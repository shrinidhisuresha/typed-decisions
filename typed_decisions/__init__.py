"""A System-One-style decision wrapper over open-weights models.

state + typed questions -> calibrated typed answers, in one shared-prefix pass,
generating zero output tokens. See DESIGN.md sections 2 and 5.
"""

from .client import Calibration, Decider, Result, Usage
from .calibration.harness import (
    Prediction,
    Report,
    TemperatureFit,
    diagnose_temperature,
    fit_temperature_from_predictions,
    report,
)
from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from .scoring import ChoiceAnswer, NoulAnswer, ScoreAnswer

__all__ = [
    "Decider",
    "Calibration",
    "Prediction",
    "Report",
    "TemperatureFit",
    "diagnose_temperature",
    "report",
    "fit_temperature_from_predictions",
    "Result",
    "Usage",
    "ChoiceQuestion",
    "NoulQuestion",
    "ScoreQuestion",
    "Question",
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
]
