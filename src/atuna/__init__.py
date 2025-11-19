"""Atuna: Fine-tuning assistant for large language models."""

from .core import Atuna
from .config import (
    AtunaConfig,
    TrainingConfig,
    HyperConfig,
    ModelConfig,
)
from .models import (
    TrainingResult,
    TrainingPoint,
    TrainingEvaluationPoint,
    StopReason,
    MemoryInfo,
    HyperRun,
)
from .registry import model_registry

__version__ = "0.3.3"
__author__ = "Pasieka Manuel, manuel.pasieka@protonmail.ch"

__all__ = [
    "Atuna",
    "AtunaConfig",
    "TrainingConfig",
    "HyperConfig",
    "ModelConfig",
    "model_registry",
    "TrainingResult",
    "TrainingPoint",
    "TrainingEvaluationPoint",
    "StopReason",
    "MemoryInfo",
    "HyperRun",
]
