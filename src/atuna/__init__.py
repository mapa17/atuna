"""Atuna: Fine-tuning assistant for large language models."""

from .core import Tuna
from .config import (
    TunaConfig,
    TrainingConfig,
    HyperparpamConfig,
    ModelConfig,
)
from .models import (
    TrainingResult,
    TrainingPoint,
    TrainingEvaluationPoint,
    StopReason,
    MemoryInfo,
)
from .registry import model_registry

__version__ = "0.2.1"
__author__ = "Pasieka Manuel, manuel.pasieka@protonmail.ch"

__all__ = [
    "Tuna",
    "TunaConfig",
    "TrainingConfig",
    "HyperparpamConfig",
    "ModelConfig",
    "model_registry",
    "TrainingResult",
    "TrainingPoint",
    "TrainingEvaluationPoint",
    "StopReason",
    "MemoryInfo",
]
