"""Data models for Tuna fine-tuning assistant."""

from enum import StrEnum
from pydantic import BaseModel, Field
import optuna
from typing import Self

from atuna.config import (
    HyperConfig,
    TrainingConfig,
    AtunaConfig,
)


class TrainingPoint(BaseModel):
    """A single training step data point."""

    loss: float
    learning_rate: float
    epoch: float


class TrainingEvaluationPoint(BaseModel):
    """A single evaluation step data point."""

    eval_loss: float
    epoch: float


class StopReason(StrEnum):
    """Reasons why training stopped."""

    EARLY_STOPPING = "EARLY_STOPPING"
    MAX_EPOCHS = "MAX_EPOCHS"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"


class TrainingResult(BaseModel):
    """Results from a training run."""

    success: bool = True
    epochs: float
    duration: float
    stop_reason: StopReason
    training: list[TrainingPoint] = []
    evaluations_loss: list[TrainingEvaluationPoint] = []
    evaluation_prompts_pre_training: list[str] = Field(default_factory=list)
    evaluation_prompts_post_training: list[str] = Field(default_factory=list)

    def add_to_trial(self, trial: optuna.trial.Trial) -> None:
        """Add training results to Optuna trial as user attributes."""
        for k, v in self.model_dump().items():
            trial.set_user_attr(key=k, value=v)

    @classmethod
    def failed_training(cls) -> Self:
        """Create a TrainingResult for a failed run."""
        return cls(
            success=False,
            epochs=0.0,
            duration=0.0,
            stop_reason=StopReason.FAILED,
        )


class MemoryInfo(BaseModel):
    """GPU memory usage information."""

    reserved_gpu_memory: float
    max_memory: float

    def used_memory(self) -> float:
        """Calculate percentage of memory used."""
        return (self.reserved_gpu_memory / self.max_memory) * 100.0


class HyperRunStatus(StrEnum):
    INITIALIZED = "INITIALIZED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"


class HyperRun(BaseModel):
    name: str
    description: str = ""
    status: HyperRunStatus = HyperRunStatus.INITIALIZED
    config: HyperConfig
    trials: list[tuple[TrainingConfig, AtunaConfig, TrainingResult]] = Field(
        default_factory=list
    )
    best_trial: int = 0

    def calculate_best_trial(self) -> int:
        """Get the index of the best trial based on evaluation loss."""

        if len(self.trials) == 0:
            raise ValueError("No trials available to determine the best trial.")

        # Determine best trial based on minimum evaluation loss
        best_trial = 0
        best_eval_loss = float("inf")
        for idx, trial in enumerate(self.trials):
            min_eval_loss = min([ep.eval_loss for ep in trial[2].evaluations_loss])
            if min_eval_loss < best_eval_loss:
                best_eval_loss = min_eval_loss
                best_trial = idx

        return best_trial
