"""Data models for Tuna fine-tuning assistant."""

from enum import Enum
from pydantic import BaseModel, Field
import optuna


class TrainingPoint(BaseModel):
    """A single training step data point."""

    loss: float
    learning_rate: float
    epoch: float


class TrainingEvaluationPoint(BaseModel):
    """A single evaluation step data point."""

    eval_loss: float
    epoch: float


class StopReason(str, Enum):
    """Reasons why training stopped."""

    EARLY_STOPPING = "EARLY_STOPPING"
    MAX_EPOCHS = "MAX_EPOCHS"
    UNKNOWN = "UNKNOWN"


class TrainingResult(BaseModel):
    """Results from a training run."""

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


class MemoryInfo(BaseModel):
    """GPU memory usage information."""

    reserved_gpu_memory: float
    max_memory: float

    def used_memory(self) -> float:
        """Calculate percentage of memory used."""
        return (self.reserved_gpu_memory / self.max_memory) * 100.0
