"""Configuration classes for Tuna fine-tuning assistant."""

from typing import Literal, Optional, Union, Any
import os
from pydantic import Field
from pydantic_settings import BaseSettings
from trl.trainer.sft_config import SFTConfig
import optuna
from loguru import logger


class ModelConfig(BaseSettings):
    """Configuration for a specific model."""

    model_name: str
    chat_template: str
    instruction_part: str
    response_part: str
    temperature: float
    top_p: float
    top_k: int


class AtunaConfig(BaseSettings):
    """Main configuration for Tuna fine-tuning."""

    model_cfg: ModelConfig
    max_seq_length: int = 2048
    precision: Literal[4, 8, 16] = Field(default=16)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    full_finetuning: bool = False
    peft_r: int = 32
    peft_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.0
    peft_bias: str = "none"
    peft_use_gradient_checkpointing: str = "unsloth"
    seed: int = 3407
    use_rslora: bool = True
    cache_dir: Optional[str] = None
    workspace: str = "./atuna_workspace"
    dashboard_host: str = "127.0.0.1"
    dashboard_tb_port: int = 6006
    dashboard_optuna_port: int = 8080

    def model_post_init(self, __context: Any) -> None:
        """Set quantization flags based on precision."""
        match self.precision:
            case 4:
                self.load_in_4bit = True
                self.load_in_8bit = False
            case 8:
                self.load_in_4bit = False
                self.load_in_8bit = True
            case 16:
                self.load_in_4bit = False
                self.load_in_8bit = False


class TrainingConfig(BaseSettings):
    """Configuration for training parameters."""

    dataset: str
    dataset_sample: float = 1.0
    dataset_text_field: str = "text"
    dataset_test_size: float = 0.01

    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    response_only: bool = False
    seed: Optional[int] = None
    eval_epochs: Optional[float] = None
    eval_steps: Union[int, float] = 0.1
    enable_early_stopping: bool = True
    evaluation_prompts: list[str] = Field(default_factory=list)
    weight_decay: float = 0.01

    def calculate_eval_steps(self, training_data_size: int) -> None:
        """Set eval_steps based on eval_epochs and training data size."""
        steps_per_epoch = training_data_size // (
            self.batch_size * self.gradient_accumulation_steps
        )

        if self.eval_epochs:
            self.eval_steps = int(steps_per_epoch * self.eval_epochs)
        elif self.eval_steps:
            logger.debug(f"Using provided eval_steps: {self.eval_steps}")
        else:
            self.eval_steps = steps_per_epoch // 4
        logger.debug(
            f"Set eval_steps to {self.eval_steps} based on eval_epochs {self.eval_epochs} and training data size {training_data_size}"
        )

    def SFTConfig(self, tuna_config: AtunaConfig) -> SFTConfig:
        """Convert to SFTConfig for the trainer."""
        if self.seed:
            seed = self.seed
        else:
            seed = 3407

        if self.eval_steps <= 0.0:
            eval_strategy = "no"
            eval_steps = 0

        output_dir = tuna_config.workspace + "/checkpoints"
        os.makedirs(output_dir, exist_ok=True)
        logging_dir = tuna_config.workspace + "/logs"
        os.makedirs(logging_dir, exist_ok=True)

        if self.enable_early_stopping:
            save_strategy = "best"  # save model every N steps
            save_total_limit = 3  # keep only 3 saved checkpoints to save disk space
            load_best_model_at_end = True  # MUST USE for early stopping
            metric_for_best_model = "eval_loss"  # metric we want to early stop on
            greater_is_better = False  # the lower the eval loss, the better
        else:
            save_strategy = "no"
            save_total_limit = None
            load_best_model_at_end = False
            metric_for_best_model = None
            greater_is_better = None

        return SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            optim="adamw_8bit",  # Using bitsandbytes optimizer that stores optimizer states in 8bit precision, expanding them on the fly for gradient calculation.
            weight_decay=self.weight_decay,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            seed=seed,
            fp16_full_eval=True,
            per_device_eval_batch_size=self.batch_size,
            eval_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            output_dir=output_dir,
            save_strategy=save_strategy,
            save_steps=eval_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to="tensorboard",  # Choose "none" or "tensorboard"
            logging_dir=logging_dir,  # TensorBoard log directory
            logging_steps=0.1,  # log every 10% of an epoch
        )


class HyperConfig(BaseSettings):
    """Configuration for hyperparameter optimization."""

    n_trials: int
    learning_rate: list[float] = Field(default_factory=list)
    peft_r: list[int] = Field(default_factory=list)
    lora_alpha: list[int] = Field(default_factory=list)
    weight_decay: list[float] = Field(default_factory=list)

    learning_rate_min_max: Optional[tuple[float, float]] = None
    peft_r_min_max: Optional[tuple[int, int]] = None
    lora_alpha_min_max: Optional[tuple[int, int]] = None
    weight_decay_min_max: Optional[tuple[float, float]] = None

    # If overwritten enable loading models in different precisions
    precision: list[Literal[4, 8, 16]] = Field(default_factory=list)

    enable_slora: bool = False

    @staticmethod
    def _minmax(
        trial: optuna.Trial,
        name: str,
        choices: Union[list[float], list[int]],
        min_max: Optional[Union[tuple[float, float], tuple[int, int]]],
    ) -> Any:
        """Helper to choose between an element in choices, or if min_max is set use the trial to suggest a value."""
        if min_max:
            if isinstance(min_max[0], float):
                return trial.suggest_float(name, min_max[0], min_max[1])
            elif isinstance(min_max[0], int) and isinstance(min_max[1], int):
                return trial.suggest_int(name, min_max[0], min_max[1])
        else:
            return trial.suggest_categorical(name, choices)

    @staticmethod
    def _categorical(
        trial: optuna.Trial,
        name: str,
        choices: list[Any],
        default: Union[float, int],
    ) -> Any:
        if len(choices) > 0:
            return trial.suggest_categorical(name, choices)
        return default

    def build_configs(
        self, trial: optuna.Trial, training_cfg: TrainingConfig, tuna_cfg: AtunaConfig
    ) -> tuple[AtunaConfig, TrainingConfig]:
        """Build configurations with hyperparameters suggested by Optuna trial."""
        lr = self._minmax(
            trial, "learning_rate", self.learning_rate, self.learning_rate_min_max
        )
        r = self._minmax(trial, "peft_r", self.peft_r, self.peft_r_min_max)
        alpha = self._minmax(
            trial, "lora_alpha", self.lora_alpha, self.lora_alpha_min_max
        )
        wd = self._minmax(
            trial, "weight_decay", self.weight_decay, self.weight_decay_min_max
        )

        precision = self._categorical(
            trial, "precision", self.precision, tuna_cfg.precision
        )

        if self.enable_slora:
            use_rslora = self._categorical(trial, "use_rslora", [True, False], False)
        else:
            use_rslora = False

        training_cfg = training_cfg.model_copy(deep=True)
        tuna_cfg = tuna_cfg.model_copy(deep=True)

        training_cfg.learning_rate = lr
        training_cfg.weight_decay = wd
        tuna_cfg.peft_r = r
        tuna_cfg.peft_lora_alpha = alpha
        tuna_cfg.use_rslora = use_rslora
        tuna_cfg.precision = precision
        tuna_cfg.model_post_init(None)

        logger.debug(
            f"Build configs: tuna config: {tuna_cfg.model_dump()}, training config: {training_cfg.model_dump()}"
        )

        return tuna_cfg, training_cfg
