from typing import Literal, Optional, Union, Any, cast
from enum import Enum
import time
import os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import torch
from datasets import Dataset, DatasetDict
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)

from transformers.generation.utils import GenerationMixin
import optuna
from loguru import logger


class ModelConfig(BaseSettings):
    model_name: str
    chat_template: str
    instruction_part: str
    response_part: str
    temperature: float
    top_p: float
    top_k: int


model_registry = {
    "unsloth/Qwen3-4B-Instruct-2507": ModelConfig(
        model_name="unsloth/Qwen3-4B-Instruct-2507",
        chat_template="qwen3-instruct",
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        temperature=0.7,
        top_p=0.8,
        top_k=20,  # For non thinking
    ),
    "unsloth/Qwen3-0.6B-GGUF": ModelConfig(
        model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        chat_template="qwen3-instruct",
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        temperature=0.7,
        top_p=0.8,
        top_k=20,  # For non thinking
    ),
}


class TunaConfig(BaseSettings):
    model_cfg: ModelConfig
    dataset_path: str
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
    peft_random_state: int = 3407
    use_rslora: bool = True
    cache_dir: Optional[str] = None
    workspace: str = "./atuna_workspace"

    def model_post_init(self, __context: Any) -> None:
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
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    response_only: bool = False
    seed: Optional[int] = None
    # eval_steps: Optional[int] = None
    eval_epochs: Optional[float] = None
    enable_early_stopping: bool = True
    evaluation_prompts: list[str] = Field(default_factory=list)
    weight_decay: float = 0.01
    data_sample: float = 1.0

    def SFTConfig(self, tuna_config: TunaConfig) -> SFTConfig:
        if self.seed:
            seed = self.seed
        else:
            seed = 3407

        # eval_steps = self.eval_steps
        eval_steps = self.eval_epochs if self.eval_epochs else 0
        if self.eval_epochs:
            eval_strategy = "epoch"
        # elif self.eval_steps:
        #    eval_strategy = "steps"
        else:
            eval_strategy = "no"

        output_dir = tuna_config.workspace + "/checkpoints"
        logging_dir = tuna_config.workspace + "/logging"

        if self.enable_early_stopping:
            save_strategy = "best"  # save model every N steps
            save_steps = eval_steps  # how many steps until we save the model
            save_total_limit = 3  # keep ony 3 saved checkpoints to save disk space
            load_best_model_at_end = True  # MUST USE for early stopping
            metric_for_best_model = "eval_loss"  # metric we want to early stop on
            greater_is_better = False  # the lower the eval loss, the better
        else:
            save_strategy = "no"
            save_steps = 0
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
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to="tensorboard",  # Change from "none" to "tensorboard"
            logging_dir=logging_dir,  # TensorBoard log directory
            logging_steps=1,
        )


class HyperparpamConfig(BaseSettings):
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
        self, trial: optuna.Trial, training_cfg: TrainingConfig, tuna_cfg: TunaConfig
    ) -> tuple[TunaConfig, TrainingConfig]:
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


class TrainingPoint(BaseModel):
    loss: float
    learning_rate: float
    epoch: float


class TrainingEvaluationPoint(BaseModel):
    eval_loss: float
    epoch: float


class StopReason(str, Enum):
    EARLY_STOPPING = "EARLY_STOPPING"
    MAX_EPOCHS = "MAX_EPOCHS"
    UNKNOWN = "UNKNOWN"


class TrainingResult(BaseModel):
    epochs: float
    duration: float
    stop_reason: StopReason
    training: list[TrainingPoint] = []
    evaluations_loss: list[TrainingEvaluationPoint] = []
    evaluation_prompts_pre_training: list[str] = Field(default_factory=list)
    evaluation_prompts_post_training: list[str] = Field(default_factory=list)

    def add_to_trial(self, trial: optuna.trial.Trial):
        for k, v in self.model_dump().items():
            trial.set_user_attr(key=k, value=v)


class MemoryInfo(BaseModel):
    reserved_gpu_memory: float
    max_memory: float

    def used_memory(self) -> float:
        return (self.reserved_gpu_memory / self.max_memory) * 100.0


# Tuna is a fine-tuner friendly helper
class Tuna:
    def __init__(self, config: TunaConfig):
        self.config: TunaConfig = config
        self.data: Optional[DatasetDict] = None
        self.model: Union[AutoModelForCausalLM, GenerationMixin, None] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[SFTTrainer] = None
        self.hyper_trainer: Optional[SFTTrainer] = None

    @staticmethod
    def _model_init(
        config: TunaConfig,
    ) -> tuple[Union[AutoModelForCausalLM, GenerationMixin], PreTrainedTokenizer]:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_cfg.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            full_finetuning=config.full_finetuning,
            cache_dir=config.cache_dir,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.peft_r,
            target_modules=config.peft_target_modules,
            lora_alpha=config.peft_lora_alpha,
            lora_dropout=config.peft_lora_dropout,
            bias=config.peft_bias,
            use_gradient_checkpointing=config.peft_use_gradient_checkpointing,
            random_state=config.peft_random_state,
            use_rslora=config.use_rslora,
            loftq_config=None,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=config.model_cfg.chat_template,
        )
        return model, tokenizer

    @staticmethod
    def _load_data(
        path: str, tokenizer: PreTrainedTokenizer, sample: float = 1.0
    ) -> DatasetDict:
        data = pd.read_csv(path)
        if sample < 1.0:
            data = data.sample(frac=sample, random_state=3407).reset_index(drop=True)
        td = []
        for idx, record in data.iterrows():
            td.append(
                [
                    {"role": "user", "content": record["request"]},
                    {"role": "assistant", "content": record["response"]},
                ]
            )
        ds = Dataset.from_dict({"text": td})

        def apply_template(example: dict[str, list[Any]]):
            conv = example["text"]
            texts = [
                tokenizer.apply_chat_template(
                    e, tokenize=False, add_generation_prompt=False
                )
                for e in conv
            ]
            return {
                "text": texts,
            }

        ds_ready = ds.map(apply_template, batched=True)

        split_dataset = ds_ready.train_test_split(
            test_size=0.01, shuffle=True, seed=3407
        )

        return split_dataset

    @staticmethod
    def _add_early_stopping(trainer: SFTTrainer):
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,  # How many steps we will wait if the eval loss doesn't decrease
            # For example the loss might increase, but decrease after 3 steps
            early_stopping_threshold=0.0,  # Can set higher - sets how much loss should decrease by until
            # we consider early stopping. For eg 0.01 means if loss was
            # 0.02 then 0.01, we consider to early stop the run.
        )
        trainer.add_callback(early_stopping_callback)

    @staticmethod
    def _get_trainer(
        model: Union[AutoModelForCausalLM, GenerationMixin],
        tokenizer: PreTrainedTokenizer,
        data: DatasetDict,
        train_config: TrainingConfig,
        tuna_config: TunaConfig,
    ) -> SFTTrainer:
        trainer = SFTTrainer(
            model=cast(torch.nn.Module, model),
            tokenizer=tokenizer,  # type: ignore[arg-type]
            train_dataset=data["train"],
            eval_dataset=data["test"],
            args=train_config.SFTConfig(tuna_config),
        )

        if train_config.enable_early_stopping:
            Tuna._add_early_stopping(trainer)

        if train_config.response_only:
            trainer = train_on_responses_only(
                trainer,
                instruction_part=tuna_config.model_cfg.instruction_part,
                response_part=tuna_config.model_cfg.response_part,
            )
        return cast(SFTTrainer, trainer)

    def evaluate_prompts(
        self, prompts: list[str], max_tokens: int = 300, n_samples: int = 1
    ) -> list[str]:
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model and tokenizer must be initialized before evaluation."
            )

        return Tuna._evaluate_prompts(
            prompts,
            self.model,
            self.tokenizer,
            self.config,
            max_tokens=max_tokens,
            n_samples=n_samples,
        )

    @staticmethod
    def _evaluate_prompts(
        prompts: list[str],
        model: Union[AutoModelForCausalLM, GenerationMixin],
        tokenizer: PreTrainedTokenizer,
        config: TunaConfig,
        max_tokens: int = 300,
        n_samples: int = 3,
    ) -> list[str]:
        results = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            text = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,  # Must add for generation
            )

            inputs: dict[str, torch.Tensor] = tokenizer(text, return_tensors="pt").to(
                "cuda"
            )

            for _ in range(n_samples):
                generated_tokens = model.generate(  # type: ignore[attr-defined]
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    temperature=config.model_cfg.temperature,
                    top_p=config.model_cfg.top_p,
                    top_k=config.model_cfg.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Generated tokens contains input message + new tokens, we only want new tokens
                input_length = inputs["input_ids"].shape[1]
                new_tokens = generated_tokens[0][input_length:]

                # 5. Decode tokens back to text
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(generated_text)

        return results

    @staticmethod
    def get_mem_info() -> MemoryInfo:
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        return MemoryInfo(
            reserved_gpu_memory=reserved_gpu_memory,
            max_memory=max_memory,
        )

    def train(self, config: TrainingConfig) -> TrainingResult:
        # Allow for partial model/tokenizer reuse
        if self.model is None or self.tokenizer is None:
            new_model, new_tokenizer = self._model_init(self.config)
            self.model = new_model if self.model is None else self.model
            self.tokenizer = new_tokenizer if self.tokenizer is None else self.tokenizer
        if self.data is None:
            self.data = self._load_data(
                self.config.dataset_path, self.tokenizer, sample=config.data_sample
            )

        result = self._train(
            model=self.model,
            tokenizer=self.tokenizer,
            data=self.data,
            training_config=config,
            tuna_config=self.config,
        )
        self.training_result = result

        return result

    def _hyper_train(
        self,
        training_config: TrainingConfig,
        hyper_config: HyperparpamConfig,
        trial: optuna.trial.Trial,
    ) -> float:
        # Modify tuna_config to ensure each study trial is stored in a different folder
        self.config.workspace = (
            f"{self.original_workspace}/studies/{trial.study.study_name}/{trial.number}"
        )

        logger.debug(
            f"Starting trial {trial.number} of study {trial.study}, stored in {self.config.workspace}"
        )
        # Let Optuna suggest the hyperparameters
        # Create a TrainingConfig with the suggested hyperparameters
        tuna_cfg, training_cfg = hyper_config.build_configs(
            trial=trial, tuna_cfg=self.config, training_cfg=training_config
        )
        model, tokenizer = self._model_init(tuna_cfg)
        if self.data is None:
            self.data = self._load_data(
                tuna_cfg.dataset_path, tokenizer, sample=training_cfg.data_sample
            )

        result = self._train(
            model=model,
            tokenizer=tokenizer,
            data=self.data,
            training_config=training_cfg,
            tuna_config=tuna_cfg,
        )
        self.training_result = result

        # Add training results to optuna trial
        result.add_to_trial(trial)

        # Return the evaluation loss for optuna optimization
        eval_losses = [eval_point.eval_loss for eval_point in result.evaluations_loss]
        return min(eval_losses) if eval_losses else float("inf")

    @staticmethod
    def _determine_stop_reason(trainer: SFTTrainer) -> StopReason:
        """Determine why training stopped by examining trainer state."""

        # Look for early stopping callback in trainer
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                if hasattr(callback, "early_stopping_patience_counter"):
                    if (
                        callback.early_stopping_patience_counter
                        >= callback.early_stopping_patience
                    ):
                        return StopReason.EARLY_STOPPING

        # Check if we reached max epochs
        if trainer.state.epoch and trainer.state.epoch >= trainer.args.num_train_epochs:
            return StopReason.MAX_EPOCHS

        return StopReason.UNKNOWN

    @staticmethod
    def _train(
        model: AutoModelForCausalLM | GenerationMixin,
        tokenizer: PreTrainedTokenizer,
        data: DatasetDict,
        training_config: TrainingConfig,
        tuna_config: TunaConfig,
    ) -> TrainingResult:
        start_time = time.time()
        # start_mem = Tuna.get_mem_info()

        trainer = Tuna._get_trainer(
            model=model,
            tokenizer=tokenizer,
            data=data,
            train_config=training_config,
            tuna_config=tuna_config,
        )

        evaluation_prompts_pre_training = []
        evaluation_prompts_post_training = []

        if training_config.evaluation_prompts:
            evaluation_prompts_pre_training = Tuna._evaluate_prompts(
                prompts=training_config.evaluation_prompts,
                model=model,
                tokenizer=tokenizer,
                config=tuna_config,
            )

        trainer.train()

        if training_config.evaluation_prompts:
            evaluation_prompts_post_training = Tuna._evaluate_prompts(
                prompts=training_config.evaluation_prompts,
                model=model,
                tokenizer=tokenizer,
                config=tuna_config,
            )

        train_points, eval_points = Tuna._log_history_to_points(trainer)

        stop_reason = Tuna._determine_stop_reason(trainer)

        # end_mem = Tuna.get_mem_info()
        end_time = time.time()

        return TrainingResult(
            epochs=float(trainer.state.epoch) if trainer.state.epoch else 0.0,
            duration=end_time - start_time,
            stop_reason=stop_reason,
            training=train_points,
            evaluations_loss=eval_points,
            evaluation_prompts_pre_training=evaluation_prompts_pre_training,
            evaluation_prompts_post_training=evaluation_prompts_post_training,
        )

    @staticmethod
    def compute_objective(metric: dict[str, float]) -> float:
        return metric["eval_loss"]

    def _setup_hyperparam_tune(self, study_name: str) -> optuna.study.Study:
        # Ensure that the study_name is a valid folder name
        try:
            try:
                os.removedirs(f"{self.config.workspace}/studies/{study_name}")
            except Exception:
                logger.debug("Study folder does not exist yet, no need to remove.")
            os.makedirs(
                f"{self.config.workspace}/studies/{study_name}",
                exist_ok=True,
            )
        except Exception:
            study_name = "default_study"
            logger.warning(
                f"Study name '{study_name}' is not a valid folder name. Using default name 'default_study' instead."
            )
        # Store original temp dir because we overwrite it for each trial
        self.original_workspace = os.path.abspath(self.config.workspace)

        # Store optuna in a sqlite database in the temp dir
        storage_name = f"sqlite:///{self.original_workspace}/optuna_studies.db"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_name,
            load_if_exists=True,
        )

        logger.info(
            f"Optuna study '{study_name}' created with storage '{storage_name}'.\n"
            f"Open dashboard with: > optuna-dashboard sqlite:///{self.original_workspace}/optuna_studies.db\n"
            f"Track individual trainings with tensorboard: > tensorboard --logdir {self.original_workspace}/logs\n"
        )

        return study

    def hyperparam_tune(
        self,
        study_name: str,
        train_config: TrainingConfig,
        hyper_config: HyperparpamConfig,
    ) -> optuna.study.Study:
        logger.info(
            f"Starting hyperparameter tuning for study {study_name} with {hyper_config.n_trials} trials."
        )
        study = self._setup_hyperparam_tune(study_name=study_name)

        study.optimize(
            func=lambda trial: self._hyper_train(
                training_config=train_config, hyper_config=hyper_config, trial=trial
            ),
            n_trials=hyper_config.n_trials,
        )

        best_trial_path = (
            f"{self.original_workspace}/studies/{study_name}/{study.best_trial.number}"
        )
        logger.info(
            f"Best trial: {study.best_trial.number} with value: {study.best_trial.value}. Model stored in {best_trial_path}"
        )
        return study

    @staticmethod
    def _log_history_to_points(
        trainer: SFTTrainer,
    ) -> tuple[list[TrainingPoint], list[TrainingEvaluationPoint]]:
        train_points = []
        eval_points = []

        for log in trainer.state.log_history:
            if "loss" in log:
                tp = TrainingPoint(
                    loss=log["loss"],
                    learning_rate=log["learning_rate"],
                    epoch=log["epoch"],
                )
                train_points.append(tp)
            if "eval_loss" in log:
                ep = TrainingEvaluationPoint(
                    eval_loss=log["eval_loss"],
                    epoch=log["epoch"],
                )
                eval_points.append(ep)

        return train_points, eval_points
