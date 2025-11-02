from typing import Optional, Union, Any, cast
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
    TextStreamer,
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
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    # load_in_16bit: bool = False
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
    peft_use_rslora: bool = False
    cache_dir: Optional[str] = None
    temp_dir: str = "./tuna_temp"


class TrainingConfig(BaseSettings):
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    response_only: bool = False
    seed: Optional[int] = None
    # eval_steps: Optional[int] = None
    eval_epochs: Optional[int] = None
    enable_early_stopping: bool = True

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

        output_dir = tuna_config.temp_dir + "/output"

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
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            report_to="none",
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
        )


class HyperparpamConfig(BaseSettings):
    n_trials: int
    learning_rate: list[float] = Field(default_factory=list)
    peft_r: list[int] = Field(default_factory=list)
    lora_alpha: list[int] = Field(default_factory=list)

    learning_rate_min_max: Optional[tuple[float, float]] = None
    peft_r_min_max: Optional[tuple[int, int]] = None
    lora_alpha_min_max: Optional[tuple[int, int]] = None

    def get_hpo_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        if self.learning_rate_min_max:
            lr = trial.suggest_float(
                "learning_rate",
                self.learning_rate_min_max[0],
                self.learning_rate_min_max[1],
            )
        elif self.learning_rate:
            lr = trial.suggest_categorical("learning_rate", self.learning_rate)

        if self.peft_r_min_max:
            r = trial.suggest_int(
                "peft_r", self.peft_r_min_max[0], self.peft_r_min_max[1]
            )
        elif self.peft_r:
            r = trial.suggest_categorical("peft_r", self.peft_r)

        if self.lora_alpha_min_max:
            alpha = trial.suggest_int(
                "lora_alpha",
                self.lora_alpha_min_max[0],
                self.lora_alpha_min_max[1],
            )
        elif self.lora_alpha:
            alpha = trial.suggest_categorical("lora_alpha", self.lora_alpha)

        return {
            "learning_rate": lr,
            "peft_r": r,
            "lora_alpha": alpha,
        }


class TrainingPoint(BaseModel):
    loss: float
    learning_rate: float
    epoch: float


class TrainingEvaluationPoint(BaseModel):
    eval_loss: float
    epoch: float


class StopReason(str, Enum):
    EARLY_STOPPING = "EARLY_STOPPING"
    MAX_STEPS = "MAX_STEPS"


class TrainingResult(BaseModel):
    epochs: float
    duration: float
    stop_reason: StopReason
    training: list[TrainingPoint] = []
    evaluations: list[TrainingEvaluationPoint] = []


class MemoryInfo(BaseModel):
    reserved_gpu_memory: float
    max_memory: float

    def used_memory(self) -> float:
        return (self.reserved_gpu_memory / self.max_memory) * 100.0


# Tuna is a fine-tuner friendly helper
class Tuna:
    def __init__(self, config: TunaConfig):
        self.config = config
        self.data: Optional[DatasetDict] = None
        self.model: Union[AutoModelForCausalLM, GenerationMixin, None] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[SFTTrainer] = None
        self.hyper_trainer: Optional[SFTTrainer] = None

    def _model_init(
        self,
    ) -> tuple[Union[AutoModelForCausalLM, GenerationMixin], PreTrainedTokenizer]:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_cfg.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            # load_in_16bit=self.config.load_in_16bit,
            full_finetuning=self.config.full_finetuning,
            cache_dir=self.config.cache_dir,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.peft_r,
            target_modules=self.config.peft_target_modules,
            lora_alpha=self.config.peft_lora_alpha,
            lora_dropout=self.config.peft_lora_dropout,
            bias=self.config.peft_bias,
            use_gradient_checkpointing=self.config.peft_use_gradient_checkpointing,
            random_state=self.config.peft_random_state,
            use_rslora=self.config.peft_use_rslora,
            loftq_config=None,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.config.model_cfg.chat_template,
        )
        return model, tokenizer

    @staticmethod
    def _load_data(path: str, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        data = pd.read_csv(path)
        data = data.sample(frac=0.1, random_state=3407).reset_index(drop=True)
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

    def evaluate(self, prompts: list[str], max_tokens: int = 300) -> list[str]:
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            text = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,  # Must add for generation
            )

            _ = self.model.generate(  # type: ignore[arg-type]
                **self.tokenizer(text, return_tensors="pt").to("cuda"),  # type: ignore[arg-type]
                max_new_tokens=max_tokens,
                temperature=self.config.model_cfg.temperature,
                top_p=self.config.model_cfg.top_p,
                top_k=self.config.model_cfg.top_k,  # For non thinking
                streamer=TextStreamer(self.tokenizer, skip_prompt=False),  # type: ignore[arg-type]
            )

        return []

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
            new_model, new_tokenizer = self._model_init()
            self.model = new_model if self.model is None else self.model
            self.tokenizer = new_tokenizer if self.tokenizer is None else self.tokenizer
        if self.data is None:
            self.data = self._load_data(self.config.dataset_path, self.tokenizer)

        result = self._train(
            mode=self.model,
            tokenizer=self.tokenizer,
            data=self.data,
            training_config=config,
            tuna_config=self.config,
        )
        self.training_result = result

        return result

    def hyper_train(
        self,
        training_config: TrainingConfig,
        hyper_config: HyperparpamConfig,
        trial: optuna.trial.Trial,
    ) -> float:
        # Modify tuna_config to ensure each study trial is stored in a different folder
        self.config.temp_dir = (
            f"{self.original_temp_dir}/studies/{trial.study.study_name}/{trial.number}"
        )

        logger.debug(
            f"Starting trial {trial.number} of study {trial.study}, stored in {self.config.temp_dir}"
        )
        # Let Optuna suggest the hyperparameters
        params = hyper_config.get_hpo_search_space(trial)
        # Create a TrainingConfig with the suggested hyperparameters
        config = training_config.model_copy(deep=True, update=params)
        logger.debug(f"Training config: {config.model_dump()}")

        # Train the model with the current hyperparameters
        self.model = None  # Reset model
        self.tokenizer = None
        # Keep tokenizer the same to avoid processing training data again
        result = self.train(config)
        trial.set_user_attr("result", result)

        # Return the evaluation loss for optuna optimization
        eval_losses = [eval_point.eval_loss for eval_point in result.evaluations]
        return min(eval_losses) if eval_losses else float("inf")

    @staticmethod
    def _train(
        mode: AutoModelForCausalLM | GenerationMixin,
        tokenizer: PreTrainedTokenizer,
        data: DatasetDict,
        training_config: TrainingConfig,
        tuna_config: TunaConfig,
    ) -> TrainingResult:
        start_time = time.time()
        # start_mem = Tuna.get_mem_info()

        trainer = Tuna._get_trainer(
            model=mode,
            tokenizer=tokenizer,
            data=data,
            train_config=training_config,
            tuna_config=tuna_config,
        )
        trainer.train()

        train_points, eval_points = Tuna._log_history_to_points(trainer)

        if train_points[-1].epoch >= training_config.num_train_epochs:
            stop_reason = StopReason.MAX_STEPS
        else:
            stop_reason = StopReason.EARLY_STOPPING

        # end_mem = Tuna.get_mem_info()
        end_time = time.time()

        return TrainingResult(
            epochs=float(trainer.state.epoch),
            duration=end_time - start_time,
            stop_reason=stop_reason,
            training=train_points,
            evaluations=eval_points,
        )

    @staticmethod
    def compute_objective(metric: dict[str, float]) -> float:
        return metric["eval_loss"]

    def hyperparam_tune(
        self,
        study_name: str,
        train_config: TrainingConfig,
        hyper_config: HyperparpamConfig,
    ) -> optuna.study.Study:
        # Ensure that the study_name is a valid folder name
        try:
            os.makedirs(
                f"{self.config.temp_dir}/studies/{study_name}",
                exist_ok=True,
            )
        except Exception:
            logger.warning(
                f"Study name '{study_name}' is not a valid folder name. Using default name 'default_study' instead."
            )
            study_name = "default_study"
        # Store original temp dir because we overwrite it for each trial
        self.original_temp_dir = self.config.temp_dir
        study = optuna.create_study(study_name=study_name, direction="minimize")

        logger.info(
            f"Starting hyperparameter tuning for study {study_name} with {hyper_config.n_trials} trials."
        )
        study.optimize(
            func=lambda trial: self.hyper_train(
                training_config=train_config, hyper_config=hyper_config, trial=trial
            ),
            n_trials=hyper_config.n_trials,
        )

        best_trial_path = f"{self.original_temp_dir}/hyperparameter_study/{study_name}/{study.best_trial.number}"
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
