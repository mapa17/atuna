"""Core Tuna fine-tuning assistant class."""

from typing import Union, Optional, Any, cast
import time
import os
import subprocess  # nosec
import pathlib

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from trl.trainer.sft_trainer import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin
import optuna
from loguru import logger

from .config import TunaConfig, TrainingConfig, HyperparpamConfig
from .models import (
    TrainingResult,
    TrainingPoint,
    TrainingEvaluationPoint,
    StopReason,
    MemoryInfo,
)


class Tuna:
    """Fine-tuning assistant for large language models."""

    def __init__(self, config: TunaConfig):
        """Initialize Tuna with configuration."""
        self.config: TunaConfig = config
        self.data: Optional[DatasetDict] = None
        self.model: Union[AutoModelForCausalLM, GenerationMixin, None] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.trainer: Optional[SFTTrainer] = None
        self.hyper_trainer: Optional[SFTTrainer] = None
        self.training_result: Optional[TrainingResult] = None
        self.original_workspace = (
            pathlib.Path(self.config.workspace).absolute().as_posix()
        )
        self._optuna_process: Optional[subprocess.Popen] = None
        self._tensorboard_process: Optional[subprocess.Popen] = None

    def __del__(self):
        """Called when the object is about to be destroyed."""
        self.stop_dashboards()

    def stop_dashboards(self):
        """Clean up dashboard processes."""
        if self._optuna_process:
            try:
                self._optuna_process.terminate()
                self._optuna_process.wait(timeout=5)  # Wait up to 5 seconds
                logger.info("Optuna Dashboard stopped")
            except Exception as e:
                logger.warning(f"Failed to stop Optuna Dashboard: {e}")
            finally:
                self._optuna_process = None

        if self._tensorboard_process:
            try:
                self._tensorboard_process.terminate()
                self._tensorboard_process.wait(timeout=5)
                logger.info("TensorBoard stopped")
            except Exception as e:
                logger.warning(f"Failed to stop TensorBoard: {e}")
            finally:
                self._tensorboard_process = None

    @staticmethod
    def _model_init(
        config: TunaConfig,
    ) -> tuple[Union[AutoModelForCausalLM, GenerationMixin], PreTrainedTokenizerBase]:
        """Initialize model and tokenizer."""
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
            random_state=config.seed,
            use_rslora=config.use_rslora,
            loftq_config=None,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=config.model_cfg.chat_template,
        )
        return model, tokenizer

    def model_init(self):
        """Initialize model and tokenizer for Tuna instance."""
        model, tokenizer = self._model_init(self.config)
        self.model = model
        self.tokenizer = tokenizer

    def _load_data(
        self,
    ) -> DatasetDict:
        """Load and preprocess training data."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before loading data.")

        tokenizer: PreTrainedTokenizerBase = self.tokenizer

        def apply_template(example: dict[str, list[Any]]):
            if self.config.dataset_text_field not in example:
                raise ValueError(
                    f"Dataset does not contain field '{self.config.dataset_text_field}'. Available fields are: {list(example.keys())}"
                )
            conv = example[self.config.dataset_text_field]
            texts = [
                tokenizer.apply_chat_template(
                    e, tokenize=False, add_generation_prompt=False
                )
                for e in conv
            ]
            return {
                "text": texts,
            }

        # Load dataset from file or dataset hub
        if os.path.exists(self.config.dataset):
            _, ext = os.path.splitext(self.config.dataset)
            ext = ext[1:]
            logger.debug(
                f"Loading dataset from file: {self.config.dataset} with extension: {ext}"
            )
            data = load_dataset(ext, data_files=self.config.dataset, revision="main")  # nosec B615
        else:
            logger.debug(f"Loading dataset from hub: {self.config.dataset}")
            data = load_dataset(self.config.dataset, revision="main")  # nosec B615

        # Apply tokenizer specific chat template and split into test and train
        match data:
            case DatasetDict():
                if len(data.keys()) == 0:
                    raise ValueError("Loaded dataset is empty.")
                if "train" in data and "test" in data:
                    logger.info("Using existing train/test split from dataset.")
                    data["train"] = data["train"].map(apply_template, batched=True)
                    data["test"] = data["test"].map(apply_template, batched=True)
                    split_dataset = data
                else:
                    key = list(data.keys())[0]
                    logger.info(
                        f"Loading {key=} from dataset and creating train/test split."
                    )
                    data_templated = data[key].map(apply_template, batched=True)
                    split_dataset = data_templated.train_test_split(
                        test_size=self.config.dataset_test_size,
                        shuffle=True,
                        seed=self.config.seed,
                    )

            case Dataset():
                logger.info("Loading dataset and creating train/test split.")
                data_templated = data.map(apply_template, batched=True)
                split_dataset = data_templated.train_test_split(
                    test_size=self.config.dataset_test_size,
                    shuffle=True,
                    seed=self.config.seed,
                )
            case IterableDataset():
                raise NotImplementedError("IterableDataset not supported yet.")
            case IterableDatasetDict():
                raise NotImplementedError("IterableDatasetDict not supported yet.")

        # Sample a subset of training and test if required
        if self.config.dataset_sample < 1.0:
            for key in ["train", "test"]:
                n_samples = int(len(split_dataset[key]) * self.config.dataset_sample)
                split_dataset[key] = (
                    split_dataset[key]
                    .shuffle(seed=self.config.seed)
                    .select(range(n_samples))
                )
                logger.info(f"Sampled {n_samples} items from {key} data.")

        return split_dataset

    @staticmethod
    def _add_early_stopping(trainer: SFTTrainer) -> None:
        """Add early stopping callback to trainer."""
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
        tokenizer: PreTrainedTokenizerBase,
        data: DatasetDict,
        train_config: TrainingConfig,
        tuna_config: TunaConfig,
    ) -> SFTTrainer:
        """Create and configure SFT trainer."""
        n_samples = len(data["train"])
        train_config.calculate_eval_steps(n_samples)
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
        """Evaluate model on a list of prompts."""
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
        tokenizer: PreTrainedTokenizerBase,
        config: TunaConfig,
        max_tokens: int = 300,
        n_samples: int = 3,
    ) -> list[str]:
        """Evaluate model on prompts and return generated responses."""
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
        """Get current GPU memory usage information."""
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
        """Train the model with given configuration."""
        # Allow for partial model/tokenizer reuse
        if self.model is None or self.tokenizer is None:
            new_model, new_tokenizer = self._model_init(self.config)
            self.model = new_model if self.model is None else self.model
            self.tokenizer = new_tokenizer if self.tokenizer is None else self.tokenizer

        if self.data is None:
            self.data = self._load_data()

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
        """Train model with hyperparameters suggested by Optuna trial."""
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
            self.data = self._load_data()

        result = self._train(
            model=model,
            tokenizer=tokenizer,
            data=self.data,
            training_config=training_cfg,
            tuna_config=tuna_cfg,
        )
        self.training_result = result

        # Add full training result to trial attributes
        result.add_to_trial(trial)

        trial.set_user_attr(
            "min_eval_loss",
            min([ep.eval_loss for ep in result.evaluations_loss])
            if result.evaluations_loss
            else None,
        )

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
        model: Union[AutoModelForCausalLM, GenerationMixin],
        tokenizer: PreTrainedTokenizerBase,
        data: DatasetDict,
        training_config: TrainingConfig,
        tuna_config: TunaConfig,
    ) -> TrainingResult:
        """Core training logic."""
        start_time = time.time()

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
        """Compute objective value for optimization."""
        return metric["eval_loss"]

    def _setup_hyperparam_tune(self, study_name: str) -> optuna.study.Study:
        """Set up Optuna study for hyperparameter tuning."""
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
        """Run hyperparameter optimization."""
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
        """Convert trainer log history to structured data points."""
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

    def start_optuna_dashboard(
        self,
    ):
        """Start Optuna Dashboard with authentication."""
        storage_url = f"sqlite:///{self.original_workspace}/optuna_studies.db"

        try:
            cmd = [
                "optuna-dashboard",
                storage_url,
                "--port",
                str(self.config.dashboard_optuna_port),
                "--host",
                self.config.dashboard_host,
            ]

            self._optuna_process = subprocess.Popen(cmd)  # nosec B603

            url = f"http://{self.config.dashboard_host}:{self.config.dashboard_optuna_port}"
            logger.info(f"Optuna Dashboard started at: {url}")
            logger.debug(f"Optuna Dashboard PID: {self._optuna_process.pid}")
        except FileNotFoundError:
            logger.error(
                "optuna-dashboard command not found. Install with: pip install optuna-dashboard"
            )
        except Exception as e:
            logger.error(f"Failed to start Optuna Dashboard: {e}")

    def start_tensorboard(self):
        """Start TensorBoard using subprocess for better process control."""
        logdir = f"{self.original_workspace}/logs"
        os.makedirs(logdir, exist_ok=True)

        try:
            cmd = [
                "tensorboard",
                "--logdir",
                logdir,
                "--port",
                str(self.config.dashboard_tb_port),
                "--host",
                self.config.dashboard_host,
                "--reload_interval",
                "30",
            ]

            self._tensorboard_process = subprocess.Popen(cmd)  # nosec B603

            url = f"http://{self.config.dashboard_host}:{self.config.dashboard_tb_port}"
            logger.info(f"TensorBoard started at: {url}")
            logger.debug(f"TensorBoard PID: {self._tensorboard_process.pid}")
        except FileNotFoundError:
            logger.error(
                "tensorboard command not found. Install with: pip install tensorboard"
            )
        except Exception as e:
            logger.error(f"Failed to start TensorBoard: {e}")

    def start_dashboards(
        self,
    ):
        """Start both TensorBoard and Optuna dashboards."""
        self.start_tensorboard()
        self.start_optuna_dashboard()
