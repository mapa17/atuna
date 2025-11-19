"""Example: Hyperparameter tuning with Atuna."""

from loguru import logger

from atuna import (
    AtunaConfig,
    Atuna,
    model_registry,
    TrainingConfig,
    HyperConfig,
    HyperRun,
)

# Configure the model
# model = model_registry["unsloth/Qwen3-0.6B-GGUF"]
model = model_registry["unsloth/Qwen3-4B-Instruct-2507"]
exp_config = AtunaConfig(
    model_cfg=model,
    max_seq_length=2048,
    precision=4,
    cache_dir="./hf_cache",
)

# Create Tuna instance for hyperparameter tuning
hyper_exp = Atuna(config=exp_config)

# Define evaluation prompts to track during training
eval_prompts = [
    "Was kannst du mir über den Stephansdom in Wien erzählen?",
    "Was kannst du mir über den Stephansdom in Wien, auf wienerisch erzählen?",
]

# Configure training parameters
run_config = TrainingConfig(
    dataset="251105_hawara_training.parquet",
    dataset_sample=0.05,  # use only 10% of the dataset for faster testing
    # dataset="fka/awesome-chatgpt-prompts",
    num_train_epochs=2,
    eval_epochs=0.25,
    batch_size=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=16,
    evaluation_prompts=eval_prompts,
)

# Configure hyperparameter search space
hyper_config = HyperConfig(
    n_trials=1,
    learning_rate=[1e-5, 5e-5, 7e-5, 1e-4],
    weight_decay=[0.001, 0.01, 0.1],
    peft_r=[16, 32],
    lora_alpha=[32, 50, 64],
    enable_slora=True,
)

# Start dashboards
# Optuna: http://127.0.0.1:8080
# TensorBoard: http://127.0.0.1:6006
hyper_exp.start_dashboards()

run = HyperRun(
    name="MyFineTuneTest-3",
    description="Hyperparameter tuning for fine-tuning Qwen-4B on Hawara dataset",
    config=hyper_config,
)

# Run hyperparameter optimization
study = hyper_exp.hyperparam_tune(
    run=run,
    train_config=run_config,
)

logger.info("============\nHyperparameter tuning completed!\n============")
logger.info(f"Best trial: {study.best_trial.number}")
logger.info(f"Best value: {study.best_trial.value}")
logger.info(f"Best parameters: {study.best_trial.params}")

# Use the best hyperparameter configuration to perform a final training run
training_cfg, tuna_cfg, _ = run.trials[run.best_trial]
training_cfg.dataset_sample = 1.0  # use full dataset for final training
tuna_cfg.workspace = "./atuna_workspace/final_training"

logger.info(f"For final training, using config: {tuna_cfg=}, {training_cfg=}")

optimal_finetune = Atuna(config=tuna_cfg)
optimal_training_results = optimal_finetune.train(training_cfg)
logger.info("Final training results:", optimal_training_results)


hyper_exp.stop_dashboards()
