"""Example: Hyperparameter tuning with Atuna."""

from atuna import (
    TunaConfig,
    Tuna,
    model_registry,
    TrainingConfig,
    HyperparpamConfig,
)

# Configure the model
# model = model_registry["unsloth/Qwen3-0.6B-GGUF"]
model = model_registry["unsloth/Qwen3-4B-Instruct-2507"]
exp_config = TunaConfig(
    model_cfg=model,
    dataset="251105_hawara_training.parquet",
    dataset_sample=0.3,
    # dataset="fka/awesome-chatgpt-prompts",
    max_seq_length=2048,
    precision=4,
    cache_dir="./hf_cache",
)

# Create Tuna instance for hyperparameter tuning
hyper_exp = Tuna(config=exp_config)

# Define evaluation prompts to track during training
eval_prompts = [
    "Was kannst du mir über den Stephansdom in Wien erzählen?",
    "Was kannst du mir über den Stephansdom in Wien, auf wienerisch erzählen?",
]

# Configure training parameters
run_config = TrainingConfig(
    num_train_epochs=2,
    eval_epochs=0.25,
    batch_size=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=16,
    evaluation_prompts=eval_prompts,
    data_sample=0.3,  # Use only 30% of data for faster experimentation
)

# Configure hyperparameter search space
hyper_config = HyperparpamConfig(
    n_trials=10,
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

# Run hyperparameter optimization
study = hyper_exp.hyperparam_tune(
    study_name="MyFineTuneTest-3",
    train_config=run_config,
    hyper_config=hyper_config,
)

print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_trial.value}")
print(f"Best parameters: {study.best_trial.params}")

hyper_exp.stop_dashboards()
