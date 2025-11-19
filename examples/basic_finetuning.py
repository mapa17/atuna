"""Example: Basic fine-tuning with Atuna."""

from atuna import (
    AtunaConfig,
    Atuna,
    model_registry,
    TrainingConfig,
)

# Configure the model
model = model_registry["unsloth/Qwen3-4B-Instruct-2507"]
config = AtunaConfig(
    model_cfg=model,
    max_seq_length=2048,
    precision=16,  # Use 16-bit precision for faster training
    cache_dir="./hf_cache",
)

# Create Tuna instance
tuna = Atuna(config=config)

# Configure training
training_config = TrainingConfig(
    dataset="./data/251015_training_set.csv",
    num_train_epochs=3,
    batch_size=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=16,
    eval_epochs=1.0,  # Evaluate every epoch
    enable_early_stopping=True,
)

# Train the model
result = tuna.train(config=training_config)

print(f"Training completed in {result.duration:.2f} seconds")
print(f"Final epochs: {result.epochs}")
print(f"Stop reason: {result.stop_reason}")

# Evaluate with some prompts
responses = tuna.evaluate_prompts(
    [
        "What is machine learning?",
        "Explain fine-tuning in simple terms.",
    ]
)

for i, response in enumerate(responses):
    print(f"Response {i + 1}: {response}")
