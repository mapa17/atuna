<div align="center">
  <img src="docs/logo.png" alt="Atuna Logo" width="200" />

  # 🐟 Atuna

  **A tasty layer on top of Unsloth for Large Language Model fine-tuning**
</div>

A fine-tuning assistant for large language models with built-in hyperparameter optimization.

## Features

- **Easy fine-tuning** with Unsloth integration for efficient training
- **Hyperparameter optimization** with Optuna for automated tuning
- **Built-in evaluation** and monitoring capabilities
- **TensorBoard integration** for training visualization

## Installation

### From source (development)

```bash
git clone https://github.com/mapa17/tuna.git
cd tuna
uv sync
uv pip install -e .
```

### Using the library

```bash
pip install tuna  # When published to PyPI
```

## Quick Start

### Basic Fine-tuning

```python
from tuna import Tuna, TunaConfig, TrainingConfig, model_registry

# Configure your model
model = model_registry["unsloth/Qwen3-4B-Instruct-2507"]
config = TunaConfig(
    model_cfg=model,
    dataset_path="./data/training_set.csv",
    max_seq_length=2048,
    precision=16,  # Use 16-bit precision
)

# Create trainer
tuna = Tuna(config=config)

# Configure training
training_config = TrainingConfig(
    num_train_epochs=3,
    batch_size=1,
    learning_rate=5e-5,
    eval_epochs=1.0,
    enable_early_stopping=True,
)

# Train the model
result = tuna.train(config=training_config)
print(f"Training completed: {result.stop_reason}")
```

### Hyperparameter Optimization

```python
from tuna import Tuna, TunaConfig, TrainingConfig, HyperparpamConfig, model_registry

# Setup configuration
model = model_registry["unsloth/Qwen3-4B-Instruct-2507"]
config = TunaConfig(
    model_cfg=model,
    dataset_path="./data/training_set.csv",
    max_seq_length=2048,
    precision=4,  # Use 4-bit quantization for efficiency
)

tuna = Tuna(config=config)

# Configure training
training_config = TrainingConfig(
    num_train_epochs=2,
    eval_epochs=0.25,
    batch_size=1,
    data_sample=0.3,  # Use subset for faster experimentation
)

# Configure hyperparameter search
hyper_config = HyperparpamConfig(
    n_trials=10,
    learning_rate=[1e-5, 5e-5, 7e-5, 1e-4],
    weight_decay=[0.001, 0.01, 0.1],
    peft_r=[16, 32],
    lora_alpha=[32, 50, 64],
    enable_slora=True,
)

# Run optimization
study = tuna.hyperparam_tune(
    study_name="MyOptimization",
    train_config=training_config,
    hyper_config=hyper_config,
)

print(f"Best parameters: {study.best_trial.params}")
```

### Model Evaluation

```python
# Evaluate trained model
responses = tuna.evaluate_prompts([
    "What is machine learning?",
    "Explain fine-tuning in simple terms.",
])

for response in responses:
    print(f"Response: {response}")
```

## CLI Usage

```bash
# Show version
tuna --version

# List available models
tuna --list-models
```

## Data Format

Your training data should be a CSV file with `request` and `response` columns:

```csv
request,response
"What is Python?","Python is a programming language..."
"How does machine learning work?","Machine learning works by..."
```

## Monitoring and Visualization

### Optuna Dashboard
```bash
# View hyperparameter optimization results
optuna-dashboard sqlite:///./atuna_workspace/optuna_studies.db
```

### TensorBoard
```bash
# View training metrics
tensorboard --logdir ./atuna_workspace/logs
```

## Supported Models

- `unsloth/Qwen3-4B-Instruct-2507`
- `unsloth/Qwen3-0.6B-GGUF`

More models can be easily added to the `model_registry`.

## Configuration Options

### TunaConfig
- `model_cfg`: Model configuration from registry
- `dataset_path`: Path to training CSV file
- `max_seq_length`: Maximum sequence length (default: 2048)
- `precision`: Training precision - 4, 8, or 16 bit (default: 16)
- `peft_r`: LoRA rank parameter (default: 32)
- `workspace`: Working directory (default: "./atuna_workspace")

### TrainingConfig
- `learning_rate`: Learning rate (default: 2e-5)
- `batch_size`: Batch size (default: 1)
- `num_train_epochs`: Number of training epochs (default: 1.0)
- `eval_epochs`: Evaluation frequency in epochs
- `enable_early_stopping`: Enable early stopping (default: True)
- `data_sample`: Fraction of data to use (default: 1.0)

### HyperparpamConfig
- `n_trials`: Number of optimization trials
- `learning_rate`: List of learning rates to try
- `peft_r`: List of LoRA rank values to try
- `lora_alpha`: List of LoRA alpha values to try
- `weight_decay`: List of weight decay values to try

## Examples

See the `examples/` directory for complete usage examples:

- `examples/basic_finetuning.py` - Basic fine-tuning workflow
- `examples/hyperparameter_search.py` - Hyperparameter optimization

## Requirements

- Python 3.12+
- CUDA-capable GPU
- See `pyproject.toml` for full dependency list

## Development

```bash
# Clone repository
git clone https://github.com/mapa17/tuna.git
cd tuna

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Build package
uv build
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
