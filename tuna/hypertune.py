from .tuna import (
    TunaConfig,
    Tuna,
    model_registry,
    TrainingConfig,
    HyperparpamConfig,
)

# model = model_registry["unsloth/Qwen3-0.6B-GGUF"]
model = model_registry["unsloth/Qwen3-4B-Instruct-2507"]
exp_config = TunaConfig(
    model_cfg=model,
    dataset_path="./data/251015_training_set.csv",
    max_seq_length=2048,
    load_in_4bit=True,
    cache_dir="./hf_cache",
)

# exp = Tuna(config=exp_config)
# exp._load_model()

# run_config = TrainingConfig(
#     num_train_epochs=0.03,
#     eval_steps=1,
#     batch_size=1,
#     learning_rate=5e-5,
#     gradient_accumulation_steps=16,
# )
# exp.train(config=run_config)


hyper_exp = Tuna(config=exp_config)

run_config = TrainingConfig(
    num_train_epochs=2,
    eval_epochs=1,
    batch_size=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=16,
)

hyper_config = HyperparpamConfig(
    n_trials=3,
    learning_rate=[1e-5, 5e-5, 7e-5, 1e-4],
    peft_r=[16, 32],
    lora_alpha=[16, 32],
)
study = hyper_exp.hyperparam_tune(
    study_name="MyFineTuneTest",
    train_config=run_config,
    hyper_config=hyper_config,
)


# hyper_exp.evaluate(
#    prompts=[
#        "Was kannst du mir über den Stephans Dom in Wien erzählen?",
#        "Was kannst du mir über den Stephans Dom in Wien, auf wienerisch erzählen?",
#    ]
# )
