from .config import ModelConfig

# Registry of supported models
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
