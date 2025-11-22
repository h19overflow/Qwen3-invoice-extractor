"""
Training configuration - Pydantic models for training parameters.

Dependencies: pydantic
Role: Defines all configurable training hyperparameters and model settings.
"""

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.0, description="Dropout probability")
    target_modules: list[str] = Field(
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        description="Modules to apply LoRA to"
    )


class TrainingConfig(BaseModel):
    """Full training configuration."""

    # Model settings
    model_name: str = Field(default="Qwen/Qwen3-0.6B-Instruct")
    max_seq_length: int = Field(default=2048)
    load_in_4bit: bool = Field(default=True, description="Train in 4-bit to save memory")

    # LoRA settings
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Training hyperparameters
    batch_size: int = Field(default=4)
    gradient_accumulation_steps: int = Field(default=2)
    warmup_steps: int = Field(default=5)
    max_steps: int = Field(default=60)
    learning_rate: float = Field(default=2e-4)

    # Output
    output_dir: str = Field(default="outputs")
    export_dir: str = Field(default="qwen3_invoice_model")

    # Data
    train_file: str = Field(default="data/train.jsonl")
