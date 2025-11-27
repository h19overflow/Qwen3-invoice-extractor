"""
Training configuration - Pydantic models for training parameters.

Dependencies: pydantic
Role: Defines all configurable training hyperparameters and model settings.
"""

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    # Increased rank for better expressiveness (32 vs 16)
    r: int = Field(default=32, description="LoRA rank")
    # Alpha = 2x rank is optimal for most cases
    lora_alpha: int = Field(default=64, description="LoRA alpha scaling")
    # Small dropout helps prevent overfitting
    lora_dropout: float = Field(default=0.05, description="Dropout probability")
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
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    max_seq_length: int = Field(default=2048)
    load_in_4bit: bool = Field(default=True, description="Train in 4-bit to save memory")

    # LoRA settings
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Training hyperparameters
    batch_size: int = Field(default=4)
    gradient_accumulation_steps: int = Field(default=4)  # Effective batch size = 16
    warmup_steps: int = Field(default=50)  # Longer warmup for stability
    max_steps: int = Field(default=500)  # Much longer training for 22K examples
    learning_rate: float = Field(default=1e-4)  # Lower LR with larger effective batch
    weight_decay: float = Field(default=0.01, description="Weight decay for regularization")

    # Output
    output_dir: str = Field(default="outputs")
    export_dir: str = Field(default="qwen3_invoice_model")

    # Data
    train_file: str = Field(default="data/train.jsonl")
    val_file: str | None = Field(default=None, description="Path to validation dataset")
    val_size: float = Field(default=0.1, description="Validation split ratio (0.0 to disable)")

    # Logging & Evaluation
    logging_steps: int = Field(default=10)
    eval_steps: int = Field(default=50)
    save_steps: int = Field(default=100)
