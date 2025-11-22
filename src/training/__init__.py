"""
Training module - Qwen3 fine-tuning pipeline for invoice extraction.

Structure:
    configs/  - Pydantic configuration models
    utils/    - Shared utilities (prompt formatting, validation, HF loader)
    phases/   - Training pipeline phases (data prep, training, export)

Usage:
    from src.training import TrainingDataPreparer, InvoiceModelTrainer, ModelExporter
    from src.training import TrainingConfig, HuggingFaceDatasetLoader
"""

# Configs
from .configs import TrainingConfig, LoRAConfig, DatasetInfo, INVOICE_DATASETS

# Utils
from .utils import (
    SYSTEM_PROMPT,
    format_from_messages,
    format_inference_prompt,
    DataValidator,
    HuggingFaceDatasetLoader,
)

# Phases
from .phases import (
    TrainingDataPreparer,
    InvoiceModelTrainer,
    ModelExporter,
)

__all__ = [
    # Configs
    "TrainingConfig",
    "LoRAConfig",
    "DatasetInfo",
    "INVOICE_DATASETS",
    # Utils
    "SYSTEM_PROMPT",
    "format_from_messages",
    "format_inference_prompt",
    "DataValidator",
    "HuggingFaceDatasetLoader",
    # Phases
    "TrainingDataPreparer",
    "InvoiceModelTrainer",
    "ModelExporter",
]
