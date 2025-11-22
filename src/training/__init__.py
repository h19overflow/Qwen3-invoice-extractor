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
from .configs import INVOICE_DATASETS, DatasetInfo, LoRAConfig, TrainingConfig

# Phases
from .phases import (
    InvoiceModelTrainer,
    ModelExporter,
    TrainingDataPreparer,
)

# Utils
from .utils import (
    SYSTEM_PROMPT,
    DataValidator,
    HuggingFaceDatasetLoader,
    format_from_messages,
    format_inference_prompt,
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
