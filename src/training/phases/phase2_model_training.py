"""
Phase 2: Model Training - Fine-tune Qwen3 with LoRA using Unsloth.

Dependencies: unsloth, transformers, trl, datasets, torch
Role: Loads base model, applies LoRA adapters, and runs SFT training.
"""

from ..configs import TrainingConfig
from ..utils import format_from_messages


class InvoiceModelTrainer:
    """
    Phase 2: Fine-tune Qwen3 for invoice extraction using LoRA.

    Expects train.jsonl with ChatML messages format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Access the loaded model."""
        return self._model

    @property
    def tokenizer(self):
        """Access the loaded tokenizer."""
        return self._tokenizer

    def load_model(self) -> None:
        """Load base model and apply LoRA adapters."""
        from unsloth import FastLanguageModel

        print(f"Loading model: {self.config.model_name}...")

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )

        print("Applying LoRA adapters...")
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
        )

        print("✅ Model loaded with LoRA adapters")

    def _prepare_dataset(self):
        """Load and format training dataset with ChatML messages."""
        from datasets import load_dataset

        print(f"Loading dataset from {self.config.train_file}...")
        dataset = load_dataset("json", data_files=self.config.train_file, split="train")

        def formatting_func(examples):
            texts = [format_from_messages(msg) for msg in examples["messages"]]
            return {"text": texts}

        return dataset.map(formatting_func, batched=True)

    def train(self) -> None:
        """Execute Phase 2: Run fine-tuning."""
        import torch
        from transformers import TrainingArguments
        from trl import SFTTrainer

        if self._model is None:
            self.load_model()

        dataset = self._prepare_dataset()
        print(f"Training on {len(dataset)} examples...")

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=TrainingArguments(
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir=self.config.output_dir,
                optim="adamw_8bit",
            ),
        )

        trainer.train()
        print("✅ Phase 2 complete: Training finished")
