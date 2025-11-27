"""
Phase 2: Model Training - Fine-tune Qwen3 with LoRA using Unsloth.

Dependencies: unsloth, transformers, trl, datasets, torch
Role: Loads base model, applies LoRA adapters, and runs SFT training.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..configs import TrainingConfig
from ..utils import format_from_messages

load_dotenv()

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
        from huggingface_hub import login as hf_login
        from unsloth import FastLanguageModel

        print(f"Loading model: {self.config.model_name}...")

        # Check if HuggingFace token is needed (for gated models)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            print("Authenticating with HuggingFace...")
            hf_login(token=hf_token)

        try:
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
            )
        except Exception as e:
            error_msg = str(e)
            if "No config file found" in error_msg or "does not exist" in error_msg.lower():
                print(f"\n❌ Error: Model '{self.config.model_name}' not found on HuggingFace.")
                print("\nPossible solutions:")
                print("1. Check if the model name is correct")
                print("2. If it's a gated model, set HF_TOKEN environment variable:")
                print("   export HF_TOKEN=your_token_here")
                print("3. Try alternative model names:")
                print("   - Qwen/Qwen2.5-0.5B-Instruct")
                print("   - Qwen/Qwen2-0.5B-Instruct")
                print("   - microsoft/Phi-3-mini-4k-instruct")
                raise RuntimeError(f"Model not found: {self.config.model_name}") from e
            raise

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
        """Load and format training (and validation) datasets."""
        from datasets import load_dataset

        # Convert Path to string if needed
        train_file = (
            str(self.config.train_file)
            if isinstance(self.config.train_file, Path)
            else self.config.train_file
        )

        data_files = {"train": train_file}
        
        # Check for validation file
        if self.config.val_file:
            val_path = Path(self.config.val_file)
            if val_path.exists():
                print(f"Found validation file: {val_path}")
                data_files["test"] = str(val_path)
            else:
                print(f"⚠️ Validation file specified but not found: {val_path}")

        print(f"Loading datasets from {data_files}...")
        dataset = load_dataset("json", data_files=data_files)

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

        datasets = self._prepare_dataset()
        train_dataset = datasets["train"]
        eval_dataset = datasets.get("test")
        
        print(f"Training on {len(train_dataset)} examples...")
        if eval_dataset:
            print(f"Validating on {len(eval_dataset)} examples...")

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=TrainingArguments(
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=self.config.logging_steps,
                output_dir=self.config.output_dir,
                optim="adamw_8bit",
                # Learning rate schedule
                lr_scheduler_type="cosine",
                # Evaluation settings
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=self.config.eval_steps if eval_dataset else None,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
            ),
        )

        trainer.train()
        print("✅ Phase 2 complete: Training finished")
