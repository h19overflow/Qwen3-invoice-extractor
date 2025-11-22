"""
Invoice Model Trainer - Fine-tuning with Unsloth.

Dependencies: unsloth, transformers, trl, datasets, torch
Role: Handles LoRA fine-tuning and model export for SageMaker.
"""

from pathlib import Path

from .config import TrainingConfig
from .prompt_template import format_from_messages


class InvoiceModelTrainer:
    """
    Fine-tunes Qwen3 for invoice extraction using LoRA.

    Expects train.jsonl with ChatML messages format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Exports merged model in FP16 for SageMaker deployment.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load base model and apply LoRA adapters."""
        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )

        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
        )

    def _prepare_dataset(self):
        """Load and format training dataset with ChatML messages."""
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=self.config.train_file, split="train")

        def formatting_func(examples):
            texts = [format_from_messages(msg) for msg in examples["messages"]]
            return {"text": texts}

        return dataset.map(formatting_func, batched=True)

    def train(self) -> None:
        """Run fine-tuning."""
        import torch
        from transformers import TrainingArguments
        from trl import SFTTrainer

        if self._model is None:
            self.load_model()

        dataset = self._prepare_dataset()

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

    def export(self) -> Path:
        """Export merged model in FP16 for SageMaker."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        export_path = Path(self.config.export_dir)
        print(f"Saving merged model to {export_path}...")

        self._model.save_pretrained_merged(
            str(export_path),
            self._tokenizer,
            save_method="merged_16bit"
        )

        print(f"\n✅ Export complete.")
        print(f"Run: tar -czvf model.tar.gz -C {export_path} .")

        return export_path
