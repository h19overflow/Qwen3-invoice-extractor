"""
Data preparation - Combines HuggingFace datasets into training format.

Dependencies: datasets
Role: Downloads and merges invoice datasets into ChatML train.jsonl format.

Data Sources:
    - GokulRajaR/invoice-ocr-json: Synthetic invoices (strict JSON structure)
    - shubh303/Invoice-to-Json: Real/varied receipts (messy layouts, OCR noise)
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .prompt_template import SYSTEM_PROMPT


class DatasetAdapter(ABC):
    """Interface for adapting different dataset formats to ChatML."""

    @abstractmethod
    def get_text_column(self) -> str:
        """Column name containing invoice text."""
        pass

    @abstractmethod
    def get_json_column(self) -> str:
        """Column name containing target JSON."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """HuggingFace dataset identifier."""
        pass


class GokulRajaRAdapter(DatasetAdapter):
    """Adapter for GokulRajaR/invoice-ocr-json dataset."""

    def get_text_column(self) -> str:
        return "ocr_text"

    def get_json_column(self) -> str:
        return "ground_truth"

    def get_dataset_name(self) -> str:
        return "GokulRajaR/invoice-ocr-json"


class Shubh303Adapter(DatasetAdapter):
    """Adapter for shubh303/Invoice-to-Json dataset."""

    def get_text_column(self) -> str:
        return "text"

    def get_json_column(self) -> str:
        return "label"

    def get_dataset_name(self) -> str:
        return "shubh303/Invoice-to-Json"


class DataValidator:
    """Validates training examples before inclusion."""

    MIN_TEXT_LENGTH = 10

    @staticmethod
    def is_valid_json(text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def is_valid_example(self, text: str | None, json_output: str | None) -> bool:
        """
        Validate a training example.

        Filters:
            - Empty or too short text
            - Invalid JSON output
        """
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return False
        if not json_output or not self.is_valid_json(json_output):
            return False
        return True


class TrainingDataPreparer:
    """
    Prepares training data by combining multiple HuggingFace datasets.

    Outputs ChatML format:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
    """

    def __init__(
        self,
        adapters: list[DatasetAdapter] | None = None,
        validator: DataValidator | None = None,
    ) -> None:
        self._adapters = adapters or [GokulRajaRAdapter(), Shubh303Adapter()]
        self._validator = validator or DataValidator()

    def _to_chatml(self, text: str, json_output: str) -> dict:
        """Convert text/json pair to ChatML messages format."""
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text.strip()},
                {"role": "assistant", "content": json_output.strip()},
            ]
        }

    def _load_and_convert(self, adapter: DatasetAdapter) -> list[dict]:
        """Load dataset and convert to ChatML format."""
        from datasets import load_dataset

        dataset_name = adapter.get_dataset_name()
        text_col = adapter.get_text_column()
        json_col = adapter.get_json_column()

        print(f"Loading {dataset_name}...")

        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            return []

        examples = []
        skipped = 0

        for row in dataset:
            text = row.get(text_col)
            json_output = row.get(json_col)

            if self._validator.is_valid_example(text, json_output):
                examples.append(self._to_chatml(text, json_output))
            else:
                skipped += 1

        print(f"  Loaded {len(examples)} examples, skipped {skipped} invalid")
        return examples

    def prepare(self, output_path: str | Path = "train.jsonl") -> Path:
        """
        Download datasets, combine, and save as train.jsonl.

        Args:
            output_path: Path for output JSONL file.

        Returns:
            Path to created training file.
        """
        output_path = Path(output_path)
        all_examples = []

        for adapter in self._adapters:
            examples = self._load_and_convert(adapter)
            all_examples.extend(examples)

        print(f"\nTotal training examples: {len(all_examples)}")

        with open(output_path, "w", encoding="utf-8") as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"✅ Saved to {output_path}")
        return output_path
