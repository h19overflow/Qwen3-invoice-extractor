"""
Data preparation - Combines HuggingFace datasets into training format.

Dependencies: datasets
Role: Downloads and merges invoice datasets into ChatML train.jsonl format.

Data Sources:
    - mychen76/invoices-and-receipts_ocr_v1: OCR text + parsed JSON
    - shubh303/Invoice-to-Json: Question/answer format with JSON extraction
"""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .prompt_template import SYSTEM_PROMPT


class DatasetAdapter(ABC):
    """Interface for adapting different dataset formats to ChatML."""

    @abstractmethod
    def get_dataset_name(self) -> str:
        """HuggingFace dataset identifier."""
        pass

    @abstractmethod
    def extract_pair(self, row: dict) -> tuple[str | None, str | None]:
        """
        Extract (input_text, json_output) from a dataset row.

        Returns:
            Tuple of (input_text, json_output) or (None, None) if invalid.
        """
        pass


class MyChen76Adapter(DatasetAdapter):
    """
    Adapter for mychen76/invoices-and-receipts_ocr_v1 dataset.

    Structure:
        - raw_data: Contains OCR words in JSON format
        - parsed_data: Contains structured JSON output
    """

    def get_dataset_name(self) -> str:
        return "mychen76/invoices-and-receipts_ocr_v1"

    def extract_pair(self, row: dict) -> tuple[str | None, str | None]:
        raw_data = row.get("raw_data")
        parsed_data = row.get("parsed_data")

        if not raw_data or not parsed_data:
            return None, None

        try:
            # Extract OCR words from raw_data JSON
            raw_dict = json.loads(raw_data)
            ocr_words = raw_dict.get("ocr_words", "")

            # Parse the list string to get actual words
            if isinstance(ocr_words, str) and ocr_words.startswith("["):
                words_list = eval(ocr_words)  # Safe here - controlled data
                input_text = " ".join(words_list)
            else:
                input_text = str(ocr_words)

            # Extract JSON from parsed_data
            parsed_dict = json.loads(parsed_data)
            json_str = parsed_dict.get("json", "")

            # The json field contains a string with single quotes - fix it
            if isinstance(json_str, str):
                # Replace single quotes with double quotes for valid JSON
                json_str = json_str.replace("'", '"')
                # Validate it's proper JSON
                json.loads(json_str)
                return input_text, json_str

        except (json.JSONDecodeError, SyntaxError, TypeError):
            pass

        return None, None


class Shubh303Adapter(DatasetAdapter):
    """
    Adapter for shubh303/Invoice-to-Json dataset.

    Structure:
        - question: Contains extraction instruction
        - answer: Contains JSON output (sometimes with markdown fences)
    """

    def get_dataset_name(self) -> str:
        return "shubh303/Invoice-to-Json"

    def extract_pair(self, row: dict) -> tuple[str | None, str | None]:
        question = row.get("question")
        answer = row.get("answer")

        if not question or not answer:
            return None, None

        # Clean markdown fences from answer
        json_output = answer.strip()
        json_output = re.sub(r"^```json\s*", "", json_output)
        json_output = re.sub(r"\s*```$", "", json_output)

        # Use the question as input (it describes what to extract)
        return question.strip(), json_output.strip()


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
        self._adapters = adapters or [MyChen76Adapter(), Shubh303Adapter()]
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
        print(f"Loading {dataset_name}...")

        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
            return []

        examples = []
        skipped = 0

        for row in dataset:
            text, json_output = adapter.extract_pair(row)

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

        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
