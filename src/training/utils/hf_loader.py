"""
HuggingFace Hub dataset loader utility.

Dependencies: huggingface_hub, datasets
Role: Handles authentication, downloading, and inspection of HF datasets.
"""

from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login

from ..configs import DatasetInfo, INVOICE_DATASETS


class HuggingFaceDatasetLoader:
    """
    Loads datasets from HuggingFace Hub.

    Handles authentication, caching, and provides dataset inspection utilities.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize loader.

        Args:
            cache_dir: Optional custom cache directory for downloaded datasets.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._api = HfApi()

    def authenticate(self, token: str | None = None) -> None:
        """
        Authenticate with HuggingFace Hub.

        Args:
            token: HF token. If None, uses cached token or prompts for login.
        """
        if token:
            login(token=token)
        else:
            login()
        print("✅ Authenticated with HuggingFace Hub")

    def check_dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists on HuggingFace Hub."""
        try:
            self._api.dataset_info(dataset_name)
            return True
        except Exception:
            return False

    def load(self, dataset_name: str, split: str = "train") -> Dataset:
        """
        Load a dataset from HuggingFace Hub.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "user/dataset").
            split: Dataset split to load.

        Returns:
            Loaded dataset.

        Raises:
            ValueError: If dataset cannot be loaded.
        """
        print(f"Loading {dataset_name} (split={split})...")

        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
            )
            print(f"  ✅ Loaded {len(dataset)} examples")
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load {dataset_name}: {e}") from e

    def load_all_invoice_datasets(self) -> dict[str, Dataset]:
        """
        Load all registered invoice datasets.

        Returns:
            Dict mapping dataset name to loaded Dataset.
        """
        datasets = {}
        for info in INVOICE_DATASETS:
            try:
                datasets[info.name] = self.load(info.name)
            except ValueError as e:
                print(f"  ⚠️ Skipping {info.name}: {e}")
        return datasets

    def inspect(self, dataset_name: str, num_examples: int = 3) -> None:
        """
        Print dataset info and sample examples.

        Args:
            dataset_name: Dataset to inspect.
            num_examples: Number of examples to show.
        """
        dataset = self.load(dataset_name)

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        print(f"Number of examples: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        print(f"\nSample examples:")

        for i, example in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
            print(f"\n--- Example {i+1} ---")
            for key, value in example.items():
                if key == "image":
                    print(f"{key}: <image>")
                    continue
                preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                print(f"{key}: {preview}")

    @staticmethod
    def get_dataset_info(dataset_name: str) -> DatasetInfo | None:
        """Get metadata for a registered dataset."""
        for info in INVOICE_DATASETS:
            if info.name == dataset_name:
                return info
        return None

    @staticmethod
    def list_available_datasets() -> list[DatasetInfo]:
        """List all registered invoice datasets."""
        return INVOICE_DATASETS.copy()
