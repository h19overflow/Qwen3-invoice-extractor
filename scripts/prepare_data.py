"""
Script to download and prepare training data from HuggingFace Hub.

Usage:
    python -m scripts.prepare_data
    python -m scripts.prepare_data --output data/train.jsonl
    python -m scripts.prepare_data --inspect  # Preview datasets first
"""

import argparse
from pathlib import Path

from src.training import (
    HuggingFaceDatasetLoader,
    INVOICE_DATASETS,
    TrainingDataPreparer,
)


def inspect_datasets() -> None:
    """Preview available datasets before downloading."""
    loader = HuggingFaceDatasetLoader()

    print("Available invoice datasets:")
    print("-" * 40)

    for info in INVOICE_DATASETS:
        print(f"\n📦 {info.name}")
        print(f"   Description: {info.description}")
        print(f"   Text source: {info.text_source}")
        print(f"   JSON source: {info.json_source}")

        if loader.check_dataset_exists(info.name):
            print("   Status: ✅ Available")
        else:
            print("   Status: ❌ Not found")

    print("\n" + "=" * 40)
    print("To preview examples, run:")
    print("  python -m scripts.prepare_data --preview <dataset_name>")


def preview_dataset(dataset_name: str, num_examples: int = 3) -> None:
    """Show sample examples from a dataset."""
    loader = HuggingFaceDatasetLoader()
    loader.inspect(dataset_name, num_examples)


def prepare_training_data(output_path: Path) -> None:
    """Download datasets and create train.jsonl."""
    print("=" * 60)
    print("Phase 1: Preparing training data")
    print("=" * 60)

    preparer = TrainingDataPreparer()
    preparer.prepare(output_path)

    print("\n" + "=" * 60)
    print(f"Training data ready: {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare invoice training data from HuggingFace Hub"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Output path for training JSONL file (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="List available datasets and their info",
    )
    parser.add_argument(
        "--preview",
        type=str,
        metavar="DATASET",
        help="Preview examples from a specific dataset",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=3,
        help="Number of examples to show in preview (default: 3)",
    )

    args = parser.parse_args()

    if args.preview:
        preview_dataset(args.preview, args.num_examples)
    elif args.inspect:
        inspect_datasets()
    else:
        prepare_training_data(args.output)


if __name__ == "__main__":
    main()
