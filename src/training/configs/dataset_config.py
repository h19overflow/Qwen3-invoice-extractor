"""
Dataset configuration - Registry of supported HuggingFace datasets.

Role: Defines metadata for invoice datasets used in training.
"""

from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Metadata for a HuggingFace dataset."""

    name: str
    description: str
    text_source: str
    json_source: str


# Registry of supported datasets
INVOICE_DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="mychen76/invoices-and-receipts_ocr_v1",
        description="OCR text from invoices with structured JSON output",
        text_source="raw_data.ocr_words",
        json_source="parsed_data.json",
    ),
    DatasetInfo(
        name="shubh303/Invoice-to-Json",
        description="Question/answer pairs for JSON extraction from receipts",
        text_source="question",
        json_source="answer",
    ),
]
