"""
Inference module - Invoice extraction pipeline client.

Handles SageMaker invocation and response validation.
"""

from .client import InvoiceExtractorClient
from .schema import InvoiceSchema

__all__ = ["InvoiceExtractorClient", "InvoiceSchema"]
