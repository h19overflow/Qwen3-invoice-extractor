"""
Ingestion module - PDF parsing and text extraction.

Handles native text extraction with OCR fallback for scanned documents.
"""

from .pdf_parser import PDFParser

__all__ = ["PDFParser"]
