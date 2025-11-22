"""
PDF Parser - Hybrid text extraction with OCR fallback.

Dependencies: langchain, pypdf, pdf2image, pytesseract
Role: First stage of pipeline - converts PDF invoices to raw text.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class TextExtractor(ABC):
    """Interface for text extraction strategies."""

    @abstractmethod
    def extract(self, file_path: Path) -> str | None:
        """Extract text from a file."""
        pass


class NativeTextExtractor(TextExtractor):
    """Extracts text from native PDF documents using PyPDF."""

    def __init__(self) -> None:
        from langchain_community.document_loaders import PyPDFLoader
        self._loader_class = PyPDFLoader

    def extract(self, file_path: Path) -> str | None:
        try:
            loader = self._loader_class(str(file_path))
            pages = loader.load()
            return "\n".join([p.page_content for p in pages])
        except Exception as e:
            print(f"Native parse warning: {e}")
            return None


class OCRTextExtractor(TextExtractor):
    """Extracts text from scanned PDFs using Tesseract OCR."""

    def extract(self, file_path: Path) -> str | None:
        try:
            import pytesseract
            from pdf2image import convert_from_path

            images = convert_from_path(str(file_path))
            text_parts = [pytesseract.image_to_string(img) for img in images]
            return "".join(text_parts)
        except Exception as e:
            print(f"OCR failed: {e}")
            return None


class PDFParser:
    """
    Hybrid PDF parser with native extraction and OCR fallback.

    Tries native text extraction first (fastest), falls back to OCR
    if the PDF is a scanned image.
    """

    MIN_TEXT_LENGTH = 50  # Threshold for OCR fallback

    def __init__(
        self,
        native_extractor: TextExtractor | None = None,
        ocr_extractor: TextExtractor | None = None,
    ) -> None:
        self._native = native_extractor or NativeTextExtractor()
        self._ocr = ocr_extractor or OCRTextExtractor()

    def parse(self, file_path: str | Path) -> str | None:
        """
        Parse invoice PDF and extract text.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text or None if extraction fails.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {path}")
            return None

        # Try native extraction first
        text = self._native.extract(path)

        # Fallback to OCR if text is too short
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            print("⚠️ Scanned PDF detected. Switching to OCR...")
            text = self._ocr.extract(path)

        return text
