"""
Data validation utilities.

Role: Validates training examples before inclusion in the dataset.
"""

import json


# Expected schema fields - examples with these fields are higher quality
EXPECTED_HEADER_FIELDS = {"invoice_no", "invoice_date", "seller", "client"}
EXPECTED_ITEM_FIELDS = {"item_desc", "item_qty"}
EXPECTED_SUMMARY_FIELDS = {"total_net_worth", "total_gross_worth"}


class DataValidator:
    """Validates training examples with quality scoring."""

    MIN_TEXT_LENGTH = 50  # Increased minimum to filter out very short inputs
    MIN_JSON_LENGTH = 100  # JSON output should be substantial
    MAX_TEXT_LENGTH = 8000  # Avoid extremely long inputs that may be truncated

    @staticmethod
    def is_valid_json(text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def get_json_structure(json_text: str) -> dict | None:
        """Parse JSON and return the structure."""
        try:
            return json.loads(json_text)
        except (json.JSONDecodeError, TypeError):
            return None

    def calculate_quality_score(self, json_output: str) -> float:
        """
        Calculate quality score (0-1) based on schema compliance.
        Higher scores indicate better training examples.
        """
        data = self.get_json_structure(json_output)
        if not data:
            return 0.0

        score = 0.0
        max_score = 0.0

        # Check for header section
        max_score += 4
        if "header" in data and isinstance(data["header"], dict):
            score += 1  # Has header
            header = data["header"]
            for field in EXPECTED_HEADER_FIELDS:
                if field in header:
                    score += 0.75

        # Check for items section
        max_score += 3
        if "items" in data and isinstance(data["items"], list) and len(data["items"]) > 0:
            score += 1  # Has items
            first_item = data["items"][0]
            for field in EXPECTED_ITEM_FIELDS:
                if field in first_item:
                    score += 1

        # Check for summary section
        max_score += 3
        if "summary" in data and isinstance(data["summary"], dict):
            score += 1  # Has summary
            summary = data["summary"]
            for field in EXPECTED_SUMMARY_FIELDS:
                if field in summary:
                    score += 1

        return score / max_score if max_score > 0 else 0.0

    def is_valid_example(
        self,
        text: str | None,
        json_output: str | None,
        min_quality_score: float = 0.3
    ) -> bool:
        """
        Validate a training example with quality filtering.

        Filters:
            - Empty or too short/long text
            - Invalid JSON output
            - Low quality examples (poor schema compliance)
        """
        # Check input text
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return False
        if len(text.strip()) > self.MAX_TEXT_LENGTH:
            return False

        # Check JSON output
        if not json_output or len(json_output.strip()) < self.MIN_JSON_LENGTH:
            return False
        if not self.is_valid_json(json_output):
            return False

        # Check quality score
        quality = self.calculate_quality_score(json_output)
        if quality < min_quality_score:
            return False

        return True
