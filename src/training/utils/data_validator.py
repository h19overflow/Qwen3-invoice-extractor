"""
Data validation utilities.

Role: Validates training examples before inclusion in the dataset.
"""

import json


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
