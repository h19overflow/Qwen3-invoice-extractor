"""
Invoice validation schema - Pydantic models for output validation.

Dependencies: pydantic
Role: Ensures extracted data meets format requirements (99% accuracy protocol).
"""

import datetime

from pydantic import BaseModel, Field, field_validator


class InvoiceSchema(BaseModel):
    """
    Strict invoice data schema.

    Acts as the "gatekeeper" for validation - only valid data passes through.
    """

    invoice_number: str = Field(..., description="Unique ID of invoice")
    total_amount: float = Field(..., description="Final total including tax")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    vendor: str = Field(..., description="Vendor/company name")

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Ensure date is in YYYY-MM-DD format."""
        try:
            datetime.datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")

    @field_validator("total_amount")
    @classmethod
    def validate_positive_amount(cls, v: float) -> float:
        """Ensure amount is positive."""
        if v < 0:
            raise ValueError("Total amount must be positive")
        return v
