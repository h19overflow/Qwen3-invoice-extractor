"""
Invoice Extractor Client - SageMaker inference pipeline.

Dependencies: boto3, pydantic
Role: Orchestrates PDF ingestion, model invocation, and validation.
"""

import json
from pathlib import Path

from ..ingestion import PDFParser
from ..training.prompt_template import format_inference_prompt
from .schema import InvoiceSchema


class SageMakerInvoker:
    """Handles SageMaker endpoint invocation."""

    def __init__(self, endpoint_name: str) -> None:
        import boto3
        self._client = boto3.client("sagemaker-runtime")
        self._endpoint_name = endpoint_name

    def invoke(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Invoke SageMaker endpoint with prompt.

        Returns:
            Generated text from model.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }

        response = self._client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())
        return result[0]["generated_text"]


class ResponseValidator:
    """Validates and parses model output."""

    @staticmethod
    def clean_json(text: str) -> str:
        """Remove markdown fences if present."""
        return text.replace("```json", "").replace("```", "").strip()

    def validate(self, output_text: str) -> InvoiceSchema | None:
        """
        Parse and validate model output.

        Returns:
            Validated InvoiceSchema or None if validation fails.
        """
        try:
            clean_text = self.clean_json(output_text)
            data = json.loads(clean_text)
            invoice = InvoiceSchema(**data)
            print("✅ SUCCESS: Valid Data Extracted")
            return invoice
        except json.JSONDecodeError as e:
            print(f"❌ FAILURE: JSON Parse Error - {e}")
            return None
        except Exception as e:
            print(f"❌ FAILURE: Validation Error - {e}")
            return None


class InvoiceExtractorClient:
    """
    Full invoice extraction pipeline.

    Orchestrates: PDF parsing → prompt formatting → model invocation → validation.
    """

    def __init__(
        self,
        endpoint_name: str,
        parser: PDFParser | None = None,
        invoker: SageMakerInvoker | None = None,
        validator: ResponseValidator | None = None,
    ) -> None:
        self._parser = parser or PDFParser()
        self._invoker = invoker or SageMakerInvoker(endpoint_name)
        self._validator = validator or ResponseValidator()

    def process(self, pdf_path: str | Path) -> InvoiceSchema | None:
        """
        Extract invoice data from PDF.

        Args:
            pdf_path: Path to invoice PDF.

        Returns:
            Validated invoice data or None if extraction fails.
        """
        print(f"Reading {pdf_path}...")

        # 1. Parse PDF
        raw_text = self._parser.parse(pdf_path)
        if not raw_text:
            print("Failed to read PDF")
            return None

        # 2. Format prompt
        prompt = format_inference_prompt(raw_text)

        # 3. Invoke model
        print("Invoking SageMaker Endpoint...")
        output = self._invoker.invoke(prompt)

        # 4. Validate response
        return self._validator.validate(output)
