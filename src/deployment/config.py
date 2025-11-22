"""
Deployment configuration for SageMaker.

Dependencies: pydantic
Role: Defines serverless endpoint configuration.
"""

from pydantic import BaseModel, Field


class DeploymentConfig(BaseModel):
    """SageMaker serverless deployment configuration."""

    # S3 settings
    model_s3_uri: str = Field(..., description="S3 URI to model.tar.gz")

    # Serverless settings
    memory_size_mb: int = Field(
        default=3072,
        description="Memory allocation (3GB sufficient for 0.6B model)"
    )
    max_concurrency: int = Field(default=5, description="Max concurrent invocations")

    # HuggingFace container settings
    transformers_version: str = Field(default="4.37.0")
    pytorch_version: str = Field(default="2.1.0")
    python_version: str = Field(default="py310")

    # Model settings
    hf_model_id: str = Field(
        default="Qwen/Qwen3-0.6B-Instruct",
        description="Fallback model ID"
    )
    timeout_seconds: int = Field(default=3600)
