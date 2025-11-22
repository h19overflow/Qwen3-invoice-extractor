"""
Deployment module - SageMaker serverless deployment.

Handles model upload to S3 and endpoint creation.
"""

from .config import DeploymentConfig
from .sagemaker_deployer import SageMakerDeployer

__all__ = ["SageMakerDeployer", "DeploymentConfig"]
