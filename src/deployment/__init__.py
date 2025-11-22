"""
Deployment module - SageMaker serverless deployment.

Handles model upload to S3 and endpoint creation.
"""

from .sagemaker_deployer import SageMakerDeployer
from .config import DeploymentConfig

__all__ = ["SageMakerDeployer", "DeploymentConfig"]
