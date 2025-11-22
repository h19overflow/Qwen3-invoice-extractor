"""
SageMaker Serverless Deployer.

Dependencies: sagemaker, boto3
Role: Deploys fine-tuned model to SageMaker serverless endpoint.
"""

from .config import DeploymentConfig


class SageMakerDeployer:
    """
    Deploys model to SageMaker serverless endpoint.

    Cost: ~$0.00006/sec, scales to zero when idle.
    """

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self._predictor = None

    def deploy(self) -> str:
        """
        Deploy model to serverless endpoint.

        Returns:
            Endpoint name for invoking the model.
        """
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel
        from sagemaker.serverless import ServerlessInferenceConfig

        role = sagemaker.get_execution_role()

        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=self.config.memory_size_mb,
            max_concurrency=self.config.max_concurrency,
        )

        model = HuggingFaceModel(
            model_data=self.config.model_s3_uri,
            role=role,
            transformers_version=self.config.transformers_version,
            pytorch_version=self.config.pytorch_version,
            py_version=self.config.python_version,
            env={
                "HF_MODEL_ID": self.config.hf_model_id,
                "HF_TASK": "text-generation",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": str(self.config.timeout_seconds),
                "SM_HP_MODEL_LOAD_IN_4BIT": "False",
            },
        )

        print("Deploying serverless endpoint... (This takes ~5 mins)")
        self._predictor = model.deploy(serverless_inference_config=serverless_config)

        endpoint_name = self._predictor.endpoint_name
        print(f"✅ Endpoint Deployed! Name: {endpoint_name}")

        return endpoint_name

    @property
    def endpoint_name(self) -> str | None:
        """Get deployed endpoint name."""
        return self._predictor.endpoint_name if self._predictor else None
