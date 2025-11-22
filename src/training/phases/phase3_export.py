"""
Phase 3: Model Export - Merge LoRA adapters and save for deployment.

Role: Merges LoRA weights into base model and exports in FP16 for SageMaker.
"""

from pathlib import Path

from ..configs import TrainingConfig


class ModelExporter:
    """
    Phase 3: Export trained model for SageMaker deployment.

    Merges LoRA adapters into FP16 format (required for SageMaker Serverless CPU).
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def export(self, model, tokenizer) -> Path:
        """
        Execute Phase 3: Merge LoRA and export model.

        Args:
            model: The trained model with LoRA adapters.
            tokenizer: The model tokenizer.

        Returns:
            Path to exported model directory.
        """
        if model is None:
            raise RuntimeError("No model provided. Train the model first.")

        export_path = Path(self.config.export_dir)
        print(f"Merging LoRA adapters and saving to {export_path}...")

        model.save_pretrained_merged(
            str(export_path),
            tokenizer,
            save_method="merged_16bit"
        )

        print(f"\n✅ Phase 3 complete: Model exported to {export_path}")
        print(f"\nNext steps:")
        print(f"  1. Create tarball: tar -czvf model.tar.gz -C {export_path} .")
        print(f"  2. Upload to S3: aws s3 cp model.tar.gz s3://your-bucket/")
        print(f"  3. Deploy with SageMaker serverless endpoint")

        return export_path
