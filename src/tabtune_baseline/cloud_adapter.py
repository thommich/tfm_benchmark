"""Small SDK v2 boundary; all GPU job submissions use ScriptProcessor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cloud_catalog import _snapshot
from .models import CloudOutputLayout, SageMakerConfig


def _sdk():
    try:
        from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
        from sagemaker.session import Session
    except ImportError as exc:
        raise RuntimeError('Install the optional cloud dependencies: pip install -e ".[cloud]"') from exc
    return ProcessingInput, ProcessingOutput, ScriptProcessor, Session


@dataclass
class SageMakerAdapter:
    cloud: SageMakerConfig
    boto_session: Any
    sdk_session: Any = None

    def __post_init__(self) -> None:
        self.sm = self.boto_session.client("sagemaker")

    def submit(self, layout: CloudOutputLayout, *, model: str, number: int,
               job_name: str, instance_type: str) -> str:
        Input, Output, Processor, Session = _sdk()
        session = self.sdk_session or Session(boto_session=self.boto_session, default_bucket=layout.run_uri[5:].split("/", 1)[0])
        storage = next(row["storage"] for row in _snapshot()["products"] if row["type"] == instance_type)
        processor = Processor(
            role=self.cloud.role_arn, image_uri=self.cloud.image_uri, command=["python"],
            instance_count=1, instance_type=instance_type,
            volume_size_in_gb=self.cloud.volume_size_gb,
            output_kms_key=self.cloud.kms_key_id,
            volume_kms_key=self.cloud.kms_key_id if storage == "EBS only" else None,
            max_runtime_in_seconds=self.cloud.max_runtime_seconds,
            sagemaker_session=session)
        inputs = [
            Input(source=layout.dataset_uri, destination="/opt/ml/processing/input/data", input_name="dataset"),
            Input(source=layout.plan_uri, destination="/opt/ml/processing/input/plan", input_name="plan"),
            Input(source=layout.key(f"attempts/{model}/{number}/attempt.json"),
                  destination="/opt/ml/processing/input/attempt", input_name="attempt"),
        ]
        output = Output(source="/opt/ml/processing/output",
                        destination=layout.attempt_output_uri(model, number), output_name="artifacts")
        processor.run(
            code=layout.key("staged/processing_entry.py"), inputs=inputs, outputs=[output],
            arguments=["--data", "/opt/ml/processing/input/data/input.parquet",
                       "--plan", "/opt/ml/processing/input/plan/plan.json",
                       "--attempt", "/opt/ml/processing/input/attempt/attempt.json",
                       "--output", "/opt/ml/processing/output"],
            job_name=job_name, wait=False, logs=False)
        return job_name

    def describe(self, job_name: str) -> dict[str, Any]:
        return self.sm.describe_processing_job(ProcessingJobName=job_name)
