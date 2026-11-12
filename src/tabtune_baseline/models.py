"""Public configuration and result types. Plans are JSON serializable and editable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any
from uuid import uuid4


POLICY_VERSION = "1"


@dataclass
class BenchmarkConfig:
    id_col: str
    target_col: str
    split_col: str | None = None
    time_col: str | None = None
    feature_cols: list[str] | None = None
    positive_label: str | int | float | bool | None = None
    train_value: str = "train"
    validation_value: str = "validation"
    seed: int = 42
    prediction_batch_rows: int = 1024
    per_model_timeout_seconds: float | None = None


@dataclass
class ModelPlan:
    name: str
    status: str
    reason: str | None = None
    license: str | None = None
    license_url: str | None = None
    attribution_required: bool = False
    weights: str | None = None
    feature_cols: list[str] = field(default_factory=list)
    context_rows: int = 0
    validation_rows: int = 0
    model_params: dict[str, Any] = field(default_factory=dict)
    tuning_params: dict[str, Any] = field(default_factory=dict)
    processor_params: dict[str, Any] = field(default_factory=dict)
    prediction_batch_rows: int = 1024
    adaptations: list[str] = field(default_factory=list)


@dataclass
class RunPlan:
    config: BenchmarkConfig
    fingerprint: str
    positive_label: str | int | float | bool
    negative_label: str | int | float | bool
    feature_cols: list[str]
    split_source: str
    cutoff: str | None
    train_rows: int
    validation_rows: int
    profile: dict[str, Any]
    models: list[ModelPlan]
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunPlan":
        copy = dict(data)
        copy["config"] = BenchmarkConfig(**copy["config"])
        copy["models"] = [ModelPlan(**model) for model in copy["models"]]
        return cls(**copy)


@dataclass
class BenchmarkResult:
    run_dir: str
    report_html: str
    results_json: str
    manifest_json: str
    scores_dir: str
    log_jsonl: str
    models: list[dict[str, Any]]

    def leaderboard(self) -> list[dict[str, Any]]:
        return sorted(
            (m for m in self.models if m["status"] == "completed"),
            key=lambda m: m["gini"],
            reverse=True,
        )


@dataclass(frozen=True)
class SageMakerConfig:
    """Reviewed cloud settings; no AWS action occurs while constructing this value."""

    role_arn: str
    image_uri: str
    staging_uri: str
    region: str = "sa-east-1"
    max_concurrent_jobs: int = 3
    max_started_attempts: int = 5
    max_runtime_seconds: int = 6 * 3600
    external_capacity_wait_seconds: int = 30 * 60
    poll_seconds: int = 60
    volume_size_gb: int = 100
    kms_key_id: str | None = None

    def __post_init__(self) -> None:
        if self.region != "sa-east-1":
            raise ValueError("The verified catalog is limited to sa-east-1")
        if not re.fullmatch(r"arn:aws:iam::\d{12}:role/.+", self.role_arn):
            raise ValueError("role_arn must be an IAM role ARN")
        if not re.fullmatch(r"\d{12}\.dkr\.ecr\.sa-east-1\.amazonaws\.com/.+@sha256:[0-9a-f]{64}", self.image_uri):
            raise ValueError("image_uri must be a sa-east-1 ECR image pinned by SHA-256 digest")
        if not re.fullmatch(r"s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[^?#]+/?", self.staging_uri):
            raise ValueError("staging_uri must be a private S3 prefix, not a bucket root")
        if (self.max_concurrent_jobs < 1 or self.max_started_attempts < 1
                or self.max_started_attempts > 5 or self.max_runtime_seconds < 60
                or self.external_capacity_wait_seconds < 0 or self.poll_seconds < 1
                or self.poll_seconds > 60
                or self.volume_size_gb < 30):
            raise ValueError("Invalid SageMaker scheduler or storage limit")


@dataclass
class InstanceCandidate:
    instance_type: str
    vcpus: int
    ram_gib: float
    gpu_count: int
    vram_per_gpu_gib: float
    scratch_gib: float
    usd_per_minute: float
    price_source: str
    price_at: str
    estimated_minutes: float
    estimated_cost_usd: float
    quota: int | None = None
    active_jobs: int | None = None


@dataclass
class CloudModelPlan:
    name: str
    estimate: dict[str, Any]
    candidates: list[InstanceCandidate] = field(default_factory=list)
    unavailable_reasons: list[str] = field(default_factory=list)


@dataclass
class SageMakerPlan:
    run_id: str
    baseline: RunPlan
    cloud: SageMakerConfig
    models: list[CloudModelPlan]
    created_utc: str
    identity: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["identity"] = self.compute_identity()
        return data

    def compute_identity(self) -> str:
        data = asdict(self)
        data.pop("identity", None)
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate_identity(self) -> None:
        if self.identity != self.compute_identity():
            raise ValueError("Cloud plan identity changed; create a new SageMakerPlan")
        if {m.name for m in self.models} != {m.name for m in self.baseline.models}:
            raise ValueError("Cloud plan must retain every registry model")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SageMakerPlan":
        copy = dict(data)
        copy["baseline"] = RunPlan.from_dict(copy["baseline"])
        copy["cloud"] = SageMakerConfig(**copy["cloud"])
        copy["models"] = [CloudModelPlan(**{**m, "candidates": [InstanceCandidate(**c) for c in m["candidates"]]})
                          for m in copy["models"]]
        plan = cls(**copy)
        plan.validate_identity()
        return plan


@dataclass
class ProcessingAttempt:
    model: str
    number: int
    job_name: str
    instance_type: str
    image_uri: str
    usd_per_minute: float
    price_source: str
    context_rows: int
    prediction_batch_rows: int
    status: str = "submitted"
    started_utc: str | None = None
    ended_utc: str | None = None
    failure_class: str | None = None
    failure_reason: str | None = None
    checkpoint: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CloudOutputLayout:
    """S3 keys under one private run prefix."""

    run_uri: str

    def key(self, suffix: str) -> str:
        return self.run_uri.rstrip("/") + "/" + suffix.lstrip("/")

    @property
    def state_uri(self) -> str:
        return self.key("state.json")

    @property
    def dataset_uri(self) -> str:
        return self.key("staged/input.parquet")

    @property
    def plan_uri(self) -> str:
        return self.key("plan.json")

    def attempt_output_uri(self, model: str, number: int) -> str:
        return self.key(f"attempts/{model}/{number}/")


@dataclass
class CloudBenchmarkResult:
    run_uri: str
    status: str
    local_result: BenchmarkResult | None
    model_statuses: dict[str, str]

    def leaderboard(self) -> list[dict[str, Any]]:
        return self.local_result.leaderboard() if self.local_result else []


def new_cloud_run_id() -> str:
    return uuid4().hex
