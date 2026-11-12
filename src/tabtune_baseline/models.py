"""Public configuration and result types. Plans are JSON serializable and editable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
