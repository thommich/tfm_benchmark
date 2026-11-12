"""Evidence-based Processing failure classes and bounded model adaptations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .models import InstanceCandidate, ModelPlan


@dataclass(frozen=True)
class Failure:
    kind: str
    phase: str | None
    reason: str


def classify_failure(*, worker: dict[str, Any] | None = None,
                     description: dict[str, Any] | None = None,
                     logs: str = "", submission_error: str = "") -> Failure:
    worker = worker or {}
    description = description or {}
    structured = " ".join(str(x) for x in (
        worker.get("error_type", ""), worker.get("error", ""),
        description.get("FailureReason", ""), description.get("ExitMessage", ""),
        submission_error) if x)
    # Startup logs often mention checkpoints and memory even when the eventual
    # exception is unrelated. Use them only if no structured error exists.
    worker_error = bool(worker.get("error_type") or worker.get("error"))
    tail_line = next((line.strip() for line in reversed(logs.splitlines()) if line.strip()), "")
    diagnostic = "" if worker_error else tail_line[-500:]
    reason = " ".join(part for part in (structured, diagnostic) if part)
    lowered = reason.lower()
    phase = worker.get("phase")
    if any(marker in lowered for marker in ("resourcelimitexceeded", "quota exceeded", "service quota",
                                             "quota has been reached", "limit exceeded for processing")):
        return Failure("quota", phase, reason.strip())
    if any(marker in lowered for marker in ("insufficientinstancecapacity", "insufficient capacity",
                                             "capacity unavailable")):
        return Failure("capacity", phase, reason.strip())
    if any(marker in lowered for marker in ("cuda out of memory", "cuda error: out of memory",
                                             "torch.outofmemoryerror", "memoryerror", "out of memory")):
        return Failure("oom", phase, reason.strip())
    if "maxruntimeinseconds" in lowered or "runtime limit" in lowered or "timed out" in lowered:
        return Failure("timeout", phase, reason.strip())
    if any(marker in lowered for marker in ("accessdenied", "not authorized", "permission denied")):
        return Failure("iam", phase, reason.strip())
    if any(marker in lowered for marker in ("checkpoint", "huggingface", "model weights", "gatedrepoerror")):
        return Failure("checkpoint", phase, reason.strip())
    if any(marker in lowered for marker in ("fingerprint", "schema", "invalid data", "plan identity")):
        return Failure("invalid_input", phase, reason.strip())
    return Failure("unknown", phase, reason.strip() or "No worker output or SageMaker failure reason")


def adapt_for_retry(item: ModelPlan, failure: Failure, started_attempts: int,
                    current: InstanceCandidate, candidates: list[InstanceCandidate],
                    max_started_attempts: int = 5
                    ) -> tuple[ModelPlan, InstanceCandidate, str | None] | None:
    """Return settings for the next started attempt, never exceeding attempt five."""
    if failure.kind != "oom" or started_attempts >= max_started_attempts:
        return None
    updated = copy.deepcopy(item)
    notice = None
    next_type = current
    if failure.phase == "predict" and started_attempts == 1 and updated.prediction_batch_rows > 1:
        updated.prediction_batch_rows = max(1, updated.prediction_batch_rows // 2)
        if updated.name == "TabICLv2":
            updated.model_params["batch_size"] = max(1, updated.model_params.get("batch_size", 8) // 2)
        updated.adaptations.append(f"Prediction OOM: batch -> {updated.prediction_batch_rows}")
    else:
        larger = [c for c in candidates if c.vram_per_gpu_gib > current.vram_per_gpu_gib]
        if larger:
            next_type = min(larger, key=lambda c: (c.usd_per_minute, c.instance_type))
            updated.adaptations.append(f"OOM: per-GPU VRAM -> {next_type.vram_per_gpu_gib} GiB")
        elif updated.name == "TabICLv2" and not updated.model_params.get("kv_cache"):
            updated.model_params.update({"kv_cache": True, "offload_mode": "auto"})
            updated.adaptations.append("OOM: KV-cache offloading enabled")
    # Only attempt 4 or 5 may run with a reduced reference context.
    next_attempt = started_attempts + 1
    if next_attempt >= 4 and updated.context_rows > 2:
        old = updated.context_rows
        updated.context_rows = max(2, old // 2)
        notice = f"Attempt {next_attempt}: reference context reduced {old} -> {updated.context_rows}; validation unchanged"
        updated.adaptations.append(notice)
    if (updated.prediction_batch_rows == item.prediction_batch_rows and
            updated.context_rows == item.context_rows and updated.model_params == item.model_params and
            next_type.instance_type == current.instance_type):
        return None
    return updated, next_type, notice
