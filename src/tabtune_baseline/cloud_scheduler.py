"""Bounded Processing scheduler and durable S3-backed orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from .cloud_preflight import AccountPreflight
from .cloud_retry import adapt_for_retry, classify_failure
from .models import CloudBenchmarkResult, InstanceCandidate, ModelPlan, SageMakerPlan
from .cloud_store import _error_code


@dataclass
class CapacityTokens:
    """Local reservations are released at most once per recorded job name."""

    max_concurrent: int
    preflight: AccountPreflight
    running: dict[str, str] = field(default_factory=dict)
    released: set[str] = field(default_factory=set)

    def available(self, candidate: InstanceCandidate) -> bool:
        if len(self.running) >= self.max_concurrent:
            return False
        quota = self.preflight.quotas.get(candidate.instance_type)
        if quota is None:
            return True
        external = self.preflight.active_jobs.get(candidate.instance_type, 0)
        local = sum(kind == candidate.instance_type for kind in self.running.values())
        return quota - external - local > 0

    def reserve(self, job_name: str, candidate: InstanceCandidate) -> None:
        if job_name in self.running or job_name in self.released:
            raise ValueError(f"Job {job_name} already consumed a token")
        if not self.available(candidate):
            raise RuntimeError(f"No Processing capacity token for {candidate.instance_type}")
        self.running[job_name] = candidate.instance_type

    def release(self, job_name: str) -> bool:
        if job_name not in self.running:
            return False
        del self.running[job_name]
        self.released.add(job_name)
        return True


def next_candidate(candidates: list[InstanceCandidate], tokens: CapacityTokens,
                   rejected_types: set[str] | None = None) -> InstanceCandidate | None:
    rejected_types = rejected_types or set()
    return next((candidate for candidate in candidates
                 if candidate.instance_type not in rejected_types and tokens.available(candidate)), None)


def dispatch_queue(queue: list[str], choices: dict[str, list[InstanceCandidate]],
                   tokens: CapacityTokens, job_name_for, submit) -> list[str]:
    """Start cost-ordered queued models while global and per-type tokens permit."""
    started: list[str] = []
    for model in list(queue):
        candidate = next_candidate(choices[model], tokens)
        if candidate is None:
            continue
        name = job_name_for(model)
        tokens.reserve(name, candidate)
        try:
            submit(model, candidate, name)
        except Exception:
            tokens.release(name)
            raise
        queue.remove(model)
        started.append(name)
    return started


def job_name(plan: SageMakerPlan, model: str, number: int, instance_type: str) -> str:
    slug = re.sub("[^a-z0-9-]", "-", model.lower()).strip("-")[:22]
    type_code = hashlib.sha1(instance_type.encode()).hexdigest()[:6]
    return f"ttb-{plan.run_id[:12]}-{slug}-{number}-{type_code}"[:63]


def initial_state(plan: SageMakerPlan, checksum: str) -> dict[str, Any]:
    base = {m.name: m for m in plan.baseline.models}
    return {"run_id": plan.run_id, "plan_identity": plan.identity,
            "dataset_sha256": checksum, "status": "running", "submission_intents": {},
            "models": {name: {"status": "pending" if item.status == "ready" else item.status,
                              "reason": item.reason, "model_plan": asdict(item),
                              "attempts": [], "rejections": [], "result": None,
                              "preferred_vram": 0, "wait_since": None}
                       for name, item in base.items()}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def orchestrator_step(plan: SageMakerPlan, state: dict[str, Any], tokens: CapacityTokens,
                      adapter: Any, stage_attempt, read_output, persist, emit,
                      *, clock=None) -> None:
    """Reconcile terminal jobs, then fill free slots; persist intent before CreateProcessingJob."""
    clock = clock or datetime.now(timezone.utc)
    cloud_models = {m.name: m for m in plan.models}
    for name, record in state["models"].items():
        if record["status"] != "running":
            continue
        attempt = record["attempts"][-1]
        try:
            description = adapter.describe(attempt["job_name"])
        except Exception as exc:
            emit("describe_failed", model=name, job_name=attempt["job_name"], error=str(exc))
            continue
        status = description["ProcessingJobStatus"]
        if status not in {"Completed", "Failed", "Stopped"}:
            continue
        tokens.release(attempt["job_name"])
        attempt["status"] = status
        attempt["ended_utc"] = _now()
        attempt["runtime_seconds"] = max(0.0, (
            datetime.fromisoformat(attempt["ended_utc"]) -
            datetime.fromisoformat(attempt["started_utc"])).total_seconds())
        attempt["estimated_runtime_cost_usd"] = round(
            attempt["runtime_seconds"] / 60 * attempt["usd_per_minute"], 4)
        output = read_output(name, attempt["number"], description)
        if status == "Completed" and output.get("status") == "completed" and output.get("scored_rows") == plan.baseline.validation_rows:
            record["result"] = output
            record["status"] = "completed"
            attempt["checkpoint"] = output.get("checkpoint", {})
            emit("model_completed", model=name, job_name=attempt["job_name"], gini=output.get("gini"))
            persist(state)
            continue
        failure = classify_failure(worker=output if output.get("status") == "failed" else None,
                                   description=description, logs=output.get("cloudwatch_logs", ""))
        attempt["failure_class"] = failure.kind
        attempt["failure_reason"] = failure.reason
        emit("attempt_failed", model=name, job_name=attempt["job_name"], kind=failure.kind,
             phase=failure.phase, reason=failure.reason)
        current = next(c for c in cloud_models[name].candidates if c.instance_type == attempt["instance_type"])
        quota_candidates = [c for c in cloud_models[name].candidates
                            if tokens.preflight.quotas.get(c.instance_type) != 0]
        adapted = adapt_for_retry(ModelPlan(**record["model_plan"]), failure,
                                  len(record["attempts"]), current, quota_candidates,
                                  plan.cloud.max_started_attempts)
        if adapted:
            next_item, preferred, notice = adapted
            record["model_plan"] = asdict(next_item)
            record["preferred_vram"] = preferred.vram_per_gpu_gib
            record["status"] = "pending"
            emit("retry_queued", model=name, attempt=len(record["attempts"]) + 1,
                 preferred_type=preferred.instance_type)
            if notice:
                emit("context_reduction_notice", model=name, notice=notice)
        else:
            record["status"] = "failed"
            record["reason"] = failure.reason
            emit("model_failed", model=name, reason=failure.reason)
        persist(state)
    for name, record in state["models"].items():
        if record["status"] != "pending":
            continue
        if len(tokens.running) >= tokens.max_concurrent:
            break
        candidates = [c for c in cloud_models[name].candidates
                      if c.vram_per_gpu_gib >= record["preferred_vram"]]
        rejected = {r["instance_type"] for r in record["rejections"]
                    if r["kind"] == "quota" or r["number"] == len(record["attempts"]) + 1}
        number = len(record["attempts"]) + 1
        while True:
            candidate = next_candidate(candidates, tokens, rejected)
            if candidate is None:
                if not candidates or all(c.instance_type in rejected for c in candidates):
                    record["status"] = "blocked"
                    record["reason"] = "No quota/capacity for a suitable priced GPU type; request a Processing quota increase or retry later"
                    emit("model_blocked", model=name, reason=record["reason"])
                elif any((c.instance_type not in tokens.preflight.quotas or
                          tokens.preflight.quotas[c.instance_type] -
                          tokens.preflight.active_jobs.get(c.instance_type, 0) > 0)
                         for c in candidates if c.instance_type not in rejected):
                    # Suitable account quota exists; this run's own jobs hold it.
                    record["wait_since"] = None
                elif record["wait_since"] is None:
                    record["wait_since"] = clock.isoformat()
                elif (clock - datetime.fromisoformat(record["wait_since"])).total_seconds() >= plan.cloud.external_capacity_wait_seconds:
                    record["status"] = "blocked"
                    record["reason"] = "External quota/capacity wait exceeded 30 minutes"
                    emit("model_blocked", model=name, reason=record["reason"])
                persist(state)
                break
            job = job_name(plan, name, number, candidate.instance_type)
            tokens.reserve(job, candidate)
            intent = {"model": name, "number": number, "job_name": job,
                      "instance_type": candidate.instance_type,
                      "image_uri": plan.cloud.image_uri,
                      "usd_per_minute": candidate.usd_per_minute,
                      "price_source": candidate.price_source,
                      "price_at": candidate.price_at,
                      "context_rows": record["model_plan"]["context_rows"],
                      "prediction_batch_rows": record["model_plan"]["prediction_batch_rows"],
                      "status": "submitting", "started_utc": _now()}
            stage_attempt(name, number, record["model_plan"])
            state["submission_intents"][job] = intent
            persist(state)
            try:
                adapter.submit(name, number, job, candidate.instance_type)
            except Exception as exc:
                # A lost CreateProcessingJob response can hide a started job.
                # Reconcile its deterministic name before treating the call as rejected.
                try:
                    adapter.describe(job)
                except Exception as describe_exc:
                    if _error_code(describe_exc) not in {"ResourceNotFound", "ResourceNotFoundException"}:
                        raise RuntimeError(f"Submission outcome for {job} is ambiguous; resume will reconcile it") from describe_exc
                else:
                    emit("submission_reconciled", model=name, job_name=job,
                         original_error=str(exc))
                    intent["status"] = "submitted"
                    state["submission_intents"].pop(job, None)
                    record["attempts"].append(intent)
                    record["status"] = "running"
                    persist(state)
                    break
                tokens.release(job)
                failure = classify_failure(submission_error=str(exc))
                state["submission_intents"].pop(job, None)
                record["rejections"].append({"number": number, "instance_type": candidate.instance_type,
                                             "kind": failure.kind, "reason": failure.reason})
                emit("submission_rejected", model=name, instance_type=candidate.instance_type,
                     kind=failure.kind, reason=failure.reason)
                persist(state)
                if failure.kind in {"quota", "capacity"}:
                    rejected.add(candidate.instance_type)
                    continue
                record["status"] = "failed"
                record["reason"] = failure.reason
                persist(state)
                break
            intent["status"] = "submitted"
            state["submission_intents"].pop(job, None)
            record["attempts"].append(intent)
            record["status"] = "running"
            record["wait_since"] = None
            emit("submitted", model=name, attempt=number, job_name=job,
                 instance_type=candidate.instance_type, estimated_cost_usd=candidate.estimated_cost_usd)
            persist(state)
            break


def reconcile_submission_intents(state: dict[str, Any], adapter: Any, persist, emit) -> None:
    """Resolve recorded names before dispatching; a found job is never resubmitted."""
    for job, intent in list(state.get("submission_intents", {}).items()):
        try:
            adapter.describe(job)
        except Exception as exc:
            if _error_code(exc) not in {"ResourceNotFound", "ResourceNotFoundException"}:
                raise
            emit("submission_intent_not_found", model=intent["model"], job_name=job)
            state["submission_intents"].pop(job)
            persist(state)
            continue
        record = state["models"][intent["model"]]
        if not any(a["job_name"] == job for a in record["attempts"]):
            intent["status"] = "submitted"
            record["attempts"].append(intent)
        record["status"] = "running"
        state["submission_intents"].pop(job)
        emit("submission_reconciled", model=intent["model"], job_name=job)
        persist(state)


def restore_tokens(plan: SageMakerPlan, state: dict[str, Any], preflight: AccountPreflight) -> CapacityTokens:
    tokens = CapacityTokens(plan.cloud.max_concurrent_jobs, preflight)
    own_counts: dict[str, int] = {}
    for record in state["models"].values():
        if record["status"] == "running" and record["attempts"]:
            attempt = record["attempts"][-1]
            tokens.running[attempt["job_name"]] = attempt["instance_type"]
            kind = attempt["instance_type"]
            if attempt["job_name"] in preflight.active_job_names:
                own_counts[kind] = own_counts.get(kind, 0) + 1
    for kind, count in own_counts.items():
        preflight.active_jobs[kind] = max(0, preflight.active_jobs.get(kind, 0) - count)
    return tokens
