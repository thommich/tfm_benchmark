"""S3 staging, lease ownership, reconciliation and notebook-side polling."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from .cloud_adapter import SageMakerAdapter
from .cloud_preflight import inspect_account
from .cloud_scheduler import (initial_state, orchestrator_step,
                              reconcile_submission_intents, restore_tokens)
from .cloud_store import S3RunStore
from .models import CloudBenchmarkResult, CloudOutputLayout, SageMakerPlan
from .planning import dataframe_fingerprint
from .runner import _validate_plan


def _session(region: str):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError('Install the optional cloud dependencies: pip install -e ".[cloud]"') from exc
    return boto3.Session(region_name=region)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_uri(plan: SageMakerPlan) -> str:
    return plan.cloud.staging_uri.rstrip("/") + "/" + plan.run_id + "/"


def _event(state: dict, store: S3RunStore, kind: str, **fields: Any) -> None:
    item = {"at": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    state.setdefault("events", []).append(item)
    print(json.dumps(item, default=str), flush=True)
    data = "".join(json.dumps(event, default=str) + "\n" for event in state["events"])
    store.put_bytes("events.jsonl", data.encode())


def _preflight_models(plan: SageMakerPlan, state: dict, account, emit) -> None:
    for cloud_model in plan.models:
        record = state["models"][cloud_model.name]
        if record["status"] != "pending":
            continue
        if not cloud_model.candidates:
            record["status"] = "blocked"
            record["reason"] = "; ".join(cloud_model.unavailable_reasons)
            emit("model_blocked", model=cloud_model.name, reason=record["reason"])
        elif all(account.quotas.get(c.instance_type) == 0 for c in cloud_model.candidates):
            record["status"] = "blocked"
            record["reason"] = "All suitable Processing instance quotas are zero; request an increase in Service Quotas"
            emit("model_blocked", model=cloud_model.name, reason=record["reason"])


def _read_output(store: S3RunStore, session, name: str, number: int,
                 description: dict[str, Any]) -> dict[str, Any]:
    prefix = f"attempts/{name}/{number}"
    payload = store.maybe_bytes(f"{prefix}/result.json") or store.maybe_bytes(f"{prefix}/error.json")
    result = json.loads(payload) if payload else {"status": "failed", "error": "Processing output is unavailable"}
    if result.get("status") == "completed":
        return result
    try:
        logs = session.client("logs")
        streams = logs.describe_log_streams(
            logGroupName="/aws/sagemaker/ProcessingJobs",
            logStreamNamePrefix=description["ProcessingJobName"], limit=5).get("logStreams", [])
        lines: list[str] = []
        for stream in streams:
            events = logs.get_log_events(logGroupName="/aws/sagemaker/ProcessingJobs",
                                          logStreamName=stream["logStreamName"],
                                          startFromHead=False, limit=200).get("events", [])
            lines.extend(event["message"] for event in events)
        result["cloudwatch_logs"] = "\n".join(lines)[-20000:]
    except Exception as exc:
        result["log_access_error"] = str(exc)
    return result


def _drive(plan: SageMakerPlan, state: dict, store: S3RunStore,
           session, output_dir: Path) -> CloudBenchmarkResult:
    account = inspect_account(plan.cloud, session=session)
    adapter = SageMakerAdapter(plan.cloud, session)
    layout = CloudOutputLayout(store.run_uri)
    persist = lambda value: store.put_json("state.json", value)
    emit = lambda kind, **fields: _event(state, store, kind, **fields)
    reconcile_submission_intents(state, adapter, persist, emit)
    tokens = restore_tokens(plan, state, account)
    _preflight_models(plan, state, account, emit)
    persist(state)

    def stage_attempt(name, number, model_plan):
        store.put_json(f"attempts/{name}/{number}/attempt.json",
                       {"plan_identity": plan.identity, "dataset_sha256": state["dataset_sha256"],
                        "model": name, "number": number, "model_plan": model_plan})

    class Submitter:
        def submit(self, name, number, job, instance_type):
            return adapter.submit(layout, model=name, number=number,
                                  job_name=job, instance_type=instance_type)

        def describe(self, job):
            return adapter.describe(job)

    submitter = Submitter()
    last_status = 0.0
    while any(record["status"] in {"pending", "running"} for record in state["models"].values()):
        store.renew_lease()
        orchestrator_step(plan, state, tokens, submitter, stage_attempt,
                          lambda name, number, description: _read_output(store, session, name, number, description),
                          persist, emit)
        now = time.monotonic()
        if now - last_status >= 60:
            counts = {status: sum(record["status"] == status for record in state["models"].values())
                      for status in ("pending", "running", "completed", "failed", "blocked")}
            emit("status", **counts)
            last_status = now
        if any(record["status"] in {"pending", "running"} for record in state["models"].values()):
            time.sleep(plan.cloud.poll_seconds)
    state["status"] = "terminal"
    persist(state)
    from .cloud_artifacts import finalize_run
    return finalize_run(plan, state, store, output_dir)


def launch_run(df: pd.DataFrame, plan: SageMakerPlan, output_dir: Path) -> CloudBenchmarkResult:
    plan.validate_identity()
    _validate_plan(df, plan.baseline)
    session = _session(plan.cloud.region)
    # Account failure is reported before uploading the input snapshot.
    inspect_account(plan.cloud, session=session)
    store = S3RunStore(session.client("s3"), _run_uri(plan), kms_key_id=plan.cloud.kms_key_id)
    if not store.acquire_lease():
        state = store.get_json("state.json")
        return CloudBenchmarkResult(store.run_uri, "monitoring", None,
                                    {name: r["status"] for name, r in state["models"].items()})
    try:
        if store.maybe_bytes("state.json") is not None:
            raise ValueError(f"Run {plan.run_id} already exists in S3; use resume_sagemaker_baseline")
        with tempfile.TemporaryDirectory(prefix="tabtune-cloud-") as temporary:
            staged = Path(temporary) / "input.parquet"
            df.reset_index(drop=True).to_parquet(staged, index=False)
            if dataframe_fingerprint(pd.read_parquet(staged)) != plan.baseline.fingerprint:
                raise ValueError("Parquet roundtrip changes the dataframe fingerprint; normalize dtypes before planning")
            checksum = _sha256(staged)
            state = initial_state(plan, checksum)
            store.upload_file("staged/input.parquet", staged)
            store.put_json("plan.json", plan.to_dict())
            entry = files("tabtune_baseline").joinpath("processing_entry.py").read_bytes()
            store.put_bytes("staged/processing_entry.py", entry)
            store.put_json("state.json", state)
        return _drive(plan, state, store, session, output_dir)
    finally:
        store.release_lease()


def resume_run(run_uri: str, output_dir: Path) -> CloudBenchmarkResult:
    session = _session("sa-east-1")
    store = S3RunStore(session.client("s3"), run_uri)
    state = store.get_json("state.json")
    if not store.acquire_lease():
        return CloudBenchmarkResult(store.run_uri, "monitoring", None,
                                    {name: r["status"] for name, r in state["models"].items()})
    try:
        plan = SageMakerPlan.from_dict(store.get_json("plan.json"))
        if state["run_id"] != plan.run_id or state["plan_identity"] != plan.identity:
            raise ValueError("S3 run state and reviewed plan identity disagree")
        if state["status"] == "terminal":
            from .cloud_artifacts import finalize_run
            return finalize_run(plan, state, store, output_dir)
        return _drive(plan, state, store, session, output_dir)
    finally:
        store.release_lease()
