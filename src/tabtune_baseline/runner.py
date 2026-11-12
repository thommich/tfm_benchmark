"""Sequential isolated execution, bounded resource fallbacks, and artifacts."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .models import BenchmarkResult, ModelPlan, RunPlan, POLICY_VERSION
from .planning import _validate_config, dataframe_fingerprint, split_masks
from .report import render_report


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, default=str, indent=2, allow_nan=False), encoding="utf-8")


def _event(path: Path, event: str, **fields: Any) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def _validate_plan(df: pd.DataFrame, plan: RunPlan) -> None:
    if plan.policy_version != POLICY_VERSION:
        raise ValueError("Plan policy version differs from this package")
    if importlib.metadata.version("tabtune") != plan.profile.get("tabtune_version"):
        raise ValueError("TabTune version changed after planning; create a new RunPlan")
    if dataframe_fingerprint(df) != plan.fingerprint:
        raise ValueError("Dataframe changed after planning; create a new RunPlan")
    features, positive, negative = _validate_config(df, plan.config)
    if (features, positive, negative) != (plan.feature_cols, plan.positive_label, plan.negative_label):
        raise ValueError("Data schema or target mapping changed after planning")
    train, valid, source, cutoff = split_masks(df, plan.config)
    if (source, cutoff, int(train.sum()), int(valid.sum())) != (
        plan.split_source, plan.cutoff, plan.train_rows, plan.validation_rows
    ):
        raise ValueError("Split changed after planning")
    if len({m.name for m in plan.models}) != len(plan.models):
        raise ValueError("Plan contains duplicate model entries")
    from tabtune.registry import check_envelope, get_model_spec, list_models
    registered = {s.name for s in list_models()}
    if {m.name for m in plan.models} != registered:
        raise ValueError("Plan must retain every registered model for the report")
    for item in plan.models:
        if item.status != "ready":
            continue
        spec = get_model_spec(item.name)
        if item.name.upper() == "XRFM" or spec.license.commercial_use_ok is not True:
            raise ValueError(f"{item.name} is not eligible for this zero-shot commercial-safe run")
        if "inference" not in spec.classification_strategies:
            raise ValueError(f"{item.name} has no classification inference strategy")
        if not item.feature_cols or not set(item.feature_cols).issubset(plan.feature_cols):
            raise ValueError(f"{item.name} has invalid feature columns")
        if item.context_rows > plan.train_rows or item.context_rows < 2 or item.prediction_batch_rows < 1:
            raise ValueError(f"{item.name} has invalid context or batch size")
        check_envelope(spec, n_rows=item.context_rows, n_features=len(item.feature_cols), n_classes=2, mode="error")
        forbidden = {"epochs", "learning_rate", "peft_config", "save_checkpoint_path"}
        if forbidden.intersection(item.tuning_params):
            raise ValueError(f"{item.name} includes training-only parameters")


def _oom(result: dict[str, Any]) -> bool:
    message = (result.get("error", "") + " " + result.get("error_type", "")).lower()
    return "out of memory" in message or "memoryerror" in message or "cuda error: out of memory" in message


def _fallback(item: ModelPlan, step: int) -> bool:
    if step == 0 and item.prediction_batch_rows > 1:
        item.prediction_batch_rows = max(1, item.prediction_batch_rows // 2)
        if item.name == "TabICLv2":
            item.model_params["batch_size"] = max(1, item.model_params.get("batch_size", 8) // 2)
        item.adaptations.append(f"OOM retry: prediction batch -> {item.prediction_batch_rows}")
        return True
    if step <= 1 and item.name == "TabICLv2":
        item.model_params["kv_cache"] = True
        item.model_params["offload_mode"] = "auto"
        item.model_params["n_estimators"] = max(1, item.model_params.get("n_estimators", 8) // 2)
        item.adaptations.append("OOM retry: KV offloading and smaller ensemble")
        return True
    if item.context_rows > 2:
        item.context_rows = max(2, item.context_rows // 2)
        item.adaptations.append(f"OOM retry: reference context -> {item.context_rows}")
        return True
    return False


def run_baseline(df: pd.DataFrame, plan: RunPlan, output_dir: str | Path) -> BenchmarkResult:
    """Execute a reviewed plan and write a self-contained local run directory."""
    _validate_plan(df, plan)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir()
    scores_dir = run_dir / "scores"
    scores_dir.mkdir()
    events = run_dir / "events.jsonl"
    results: list[dict[str, Any]] = []
    _event(events, "run_started", run_id=run_id, fingerprint=plan.fingerprint)
    with tempfile.TemporaryDirectory(prefix="staging-", dir=run_dir) as staging:
        stage = Path(staging)
        data_file = stage / "input.parquet"
        # Reset the index so parquet round-trips row positions exactly.
        df.reset_index(drop=True).to_parquet(data_file, index=False)
        for original in plan.models:
            if original.status != "ready":
                results.append({"name": original.name, "status": original.status, "reason": original.reason,
                                "license": original.license, "license_url": original.license_url,
                                "attribution_required": original.attribution_required})
                _event(events, "model_excluded", model=original.name, reason=original.reason)
                continue
            item = copy.deepcopy(original)
            attempts: list[dict[str, Any]] = []
            result: dict[str, Any] = {}
            for retry in range(4):
                attempt_plan = copy.deepcopy(plan)
                attempt_plan.models = [item]
                plan_file = stage / f"{item.name}-{retry}-plan.json"
                result_file = stage / f"{item.name}-{retry}-result.json"
                scores_file = scores_dir / f"{item.name}.parquet"
                _json(plan_file, attempt_plan.to_dict())
                _event(events, "model_started", model=item.name, attempt=retry + 1,
                       context_rows=item.context_rows, feature_count=len(item.feature_cols),
                       prediction_batch_rows=item.prediction_batch_rows)
                cmd = [sys.executable, "-m", "tabtune_baseline.worker", "--data", str(data_file),
                       "--plan", str(plan_file), "--model", item.name, "--scores", str(scores_file),
                       "--result", str(result_file)]
                try:
                    worker_log = run_dir / f"{item.name}-attempt{retry+1}.log"
                    with worker_log.open("w", encoding="utf-8") as stream:
                        completed = subprocess.run(cmd, text=True, stdout=stream, stderr=subprocess.STDOUT,
                                                   timeout=plan.config.per_model_timeout_seconds, check=False)
                    result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {
                        "name": item.name, "status": "failed", "error": f"Worker exited {completed.returncode} without a result"}
                    if worker_log.stat().st_size:
                        result["worker_log"] = str(worker_log)
                except subprocess.TimeoutExpired:
                    result = {"name": item.name, "status": "failed", "error": "Per-model timeout exceeded"}
                except Exception as exc:
                    result = {"name": item.name, "status": "failed", "error": f"Worker launch failed: {exc}"}
                attempts.append({"attempt": retry + 1, "status": result["status"],
                                 "error": result.get("error"), "context_rows": item.context_rows,
                                 "prediction_batch_rows": item.prediction_batch_rows,
                                 "model_params": dict(item.model_params)})
                _event(events, "model_attempt_finished", model=item.name, attempt=retry + 1,
                       status=result["status"], error=result.get("error"))
                if result["status"] == "completed" or not _oom(result) or retry == 3:
                    break
                if not _fallback(item, retry):
                    break
            result["attempts"] = attempts
            result["license"] = original.license
            result["license_url"] = original.license_url
            result["attribution_required"] = original.attribution_required
            result["adaptations"] = list(dict.fromkeys(list(item.adaptations) + result.get("adaptations", [])))
            results.append(result)
    leaderboard = sorted((r for r in results if r["status"] == "completed"), key=lambda r: r["gini"], reverse=True)
    output = {"run_id": run_id, "metric": "gini=2*roc_auc-1", "models": results,
              "leaderboard": [{"name": r["name"], "gini": r["gini"], "auc": r["auc"]} for r in leaderboard]}
    results_file = run_dir / "results.json"
    manifest_file = run_dir / "manifest.json"
    report_file = run_dir / "report.html"
    _json(results_file, output)
    manifest = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "plan": plan.to_dict(), "resolved_versions": {
                    name: importlib.metadata.version(name) for name in ("tabtune", "pandas", "numpy", "scikit-learn")
                }, "python": sys.version, "model_attempts": {r["name"]: r.get("attempts", []) for r in results}}
    _json(manifest_file, manifest)
    report_file.write_text(render_report(plan, results, run_id), encoding="utf-8")
    _event(events, "run_finished", run_id=run_id, completed=len(leaderboard), total=len(results))
    return BenchmarkResult(run_dir=str(run_dir), report_html=str(report_file),
                           results_json=str(results_file), manifest_json=str(manifest_file),
                           scores_dir=str(scores_dir), log_jsonl=str(events), models=results)
