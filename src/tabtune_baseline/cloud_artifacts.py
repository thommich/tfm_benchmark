"""Verify complete validation scores and publish a full-registry cloud report."""

from __future__ import annotations

import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .models import BenchmarkResult, CloudBenchmarkResult, SageMakerPlan
from .planning import split_masks
from .report import render_report


def _verify_scores(path: Path, df: pd.DataFrame, plan: SageMakerPlan,
                   result: dict) -> None:
    scores = pd.read_parquet(path)
    cfg = plan.baseline.config
    _, valid, _, _ = split_masks(df, cfg)
    if list(scores.columns) != [cfg.id_col, cfg.target_col, "score_positive"]:
        raise ValueError("Score artifact has the wrong columns")
    if len(scores) != plan.baseline.validation_rows or result.get("scored_rows") != len(scores):
        raise ValueError("Score artifact is missing validation rows")
    if (scores[cfg.id_col].tolist() != df.loc[valid, cfg.id_col].tolist() or
            scores[cfg.target_col].tolist() != df.loc[valid, cfg.target_col].tolist()):
        raise ValueError("Score artifact does not match validation IDs and targets in order")
    values = scores["score_positive"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Score artifact has invalid positive-class probabilities")
    labels = (df.loc[valid, cfg.target_col] == plan.baseline.positive_label).astype(int)
    auc = float(roc_auc_score(labels, values))
    if not math.isclose(auc, result.get("auc", float("nan")), abs_tol=1e-10):
        raise ValueError("Score artifact AUC differs from worker result")
    if not math.isclose(2 * auc - 1, result.get("gini", float("nan")), abs_tol=1e-10):
        raise ValueError("Score artifact Gini differs from worker result")


def _benchmark_result(run_dir: Path, models: list[dict]) -> BenchmarkResult:
    return BenchmarkResult(run_dir=str(run_dir), report_html=str(run_dir / "report.html"),
                           results_json=str(run_dir / "results.json"),
                           manifest_json=str(run_dir / "manifest.json"),
                           scores_dir=str(run_dir / "scores"),
                           log_jsonl=str(run_dir / "events.jsonl"), models=models)


def finalize_run(plan: SageMakerPlan, state: dict, store, output_dir: Path) -> CloudBenchmarkResult:
    run_dir = output_dir.resolve() / plan.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scores").mkdir(exist_ok=True)
    if state.get("artifacts_finalized"):
        for name in ("results.json", "manifest.json", "report.html", "events.jsonl"):
            store.download_file(name, run_dir / name)
        payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        for model in payload["models"]:
            if model["status"] == "completed":
                store.download_file(f"scores/{model['name']}.parquet",
                                    run_dir / "scores" / f"{model['name']}.parquet")
        if not state.get("staged_input_deleted"):
            store.delete("staged/input.parquet")
            state["staged_input_deleted"] = True
            store.put_json("state.json", state)
        result = _benchmark_result(run_dir, payload["models"])
        return CloudBenchmarkResult(store.run_uri, "completed", result,
                                    {name: item["status"] for name, item in state["models"].items()})

    staged = run_dir / "input.staged.parquet"
    store.download_file("staged/input.parquet", staged)
    try:
        df = pd.read_parquet(staged)
        from .planning import dataframe_fingerprint
        if dataframe_fingerprint(df) != plan.baseline.fingerprint:
            raise ValueError("Staged input no longer matches the reviewed dataframe")
        models: list[dict] = []
        for planned in plan.baseline.models:
            record = state["models"][planned.name]
            base = {"name": planned.name, "license": planned.license,
                    "license_url": planned.license_url,
                    "attribution_required": planned.attribution_required,
                    "attempts": record["attempts"], "submission_rejections": record["rejections"]}
            if record["status"] == "completed":
                result = dict(record["result"])
                attempt = record["attempts"][-1]
                path = run_dir / "scores" / f"{planned.name}.parquet"
                try:
                    store.download_file(f"attempts/{planned.name}/{attempt['number']}/scores.parquet", path)
                    _verify_scores(path, df, plan, result)
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    record["status"] = "failed"
                    record["reason"] = f"Final score verification failed: {exc}"
                    models.append({**base, "status": "failed", "reason": record["reason"]})
                    continue
                store.upload_file(f"scores/{planned.name}.parquet", path)
                result["scores_file"] = str(path)
                result["adaptations"] = list(dict.fromkeys(
                    record["model_plan"].get("adaptations", []) + result.get("adaptations", [])))
                models.append({**base, **result})
            else:
                models.append({**base, "status": record["status"], "reason": record["reason"]})
        leaderboard = sorted((m for m in models if m["status"] == "completed"),
                             key=lambda m: m["gini"], reverse=True)
        result_payload = {"run_id": plan.run_id, "run_uri": store.run_uri,
                          "metric": "gini=2*roc_auc-1", "models": models,
                          "leaderboard": [{"name": m["name"], "gini": m["gini"], "auc": m["auc"]}
                                          for m in leaderboard]}
        manifest = {"run_id": plan.run_id, "run_uri": store.run_uri,
                    "dataset_sha256": state["dataset_sha256"], "plan": plan.to_dict(),
                    "image_uri": plan.cloud.image_uri,
                    "model_attempts": {name: record["attempts"] for name, record in state["models"].items()},
                    "submission_rejections": {name: record["rejections"] for name, record in state["models"].items()},
                    "worker_runtime": {m["name"]: m.get("runtime") for m in models if m["status"] == "completed"},
                    "planner_versions": {name: importlib.metadata.version(name)
                                         for name in ("tabtune", "pandas", "numpy", "scikit-learn")},
                    "planner_python": sys.version}
        (run_dir / "results.json").write_text(json.dumps(result_payload, indent=2, default=str), encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        (run_dir / "report.html").write_text(render_report(plan.baseline, models, plan.run_id), encoding="utf-8")
        (run_dir / "events.jsonl").write_text("".join(json.dumps(event, default=str) + "\n"
                                                 for event in state.get("events", [])), encoding="utf-8")
        for name in ("results.json", "manifest.json", "report.html", "events.jsonl"):
            store.upload_file(name, run_dir / name)
        state["artifacts_finalized"] = True
        store.put_json("state.json", state)
        # This deletion is strictly after all final artifacts exist in Studio and S3.
        store.delete("staged/input.parquet")
        state["staged_input_deleted"] = True
        store.put_json("state.json", state)
        result = _benchmark_result(run_dir, models)
        return CloudBenchmarkResult(store.run_uri, "completed", result,
                                    {name: item["status"] for name, item in state["models"].items()})
    finally:
        staged.unlink(missing_ok=True)
