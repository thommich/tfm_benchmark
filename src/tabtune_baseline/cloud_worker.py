"""One GPU Processing job, one TabTune model, no accepted partial result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import ModelPlan, SageMakerPlan
from .planning import dataframe_fingerprint
from .runner import _validate_plan
from .worker import execute_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(data: Path, plan_file: Path, attempt_file: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    phase = "identity"

    def emit(event: str, **fields):
        nonlocal phase
        if event == "phase":
            phase = fields["phase"]
        record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps(record, default=str), flush=True)

    try:
        cloud = SageMakerPlan.from_dict(json.loads(plan_file.read_text(encoding="utf-8")))
        attempt = json.loads(attempt_file.read_text(encoding="utf-8"))
        if attempt["plan_identity"] != cloud.identity or attempt["dataset_sha256"] != _sha256(data):
            raise ValueError("Staged plan identity or dataset checksum differs from the reviewed snapshot")
        original = next(m for m in cloud.baseline.models if m.name == attempt["model"])
        if original.status != "ready":
            raise ValueError("Requested model is not eligible")
        item = ModelPlan(**attempt["model_plan"])
        if (item.name != original.name or item.feature_cols != original.feature_cols or
                item.status != original.status or item.license != original.license or
                item.license_url != original.license_url or item.weights != original.weights or
                item.attribution_required != original.attribution_required or
                item.validation_rows != original.validation_rows or
                item.context_rows > original.context_rows or
                item.prediction_batch_rows > original.prediction_batch_rows or
                item.tuning_params != original.tuning_params or
                item.processor_params != original.processor_params or
                not 1 <= attempt["number"] <= cloud.cloud.max_started_attempts or
                set(item.model_params).difference(set(original.model_params) |
                                                  {"batch_size", "n_estimators", "kv_cache", "offload_mode"}) or
                (item.context_rows < original.context_rows and attempt["number"] < 4)):
            raise ValueError("Attempt settings exceed the reviewed model envelope or reduce context too early")
        plan = copy.deepcopy(cloud.baseline)
        plan.models = [item if m.name == item.name else m for m in plan.models]
        emit("phase", phase="snapshot_validation", model=item.name, attempt=attempt["number"])
        frame = pd.read_parquet(data)
        if dataframe_fingerprint(frame) != cloud.baseline.fingerprint:
            raise ValueError("Staged dataframe fingerprint differs from the reviewed plan")
        _validate_plan(frame, plan)
        emit("phase", phase="gpu_probe")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("No CUDA GPU visible inside the Processing job")
        props = torch.cuda.get_device_properties(0)
        emit("gpu", name=props.name, vram_bytes=props.total_memory,
             torch=torch.__version__, cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version())
        cache = Path("/opt/ml/processing/checkpoints")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache / "huggingface")
        os.environ["TORCH_HOME"] = str(cache / "torch")
        partial = output / "scores.partial.parquet"
        result = execute_model(frame, plan, item, partial,
                               on_phase=lambda value: emit("phase", phase=value))
        if result["scored_rows"] != plan.validation_rows:
            raise ValueError("Worker did not score every validation row")
        final_scores = output / "scores.parquet"
        partial.replace(final_scores)
        result["scores_file"] = "scores.parquet"
        result["attempt"] = attempt["number"]
        result["image_uri"] = cloud.cloud.image_uri
        result["runtime"] = {"python": sys.version,
                             "torch": torch.__version__, "cuda": torch.version.cuda,
                             "cudnn": torch.backends.cudnn.version(),
                             **{name: importlib.metadata.version(name)
                                for name in ("tabtune", "pandas", "numpy", "scikit-learn")}}
        (output / "result.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        emit("completed", model=item.name, scored_rows=result["scored_rows"])
        return result
    except Exception as exc:
        (output / "scores.partial.parquet").unlink(missing_ok=True)
        failure = {"status": "failed", "phase": phase, "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc()}
        (output / "error.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        emit("failed", **failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_one(Path(args.data), Path(args.plan), Path(args.attempt), Path(args.output))
    if result["status"] != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
