"""One model per process: a failed checkpoint does not poison the next run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .compat import apply_tabtune_compat
from .models import ModelPlan, RunPlan
from .planning import split_masks


def select_context(X: pd.DataFrame, y: pd.Series, maximum: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Select a seeded, class-aware reference context without touching validation."""
    if len(X) <= maximum:
        return X, y
    if maximum < 2:
        raise ValueError("Context needs at least two rows")
    labels = pd.unique(y)
    if len(labels) != 2:
        raise ValueError("Context selection needs two classes")
    counts = [int((y == label).sum()) for label in labels]
    first = max(1, min(counts[0], maximum - 1, round(maximum * counts[0] / len(y))))
    second = maximum - first
    if second > counts[1]:
        second = counts[1]
        first = maximum - second
    if first > counts[0] or second < 1:
        raise ValueError("Cannot select a two-class context of the requested size")
    rng = np.random.default_rng(seed)
    positions = np.concatenate([
        rng.choice(np.flatnonzero((y == label).to_numpy()), size=n, replace=False)
        for label, n in zip(labels, (first, second))
    ])
    positions.sort()
    return X.iloc[positions], y.iloc[positions]


def _class_order(pipeline: Any) -> list[Any]:
    processor = getattr(pipeline, "processor", None)
    custom = getattr(processor, "custom_preprocessor_", None)
    for owner in (custom, processor, getattr(pipeline, "model", None), pipeline):
        encoder = getattr(owner, "label_encoder_", None)
        if encoder is not None and hasattr(encoder, "classes_"):
            return list(encoder.classes_)
        classes = getattr(owner, "classes_", None)
        if classes is not None:
            return list(classes)
    raise ValueError("Cannot verify the pipeline's probability class order")


def _default_pipeline_factory(**kwargs: Any) -> Any:
    from tabtune import TabularPipeline
    return TabularPipeline(**kwargs)


def _checkpoint_identity(pipeline: Any, model_plan: ModelPlan) -> dict[str, Any]:
    """Record the resolved local checkpoint when TabTune exposes its path."""
    identity: dict[str, Any] = {"declared_source": model_plan.weights, "resolution": "unavailable"}
    model = getattr(pipeline, "model", None)
    if model is None:
        return identity
    for attr in ("checkpoint_version", "model_version", "checkpoint_revision"):
        value = getattr(model, attr, None)
        if isinstance(value, (str, int, float)):
            identity[attr] = str(value)
    candidates = [getattr(model, attr, None) for attr in
                  ("checkpoint_path", "model_path_", "checkpoint_file", "model_path")]
    if model_plan.name == "TabPFN" and importlib.metadata.version("tabtune") == "0.2.1":
        try:
            from tabtune.models.tabpfn.model_loading import resolve_model_path
            candidates.insert(0, resolve_model_path(None, "classifier", version="v2.5")[0])
        except Exception:
            pass
    for candidate in candidates:
        if not isinstance(candidate, (str, Path)) or str(candidate) == "auto":
            continue
        path = Path(candidate)
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            identity.update({"resolution": "local_file", "path": str(path),
                             "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size})
            break
        if isinstance(candidate, str):
            identity["reported_reference"] = candidate
    return identity


def execute_model(
    df: pd.DataFrame,
    run_plan: RunPlan,
    model_plan: ModelPlan,
    scores_path: Path,
    *,
    pipeline_factory: Callable[..., Any] = _default_pipeline_factory,
) -> dict[str, Any]:
    """Run one model, preserving every validation row in the score artifact."""
    if model_plan.status != "ready":
        raise ValueError(f"Model {model_plan.name} is not ready")
    started = time.perf_counter()
    cfg = run_plan.config
    train, valid, _, _ = split_masks(df, cfg)
    X_ref = df.loc[train, model_plan.feature_cols]
    y_ref = (df.loc[train, cfg.target_col] == run_plan.positive_label).astype(int)
    X_val = df.loc[valid, model_plan.feature_cols]
    y_val = (df.loc[valid, cfg.target_col] == run_plan.positive_label).astype(int)
    X_ref, y_ref = select_context(X_ref, y_ref, model_plan.context_rows, cfg.seed)
    compatibility = apply_tabtune_compat(model_plan.name) if pipeline_factory is _default_pipeline_factory else None
    pipeline = pipeline_factory(
        model_name=model_plan.name,
        task_type="classification",
        tuning_strategy="inference",
        tuning_params=model_plan.tuning_params,
        processor_params=model_plan.processor_params,
        model_params=model_plan.model_params,
        envelope_mode="error",
        license_mode="commercial",
        validate=True,
    )
    pipeline.fit(X_ref, y_ref)
    checkpoint = _checkpoint_identity(pipeline, model_plan) if pipeline_factory is _default_pipeline_factory else {"resolution": "test_stub"}
    order = _class_order(pipeline)
    if len(order) != 2 or set(order) != {0, 1}:
        raise ValueError(f"Expected probability classes [0, 1], found {order!r}")
    positive_index = order.index(1)
    scores = np.empty(len(X_val), dtype=np.float64)
    batch = model_plan.prediction_batch_rows
    for start in range(0, len(X_val), batch):
        stop = min(start + batch, len(X_val))
        probabilities = np.asarray(pipeline.predict_proba(X_val.iloc[start:stop]))
        if probabilities.shape != (stop - start, 2):
            raise ValueError(f"Bad probability shape {probabilities.shape}; expected {(stop-start, 2)}")
        if not np.isfinite(probabilities).all() or (probabilities < -1e-6).any() or (probabilities > 1 + 1e-6).any():
            raise ValueError("Probabilities contain non-finite or out-of-range values")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-3):
            raise ValueError("Probability rows do not sum to one")
        scores[start:stop] = probabilities[:, positive_index]
    auc = float(roc_auc_score(y_val, scores))
    score_frame = pd.DataFrame({
        cfg.id_col: df.loc[valid, cfg.id_col].to_numpy(),
        cfg.target_col: df.loc[valid, cfg.target_col].to_numpy(),
        "score_positive": scores,
    })
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    score_frame.to_parquet(scores_path, index=False)
    return {
        "name": model_plan.name, "status": "completed", "auc": auc,
        "gini": 2 * auc - 1, "scored_rows": len(scores),
        "context_rows": len(X_ref), "feature_count": len(model_plan.feature_cols),
        "positive_label": run_plan.positive_label, "probability_order": order,
        "seconds": round(time.perf_counter() - started, 3),
        "scores_file": str(scores_path), "adaptations": list(model_plan.adaptations) + ([compatibility] if compatibility else []),
        "model_params": model_plan.model_params, "tuning_params": model_plan.tuning_params,
        "processor_params": model_plan.processor_params, "weights": model_plan.weights,
        "checkpoint": checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    run_plan = RunPlan.from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    item = next(m for m in run_plan.models if m.name == args.model)
    try:
        cfg = run_plan.config
        columns = list(dict.fromkeys(item.feature_cols + [cfg.id_col, cfg.target_col,
                                                       cfg.split_col or cfg.time_col]))
        frame = pd.read_parquet(args.data, columns=columns)
        result = execute_model(frame, run_plan, item, Path(args.scores))
    except Exception as exc:
        result = {"name": args.model, "status": "failed", "error": str(exc),
                  "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
    Path(args.result).write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
