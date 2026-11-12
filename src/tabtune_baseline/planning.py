"""Validation, profiling, and inspectable model-specific run planning."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_selection import mutual_info_classif

from .models import BenchmarkConfig, ModelPlan, RunPlan


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Hash values, column names, dtypes and row order without copying the frame."""
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    except TypeError as exc:
        raise ValueError("Dataframe contains values that cannot be fingerprinted") from exc
    digest = hashlib.sha256()
    digest.update(repr([(str(c), str(t)) for c, t in df.dtypes.items()]).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate_config(df: pd.DataFrame, config: BenchmarkConfig) -> tuple[list[str], Any, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Provide a nonempty pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("Dataframe column names must be unique")
    required = [config.id_col, config.target_col]
    if config.split_col:
        required.append(config.split_col)
        if config.time_col:
            required.append(config.time_col)
    elif config.time_col:
        required.append(config.time_col)
    else:
        raise ValueError("Provide split_col or time_col; an automatic random split is not allowed")
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[config.id_col].isna().any() or df[config.id_col].duplicated().any():
        raise ValueError("ID values must be non-null and unique per row")
    if df[config.target_col].isna().any():
        raise ValueError("Target values must be non-null")
    labels = list(pd.unique(df[config.target_col]))
    if len(labels) != 2:
        raise ValueError(f"Binary classification needs exactly two target labels; found {len(labels)}")
    positive = config.positive_label
    if positive is None:
        if any(type(_native(label)) is bool and _native(label) is True for label in labels):
            positive = True
        elif any(type(_native(label)) is int and label == 1 for label in labels):
            positive = 1
        else:
            raise ValueError("Set positive_label; automatic inference is limited to 1/True")
    matches = [label for label in labels if label == positive]
    if len(matches) != 1:
        raise ValueError(f"positive_label={positive!r} does not identify exactly one target label")
    positive = _native(matches[0])
    negative = _native(next(label for label in labels if label != positive))
    excluded = {config.id_col, config.target_col, config.split_col, config.time_col}
    features = list(config.feature_cols) if config.feature_cols is not None else [c for c in df if c not in excluded]
    if not features or len(features) != len(set(features)):
        raise ValueError("feature_cols must contain unique feature names and at least one feature")
    if any(f not in df or f in excluded for f in features):
        raise ValueError("feature_cols contains a missing or reserved column")
    if config.prediction_batch_rows < 1:
        raise ValueError("prediction_batch_rows must be positive")
    if config.per_model_timeout_seconds is not None and config.per_model_timeout_seconds <= 0:
        raise ValueError("per_model_timeout_seconds must be positive")
    return features, positive, negative


def split_masks(df: pd.DataFrame, config: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    """Return disjoint reference/validation masks and a temporal cutoff if used."""
    if config.split_col:
        values = df[config.split_col]
        allowed = {config.train_value, config.validation_value}
        if config.train_value == config.validation_value or values.isna().any() or not values.isin(allowed).all():
            raise ValueError(f"split_col must contain only {config.train_value!r} and {config.validation_value!r}")
        train = values.eq(config.train_value).to_numpy()
        valid = values.eq(config.validation_value).to_numpy()
        source, cutoff = "explicit", None
    else:
        try:
            dates = pd.to_datetime(df[config.time_col], errors="raise", utc=True)
        except Exception as exc:
            raise ValueError("time_col contains invalid timestamps") from exc
        if dates.isna().any():
            raise ValueError("time_col contains missing timestamps")
        latest = dates.max()
        if dates.min() > latest - pd.DateOffset(months=6):
            raise ValueError("time_col must span at least six calendar months")
        boundary = latest - pd.DateOffset(months=3)
        train = dates.lt(boundary).to_numpy()
        valid = dates.ge(boundary).to_numpy()
        source, cutoff = "time", boundary.isoformat()
    if not train.any() or not valid.any():
        raise ValueError("The chosen split must have both reference and validation rows")
    for name, mask in (("reference", train), ("validation", valid)):
        if df.loc[mask, config.target_col].nunique() != 2:
            raise ValueError(f"The {name} split must contain both target classes")
    return train, valid, source, cutoff


def _profile(df: pd.DataFrame, features: list[str], train: np.ndarray, valid: np.ndarray, output_dir: str | None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk_path = Path(output_dir).resolve() if output_dir else Path.cwd()
    while not disk_path.exists():
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    gpu: dict[str, Any] = {"available": False}
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu = {"available": True, "name": props.name, "vram_bytes": props.total_memory}
        except Exception as exc:
            gpu["probe_error"] = str(exc)
    dtype_counts: dict[str, int] = {}
    for dtype in df[features].dtypes:
        key = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "datetime" if pd.api.types.is_datetime64_any_dtype(dtype) else "categorical_or_text"
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    return {
        "rows": len(df), "reference_rows": int(train.sum()), "validation_rows": int(valid.sum()),
        "features": len(features), "dtypes": dtype_counts,
        "missing_cells": int(df[features].isna().sum().sum()),
        "dataframe_bytes": int(df.memory_usage(index=True, deep=True).sum()),
        "cpu_count": psutil.cpu_count(logical=True), "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available, "scratch_free_bytes": disk.free, "gpu": gpu,
    }


def _feature_scores(X: pd.DataFrame, y: pd.Series, seed: int) -> dict[str, float]:
    """Univariate supervised MI on all reference rows, one column at a time."""
    y_binary = pd.factorize(y)[0]
    scores: dict[str, float] = {}
    for col in X:
        s = X[col]
        numeric = pd.api.types.is_numeric_dtype(s.dtype)
        if numeric:
            data = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill = data.median()
            values = data.fillna(0 if pd.isna(fill) else fill).to_numpy(dtype=np.float64)
        else:
            values = pd.factorize(s.astype("string").fillna("<missing>"))[0].astype(np.float64)
        if np.unique(values).size < 2:
            scores[col] = 0.0
            continue
        try:
            scores[col] = float(mutual_info_classif(values.reshape(-1, 1), y_binary, discrete_features=not numeric, random_state=seed)[0])
        except ValueError:
            scores[col] = 0.0
    return scores


def _plan_model(spec: Any, df: pd.DataFrame, train: np.ndarray, valid: np.ndarray,
                features: list[str], profile: dict[str, Any], config: BenchmarkConfig,
                score_cache: dict[str, float] | None) -> ModelPlan:
    item = ModelPlan(name=spec.name, status="ready", license=spec.license.name,
                     license_url=spec.license.url,
                     attribution_required=spec.license.requires_attribution,
                     weights=spec.weights, feature_cols=list(features),
                     context_rows=int(train.sum()), validation_rows=int(valid.sum()),
                     prediction_batch_rows=config.prediction_batch_rows)
    if spec.name.upper() == "XRFM":
        item.status, item.reason = "excluded", "Trains from scratch even in TabTune inference mode"
        return item
    if spec.license.commercial_use_ok is not True:
        item.status, item.reason = "excluded", "Weight license is not explicitly commercial-safe"
        return item
    if "inference" not in spec.classification_strategies:
        item.status, item.reason = "excluded", "No binary-classification inference strategy"
        return item
    env = spec.envelope
    if env.max_classes is not None and env.max_classes < 2:
        item.status, item.reason = "excluded", "Checkpoint cannot represent both classes"
        return item
    if env.max_features is not None and len(features) > env.max_features:
        if score_cache is None:
            raise RuntimeError("Feature scores were not prepared")
        order = sorted(features, key=lambda col: (-score_cache[col], features.index(col)))
        selected = set(order[:env.max_features])
        item.feature_cols = [col for col in features if col in selected]
        item.adaptations.append(f"Supervised train-only feature selection: {len(features)} -> {len(item.feature_cols)}")
    caps = [int(train.sum())]
    if env.max_rows:
        caps.append(env.max_rows)
    if env.max_cells:
        caps.append(env.max_cells // len(item.feature_cols))
    # Conservative in-memory reference budget; offloading models can use more.
    row_bytes = max(1, profile["dataframe_bytes"] // max(1, profile["rows"]))
    if spec.name != "TabICLv2":
        caps.append(max(2, int(profile["ram_available_bytes"] * 0.65 // (row_bytes * 4))))
    item.context_rows = min(caps)
    if item.context_rows < 2:
        item.status, item.reason = "excluded", "No valid two-class context fits the declared envelope"
        return item
    if item.context_rows < int(train.sum()):
        item.adaptations.append(f"Class-aware reference context: {int(train.sum())} -> {item.context_rows}; validation unchanged")
    gpu = profile["gpu"]
    device = "cuda" if gpu.get("available") and gpu.get("vram_bytes", 0) >= 2 * 1024**3 else "cpu"
    item.tuning_params = {"device": device}
    if spec.name == "TabICLv2":
        limited_memory = device == "cpu" or gpu.get("vram_bytes", 0) < 8 * 1024**3
        item.model_params = {"n_estimators": 4 if limited_memory else 8,
                             "batch_size": 4 if limited_memory else 8}
        if item.context_rows > 100_000:
            item.model_params.update({"kv_cache": True, "offload_mode": "auto"})
            item.adaptations.append("TabICLv2 KV cache offloading enabled")
    from tabtune.registry import check_envelope
    try:
        check_envelope(spec, n_rows=item.context_rows, n_features=len(item.feature_cols), n_classes=2, mode="error")
    except Exception as exc:
        item.status, item.reason = "excluded", str(exc)
    return item


def plan_baseline(df: pd.DataFrame, config: BenchmarkConfig, *, output_dir: str | None = None) -> RunPlan:
    """Create a dry-run plan. No model weights are loaded or downloaded."""
    features, positive, negative = _validate_config(df, config)
    train, valid, source, cutoff = split_masks(df, config)
    profile = _profile(df, features, train, valid, output_dir)
    from tabtune.registry import list_models
    specs = list_models()
    needs_selection = any(
        s.license.commercial_use_ok is True and s.name.upper() != "XRFM"
        and s.envelope.max_features and len(features) > s.envelope.max_features
        for s in specs
    )
    scores = _feature_scores(df.loc[train, features], df.loc[train, config.target_col], config.seed) if needs_selection else None
    models = [_plan_model(s, df, train, valid, features, profile, config, scores) for s in specs]
    profile["target_counts"] = {str(k): int(v) for k, v in df[config.target_col].value_counts().items()}
    profile["reference_target_counts"] = {str(k): int(v) for k, v in df.loc[train, config.target_col].value_counts().items()}
    profile["validation_target_counts"] = {str(k): int(v) for k, v in df.loc[valid, config.target_col].value_counts().items()}
    profile["tabtune_version"] = importlib.metadata.version("tabtune")
    profile["pandas_version"] = pd.__version__
    return RunPlan(config=config, fingerprint=dataframe_fingerprint(df),
                   positive_label=positive, negative_label=negative,
                   feature_cols=features, split_source=source, cutoff=cutoff,
                   train_rows=int(train.sum()), validation_rows=int(valid.sum()),
                   profile=profile, models=models)
