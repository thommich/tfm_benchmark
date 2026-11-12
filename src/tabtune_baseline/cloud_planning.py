"""CPU-only planning against declared Processing resources, never Studio hardware."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .models import (BenchmarkConfig, CloudModelPlan, SageMakerConfig,
                     SageMakerPlan, new_cloud_run_id)
from .planning import _assemble_plan, _validate_config, split_masks
from .cloud_catalog import processing_catalog, suitable_candidates

GIB = 1024 ** 3


def _remote_profile(df: pd.DataFrame, features: list[str], train: Any, valid: Any) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    for dtype in df[features].dtypes:
        kind = ("numeric" if pd.api.types.is_numeric_dtype(dtype) else
                "datetime" if pd.api.types.is_datetime64_any_dtype(dtype) else "categorical_or_text")
        dtype_counts[kind] = dtype_counts.get(kind, 0) + 1
    size = int(df.memory_usage(index=True, deep=True).sum())
    reference_row_bytes = max(1, int(df.loc[train, features].memory_usage(index=True, deep=True).sum()
                                     / int(train.sum())))
    return {
        "rows": len(df), "reference_rows": int(train.sum()), "validation_rows": int(valid.sum()),
        "features": len(features), "dtypes": dtype_counts,
        "missing_cells": int(df[features].isna().sum().sum()), "dataframe_bytes": size,
        "reference_row_bytes": reference_row_bytes,
        # Planning headroom is a logical envelope, not an observation of the Studio host.
        "cpu_count": None, "ram_total_bytes": 1024 * GIB,
        "ram_available_bytes": 1024 * GIB, "scratch_free_bytes": 1024 * GIB,
        "gpu": {"available": True, "name": "candidate Processing GPU", "vram_bytes": 80 * GIB},
        "planning_resource": "remote_candidate_envelope",
    }


def estimate_model_resources(profile: dict[str, Any], context_rows: int,
                             feature_count: int, batch_rows: int, model_name: str) -> dict[str, Any]:
    """Conservative initial sizing, with explicit uncertainty until GPU calibration."""
    row_bytes = max(1, profile["dataframe_bytes"] / profile["rows"])
    reference_bytes = max(row_bytes * context_rows, context_rows * feature_count * 8)
    validation_bytes = max(row_bytes * profile["validation_rows"],
                           profile["validation_rows"] * feature_count * 8)
    # Full snapshot, transformations, Python/Arrow copies, output and cache margin.
    cpu_gib = 6 + (profile["dataframe_bytes"] * 4 + reference_bytes * 4 + validation_bytes * 2) / GIB
    # Single-process worker uses one GPU. These are planning heuristics, not measured peaks.
    model_floor = {"TabICLv2": 12, "TabPFN": 8, "Mitra": 12, "MitraV2": 12}.get(model_name, 10)
    # A query can attend to reference tokens; account for that interaction as
    # well as feature storage. This remains a heuristic until GPU measurements.
    query_context_bytes = context_rows * batch_rows * 64
    vram_gib = model_floor + (context_rows * feature_count * 32 +
                              batch_rows * feature_count * 64 + query_context_bytes) / GIB
    scratch_gib = 20 + (profile["dataframe_bytes"] * 3 + reference_bytes * 2 + validation_bytes) / GIB
    return {
        "cpu_ram_gib": round(cpu_gib * 1.3, 3),
        "vram_per_gpu_gib": round(vram_gib * 1.3, 3),
        "scratch_gib": round(scratch_gib * 1.3, 3),
        "input_bytes": profile["dataframe_bytes"], "reference_rows": context_rows,
        "validation_rows": profile["validation_rows"], "features": feature_count,
        "prediction_batch_rows": batch_rows,
        "method": "conservative_formula_v1",
        "uncertainty": "Uncalibrated checkpoint and transformer activations; real GPU smoke required",
    }


def build_cloud_plan(df: pd.DataFrame, config: BenchmarkConfig,
                     cloud: SageMakerConfig) -> SageMakerPlan:
    features, positive, negative = _validate_config(df, config)
    train, valid, source, cutoff = split_masks(df, config)
    profile = _remote_profile(df, features, train, valid)
    baseline = _assemble_plan(df, config, features, positive, negative,
                              train, valid, source, cutoff, profile)
    catalog, price_source, priced_at = processing_catalog(cloud)
    models: list[CloudModelPlan] = []
    for item in baseline.models:
        estimate = (estimate_model_resources(profile, item.context_rows, len(item.feature_cols),
                                             item.prediction_batch_rows, item.name)
                    if item.status == "ready" else {})
        candidates = suitable_candidates(estimate, cloud, catalog, price_source, priced_at) if estimate else []
        reasons = [item.reason] if item.reason else []
        if item.status == "ready" and not candidates:
            reasons.append("No priced sa-east-1 Processing GPU meets estimated RAM, per-GPU VRAM and EBS needs")
        models.append(CloudModelPlan(name=item.name, estimate=estimate, candidates=candidates,
                                     unavailable_reasons=reasons))
    plan = SageMakerPlan(run_id=new_cloud_run_id(), baseline=baseline, cloud=cloud,
                         models=models, created_utc=datetime.now(timezone.utc).isoformat())
    plan.identity = plan.compute_identity()
    return plan
