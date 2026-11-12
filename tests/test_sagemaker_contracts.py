from dataclasses import replace

import pytest

from tabtune_baseline.models import (
    CloudModelPlan, InstanceCandidate, SageMakerConfig, SageMakerPlan,
)
from test_baseline import frame
from tabtune_baseline import BenchmarkConfig, plan_baseline
from tabtune_baseline import plan_sagemaker_baseline
from tabtune_baseline.cloud_catalog import processing_catalog, suitable_candidates
from tabtune_baseline.cloud_preflight import AccountPreflight, filter_quota, inspect_account
from tabtune_baseline.cloud_planning import estimate_model_resources


def settings():
    return SageMakerConfig(
        role_arn="arn:aws:iam::123456789012:role/Benchmark",
        image_uri="123456789012.dkr.ecr.sa-east-1.amazonaws.com/baseline@sha256:" + "a" * 64,
        staging_uri="s3://private-benchmark/runs/",
    )


def test_cloud_contract_roundtrip_and_tamper_detection():
    baseline = plan_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"))
    plan = SageMakerPlan("run123", baseline, settings(),
                         [CloudModelPlan(m.name, {}) for m in baseline.models],
                         "2026-11-12T00:00:00Z")
    plan.identity = plan.compute_identity()
    restored = SageMakerPlan.from_dict(plan.to_dict())
    assert restored.identity == plan.identity
    payload = plan.to_dict()
    payload["cloud"]["max_concurrent_jobs"] = 8
    with pytest.raises(ValueError, match="identity changed"):
        SageMakerPlan.from_dict(payload)
    with pytest.raises(ValueError, match="pinned"):
        replace(settings(), image_uri="image:latest")


def test_cpu_only_cloud_plan_uses_remote_gpu_and_full_reference(monkeypatch):
    monkeypatch.setattr("tabtune_baseline.planning.psutil.virtual_memory",
                        lambda: type("Memory", (), {"total": 2**30, "available": 1024})())
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    assert plan.baseline.profile["planning_resource"] == "remote_candidate_envelope"
    assert all(m.tuning_params["device"] == "cuda" and m.context_rows == 6
               for m in plan.baseline.models if m.status == "ready")
    assert SageMakerPlan.from_dict(plan.to_dict()).identity == plan.identity


def test_validation_values_do_not_change_reference_context_budget(monkeypatch):
    df = frame()
    df["extra"] = df["extra"].astype(str)
    changed = df.copy()
    changed.loc[changed["split"] == "validation", "extra"] = "x" * 100_000
    cfg = BenchmarkConfig("id", "target", split_col="split")
    left = plan_sagemaker_baseline(df, cfg, settings())
    right = plan_sagemaker_baseline(changed, cfg, settings())
    assert left.baseline.profile["reference_row_bytes"] == right.baseline.profile["reference_row_bytes"]
    assert {m.name: m.context_rows for m in left.baseline.models} == {
        m.name: m.context_rows for m in right.baseline.models}


def test_regional_processing_price_order_and_per_gpu_limit():
    catalog, source, at = processing_catalog(settings(), use_live=False)
    assert any(r["type"] == "ml.p5.4xlarge" for r in catalog)
    assert not any(r["type"] in {"ml.p4d.24xlarge", "ml.p5.48xlarge"} for r in catalog)
    estimate = {"cpu_ram_gib": 10, "vram_per_gpu_gib": 20,
                "scratch_gib": 25, "reference_rows": 100, "validation_rows": 10}
    ordered = suitable_candidates(estimate, settings(), catalog, source, at)
    assert ordered and ordered == sorted(ordered, key=lambda c: (c.usd_per_minute, c.instance_type))
    assert all(c.vram_per_gpu_gib >= 20 for c in ordered)
    assert not any(c.instance_type.startswith("ml.g4dn") for c in ordered)


def test_zero_quota_is_explained_and_other_types_remain():
    catalog, source, at = processing_catalog(settings(), use_live=False)
    estimate = {"cpu_ram_gib": 10, "vram_per_gpu_gib": 13,
                "scratch_gib": 25, "reference_rows": 100, "validation_rows": 10}
    choices = suitable_candidates(estimate, settings(), catalog, source, at)
    preflight = AccountPreflight("123456789012", {choices[0].instance_type: 0})
    allowed, reasons = filter_quota(choices, preflight)
    assert choices[0] not in allowed and len(allowed) == len(choices) - 1
    assert "request an increase" in reasons[0]


def test_preflight_reports_inaccessible_account():
    class Session:
        def client(self, name):
            raise RuntimeError("expired SSO grant")
    with pytest.raises(RuntimeError, match="expired SSO grant"):
        inspect_account(settings(), session=Session())


def test_vram_estimate_accounts_for_reference_query_interaction():
    profile = {"dataframe_bytes": 1_000_000, "rows": 100_000,
               "validation_rows": 10_000}
    small = estimate_model_resources(profile, 1_000, 100, 1024, "TabICLv2")
    large = estimate_model_resources(profile, 100_000, 100, 1024, "TabICLv2")
    assert large["vram_per_gpu_gib"] > small["vram_per_gpu_gib"] + 5


def test_preflight_role_path_and_getrole_denial_are_nonfatal():
    role_names = []
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}
    class Client:
        def __init__(self, service):
            self.service = service
        def get_caller_identity(self):
            return {"Account": "123456789012"}
        def head_bucket(self, **kwargs):
            return {}
        def get_bucket_location(self, **kwargs):
            return {"LocationConstraint": "sa-east-1"}
        def get_role(self, *, RoleName):
            role_names.append(RoleName)
            raise Denied()
        def describe_images(self, **kwargs):
            return {}
        def get_paginator(self, name):
            return self
        def paginate(self, **kwargs):
            return []
    class Session:
        def client(self, service):
            return Client(service)
    cloud = replace(settings(), role_arn="arn:aws:iam::123456789012:role/service-role/Benchmark")
    result = inspect_account(cloud, session=Session(), check_checkpoint_host=False)
    assert role_names == ["Benchmark"]
    assert any("cannot be inspected" in warning for warning in result.warnings)
