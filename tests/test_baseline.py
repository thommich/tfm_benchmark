from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabtune_baseline import BenchmarkConfig, plan_baseline, run_baseline
from tabtune_baseline.models import RunPlan
from tabtune_baseline.planning import dataframe_fingerprint, split_masks
from tabtune_baseline.report import render_report
from tabtune_baseline.runner import _fallback
from tabtune_baseline.worker import execute_model, select_context


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": list(range(12)), "score": [0, 1] * 6, "extra": list(range(12)),
        "target": [0, 1] * 6, "split": ["train"] * 6 + ["validation"] * 6,
    })


def test_explicit_split_and_positive_label():
    df = frame()
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split"))
    assert (plan.train_rows, plan.validation_rows, plan.positive_label) == (6, 6, 1)
    assert plan.feature_cols == ["score", "extra"]
    assert len(plan.models) == len(__import__("tabtune").registry.list_models())
    assert next(m for m in plan.models if m.name == "XRFM").status == "excluded"
    assert RunPlan.from_dict(plan.to_dict()).fingerprint == plan.fingerprint


def test_temporal_last_three_calendar_months_and_minimum_span():
    df = frame().drop(columns="split")
    df["date"] = pd.date_range("2025-01-01", periods=12, freq="MS")
    cfg = BenchmarkConfig("id", "target", time_col="date")
    train, valid, source, cutoff = split_masks(df, cfg)
    assert source == "time"
    assert cutoff.startswith("2025-09-01")
    assert train.sum() == 8 and valid.sum() == 4
    plan = plan_baseline(df, cfg)
    assert plan.validation_rows == 4
    df["date"] = pd.date_range("2025-01-01", periods=12, freq="D")
    with pytest.raises(ValueError, match="six calendar months"):
        plan_baseline(df, cfg)


@pytest.mark.parametrize("change,message", [
    (lambda d: d.drop(columns="split"), "Missing required columns"),
    (lambda d: d.assign(id=0), "unique"),
    (lambda d: d.assign(target=0), "exactly two"),
])
def test_data_validation(change, message):
    with pytest.raises(ValueError, match=message):
        plan_baseline(change(frame()), BenchmarkConfig("id", "target", split_col="split"))


def test_ambiguous_positive_label_requires_choice():
    df = frame()
    df["target"] = df["target"].map({0: "good", 1: "bad"})
    with pytest.raises(ValueError, match="positive_label"):
        plan_baseline(df, BenchmarkConfig("id", "target", split_col="split"))
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split", positive_label="bad"))
    assert plan.positive_label == "bad"


def test_context_selection_is_seeded_and_keeps_both_classes():
    df = frame().iloc[:10]
    X, y = df[["score"]], df["target"]
    a, ay = select_context(X, y, 4, 7)
    b, by = select_context(X, y, 4, 7)
    assert a.index.tolist() == b.index.tolist()
    assert set(ay) == {0, 1} and len(a) == 4


def test_supervised_feature_selection_and_envelope(monkeypatch):
    from tabtune.registry import CapabilityEnvelope, LicenseSpec, ModelSpec
    import tabtune.registry as registry

    spec = ModelSpec(
        name="FixtureModel", family="icl",
        classification_strategies=frozenset({"inference"}),
        envelope=CapabilityEnvelope(max_features=1, max_rows=4),
        license=LicenseSpec(name="MIT", commercial_use_ok=True),
    )
    monkeypatch.setattr(registry, "list_models", lambda: [spec])
    df = frame()
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split"))
    item = plan.models[0]
    assert item.status == "ready"
    assert item.feature_cols == ["score"]
    assert item.context_rows == 4
    assert len(item.adaptations) == 2


class StubPipeline:
    def __init__(self, *, reversed_order=False, **kwargs):
        assert kwargs["tuning_strategy"] == "inference"
        self.classes_ = [1, 0] if reversed_order else [0, 1]
        self.reversed_order = reversed_order
        self.fit_rows = None

    def fit(self, X, y):
        self.fit_rows = len(X)
        assert set(y) == {0, 1}
        return self

    def predict_proba(self, X):
        p = np.where(X["score"].to_numpy() == 1, 0.9, 0.1)
        return np.column_stack([p, 1 - p]) if self.reversed_order else np.column_stack([1 - p, p])


def test_gini_probability_order_and_full_validation(tmp_path):
    df = frame()
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split", prediction_batch_rows=2))
    item = next(m for m in plan.models if m.status == "ready")
    item.context_rows = 4
    for reversed_order in (False, True):
        path = tmp_path / f"scores-{reversed_order}.parquet"
        result = execute_model(df, plan, item, path,
                               pipeline_factory=lambda **kw: StubPipeline(reversed_order=reversed_order, **kw))
        saved = pd.read_parquet(path)
        assert result["gini"] == pytest.approx(1.0)
        assert result["scored_rows"] == plan.validation_rows == len(saved)
        assert result["context_rows"] == 4
        assert saved["id"].tolist() == list(range(6, 12))


def test_commercial_checkpoint_parameters_are_enforced(tmp_path, monkeypatch):
    from tabtune.models.tabpfn import model_loading
    import tabtune_baseline.worker as worker

    plan = plan_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"))
    for name in ("Mitra", "MitraV2"):
        item = next(m for m in plan.models if m.name == name)
        assert item.model_params["use_pretrained_weights"] is True

    resolved = []
    def fake_resolve(path, kind, version="v2"):
        resolved.append(version)
        return (tmp_path / "tabpfn-v2.ckpt", None, None, None)
    captured = {}
    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return StubPipeline(**kwargs)
    monkeypatch.setattr(model_loading, "resolve_model_path", fake_resolve)
    monkeypatch.setattr(worker, "_default_pipeline_factory", fake_pipeline)
    monkeypatch.setattr(worker, "apply_tabtune_compat", lambda _: None)
    item = next(m for m in plan.models if m.name == "TabPFN")
    item.model_params["model_path"] = "unreviewed-v2.5.ckpt"
    result = execute_model(frame(), plan, item, tmp_path / "scores.parquet",
                           pipeline_factory=fake_pipeline)
    assert captured["model_params"]["model_path"].endswith("tabpfn-v2.ckpt")
    assert result["model_params"] == captured["model_params"]
    assert resolved and set(resolved) == {"v2"}


def test_fingerprint_and_report_artifacts(tmp_path):
    df = frame()
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split"))
    assert dataframe_fingerprint(df) != dataframe_fingerprint(df.assign(extra=0))
    # Exclude execution in this test, preserving every registry row in the report.
    for item in plan.models:
        item.status, item.reason = "excluded", "Fixture"
    result = run_baseline(df, plan, tmp_path)
    assert Path(result.report_html).is_file()
    assert Path(result.results_json).is_file()
    assert Path(result.manifest_json).is_file()
    assert Path(result.log_jsonl).is_file()
    assert len(json.loads(Path(result.results_json).read_text())["models"]) == len(plan.models)
    assert "No model completed" in Path(result.report_html).read_text()
    with pytest.raises(ValueError, match="changed after planning"):
        run_baseline(df.assign(extra=0), plan, tmp_path)


def test_tabpfn_report_displays_required_attribution():
    plan = plan_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"))
    tabpfn = next(m for m in plan.models if m.name == "TabPFN")
    html = render_report(plan, [{"name": "TabPFN", "status": "completed",
                                 "gini": 0.5, "auc": 0.75, "context_rows": 6,
                                 "feature_count": 2, "scored_rows": 6, "seconds": 1.0,
                                 "license": tabpfn.license,
                                 "attribution_required": tabpfn.attribution_required}], "fixture")
    assert "Built with PriorLabs-TabPFN" in html


def test_bool_target_inference():
    df = frame().assign(target=lambda d: d["target"].astype(bool))
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split"))
    assert plan.positive_label is True


def test_oom_fallback_changes_only_execution_budget():
    plan = plan_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"))
    item = next(m for m in plan.models if m.name == "TabICLv2")
    original_features = list(item.feature_cols)
    original_validation = item.validation_rows
    assert _fallback(item, 0)
    assert item.prediction_batch_rows == 512
    assert _fallback(item, 1)
    assert item.model_params["offload_mode"] == "auto"
    assert _fallback(item, 2)
    assert item.context_rows == 3
    assert item.feature_cols == original_features and item.validation_rows == original_validation


def test_model_failure_does_not_stop_next_model(tmp_path, monkeypatch):
    df = frame()
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split"))
    eligible = [m for m in plan.models if m.status == "ready"]
    assert len(eligible) >= 2
    names = [m.name for m in eligible[:2]]
    for item in plan.models:
        if item.name not in names:
            item.status, item.reason = "excluded", "Fixture"

    def fake_run(cmd, **kwargs):
        model = cmd[cmd.index("--model") + 1]
        result_file = Path(cmd[cmd.index("--result") + 1])
        if model == names[0]:
            record = {"name": model, "status": "failed", "error": "checkpoint unavailable"}
        else:
            score_file = Path(cmd[cmd.index("--scores") + 1])
            pd.DataFrame({"id": range(6, 12), "target": [0, 1] * 3,
                          "score_positive": [0.1, 0.9] * 3}).to_parquet(score_file)
            record = {"name": model, "status": "completed", "gini": 1.0,
                      "auc": 1.0, "scored_rows": 6, "context_rows": 6,
                      "feature_count": 2, "seconds": 0.1, "scores_file": str(score_file)}
        result_file.write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tabtune_baseline.runner.subprocess.run", fake_run)
    result = run_baseline(df, plan, tmp_path)
    assert next(r for r in result.models if r["name"] == names[0])["status"] == "failed"
    assert result.leaderboard()[0]["name"] == names[1]
    assert "checkpoint unavailable" in Path(result.report_html).read_text()
