"""Opt-in checkpoint smoke test: TABTUNE_REAL_SMOKE=1 pytest -q tests/test_real_smoke.py."""

import os
import hashlib

import pandas as pd
import pytest

from tabtune_baseline import BenchmarkConfig, plan_baseline, run_baseline
from tabtune_baseline.worker import execute_model
from tabtune_baseline.compat import apply_tabtune_compat


@pytest.mark.skipif(os.getenv("TABTUNE_REAL_SMOKE") != "1", reason="Downloads a TabTune checkpoint")
def test_real_tabpfn_zero_shot(tmp_path):
    df = pd.DataFrame({
        "id": list(range(16)),
        "feature": [0, 1, 0, 1] * 4,
        "target": [0, 1, 0, 1] * 4,
        "split": ["train"] * 12 + ["validation"] * 4,
    })
    plan = plan_baseline(df, BenchmarkConfig("id", "target", split_col="split", prediction_batch_rows=2))
    model = next(m for m in plan.models if m.name == "TabPFN")
    assert model.status == "ready"
    result = execute_model(df, plan, model, tmp_path / "scores.parquet")
    assert result["scored_rows"] == 4
    assert result["probability_order"] == [0, 1]
    assert result["checkpoint"]["resolution"] == "local_file"
    assert len(result["checkpoint"]["sha256"]) == 64
    assert -1 <= result["gini"] <= 1
    for other in plan.models:
        if other.name != "TabPFN":
            other.status, other.reason = "excluded", "Smoke-test scope"
    complete = run_baseline(df, plan, tmp_path / "runs")
    assert complete.leaderboard()[0]["name"] == "TabPFN"
    assert pd.read_parquet(complete.leaderboard()[0]["scores_file"]).shape[0] == 4

    # Verify the vendored pretrained model is unchanged by prediction.
    from tabtune import TabularPipeline
    apply_tabtune_compat("TabPFN")
    pipeline = TabularPipeline("TabPFN", task_type="classification",
                               tuning_strategy="inference", license_mode="commercial",
                               tuning_params={"device": "cpu"})
    pipeline.fit(df.loc[:11, ["feature"]], df.loc[:11, "target"])

    def weights_hash():
        digest = hashlib.sha256()
        for name, tensor in pipeline.model.model_.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    before = weights_hash()
    pipeline.predict_proba(df.loc[12:, ["feature"]])
    assert weights_hash() == before
