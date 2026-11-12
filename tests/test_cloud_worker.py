import hashlib
import json
from dataclasses import asdict

import pandas as pd

from tabtune_baseline import BenchmarkConfig, plan_sagemaker_baseline
from tabtune_baseline.cloud_worker import run_one
from test_baseline import frame
from test_sagemaker_contracts import settings


def test_processing_worker_checks_snapshot_and_complete_scores(tmp_path, monkeypatch):
    import torch
    df = frame()
    plan = plan_sagemaker_baseline(df, BenchmarkConfig("id", "target", split_col="split"), settings())
    item = next(m for m in plan.baseline.models if m.status == "ready")
    data = tmp_path / "input.parquet"
    df.to_parquet(data, index=False)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.to_dict()))
    attempt_file = tmp_path / "attempt.json"
    attempt_file.write_text(json.dumps({"plan_identity": plan.identity,
                                        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                                        "model": item.name, "number": 1,
                                        "model_plan": asdict(item)}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: type("GPU", (), {"name": "fixture", "total_memory": 24 * 1024**3})())

    def fake_execute(frame, baseline, model, path, *, on_phase):
        on_phase("predict")
        pd.DataFrame({"id": range(6, 12), "target": [0, 1] * 3,
                      "score_positive": [0.1, 0.9] * 3}).to_parquet(path, index=False)
        return {"status": "completed", "name": model.name, "scored_rows": 6,
                "gini": 1.0, "auc": 1.0, "seconds": 0.1}

    monkeypatch.setattr("tabtune_baseline.cloud_worker.execute_model", fake_execute)
    out = tmp_path / "output"
    result = run_one(data, plan_file, attempt_file, out)
    assert result["status"] == "completed" and len(pd.read_parquet(out / "scores.parquet")) == 6
    assert (out / "result.json").is_file()
    attempt = json.loads(attempt_file.read_text())
    attempt["dataset_sha256"] = "bad"
    attempt_file.write_text(json.dumps(attempt))
    failed = run_one(data, plan_file, attempt_file, tmp_path / "bad")
    assert failed["status"] == "failed" and failed["phase"] == "identity"
