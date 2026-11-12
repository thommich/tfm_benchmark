# TabTune baseline

A local, reproducible **binary-classification** benchmark for TabTune foundation models. It reports Gini (`2 × ROC AUC − 1`) from positive-class probabilities on the complete validation set. Every model in the installed TabTune registry appears in the report. The runner scores only models whose registry weight license explicitly permits commercial use and whose `inference` path is genuinely zero-shot; xRFM is excluded because it trains from scratch.

## Install

Use Python 3.10–3.12. A CUDA-enabled PyTorch installation is useful for larger models. TabTune downloads many checkpoints on first use; its dependencies and model files can be large.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The package pins `tabtune==0.2.1`, the PyPI release used for development. The actual installed registry is read at runtime, so its model list can differ from the documentation site. Use an isolated environment for this package.

On Windows, this TabTune release's vendored TabPFN CPU memory probe references a missing upstream helper. The runner applies a narrow, version-guarded runtime fix for that model; it does not modify installed TabTune files.

## Use from a notebook or Python script

```python
from tabtune_baseline import BenchmarkConfig, plan_baseline, run_baseline

config = BenchmarkConfig(
    id_col="application_id",
    target_col="defaulted",
    split_col="sample",             # values: train / validation
    # time_col="application_date",  # alternative when no split column is supplied
    positive_label=1,                # inferred for 1/True; required otherwise
)
plan = plan_baseline(df, config)

# Inspect and optionally edit each ready ModelPlan before running.
for model in plan.models:
    print(model.name, model.status, model.context_rows, len(model.feature_cols), model.adaptations, model.reason)

result = run_baseline(df, plan, output_dir="./baseline-runs")
print(result.leaderboard())
print(result.report_html)
```

An explicit split column takes precedence over a time column. Without one, `time_col` must span at least six calendar months; the latest three calendar months become validation. For example, when the latest timestamp is December 1, the validation period starts September 1. ID, target, split, and time columns are excluded from model features unless a future version introduces an explicit feature policy for them.

The `RunPlan` is a dataclass. Save it with `json.dumps(plan.to_dict())` and restore with `RunPlan.from_dict(...)`. A run rejects a changed dataframe, invalid model eligibility, and any edited context or feature count outside a model's registered envelope. Pass `per_model_timeout_seconds` in `BenchmarkConfig` to bound each attempt; the default is unlimited.

## Adaptation and interpretation

- All validation rows are scored for every completed model. Validation rows are never used for preprocessing, supervised feature selection, or context selection.
- When a model has a feature ceiling, a deterministic univariate mutual-information selector uses **reference labels** to choose its columns. This is supervised feature selection, although foundation-model weights remain unchanged. The report shows the selected feature count.
- When a model has a context ceiling or the reference data exceeds a conservative RAM estimate, a seeded class-aware sample reduces **reference rows only**. TabICLv2 uses KV-cache offloading for large contexts before this fallback where supported. Resource failures trigger a smaller prediction batch, supported offloading, then a smaller reference context.
- Different models may use different features and reference contexts. Compare Gini scores with the adaptation column in view. A failed or unavailable model is reported rather than omitted.

## Output files

Each run gets a unique directory containing `report.html`, `results.json`, `manifest.json`, `events.jsonl`, a `scores/` Parquet file per completed model, and worker logs when present. Score files contain the validation ID, original target, and positive-class probability. The manifest records the dataframe fingerprint, split, policy version, environment, model configuration, and attempts. Successful model results record the checkpoint source and a local checkpoint SHA-256 when TabTune exposes its file; otherwise the checkpoint resolution is explicitly marked unavailable. Model files and input data stay local; temporary staged input is deleted when the run ends.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Unit tests use stub pipelines and do not download checkpoints. A real model smoke test is documented in `tests/test_real_smoke.py` and is opt-in because it downloads model weights.
