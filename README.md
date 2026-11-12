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

## Optional SageMaker Processing path

The local API above is unchanged. The optional [Studio notebook](notebooks/sagemaker_processing.ipynb) plans on a CPU-only Python 3.11 Studio kernel. It does not load or download checkpoints. GPU Processing jobs use a custom ECR image pinned by digest and the SageMaker Python SDK v2 `ScriptProcessor`; each job downloads its own checkpoint into ephemeral storage.

Install in an isolated Studio environment:

```bash
python3.11 -m venv .venv-cloud
.venv-cloud/bin/python -m pip install -e '.[cloud]'
```

Build the [GPU image](cloud/Dockerfile) in Linux CI or CodeBuild using [the publish script](cloud/build_and_push.sh). The base is pinned by digest and `cloud/requirements-linux-py311.lock` contains transitive hashes. Record the ECR output digest and confirm `pip check`, the image self-test, and one real GPU Processing smoke before a production run. Build checks cannot establish the SageMaker host driver or checkpoint access. The image contains no model weights. SageMaker SDK v2 is in maintenance mode and reaches end of support on **2027-07-06**; the SDK calls are confined to `cloud_adapter.py`.

The Studio caller needs permission to inspect the account, Processing quotas and active jobs, role and ECR digest, and to submit/describe jobs and read CloudWatch Processing logs. It needs read/write/delete access to one **private** S3 prefix in `sa-east-1`. The Processing role needs ECR pull, S3 input/output, CloudWatch logging, and outbound access to checkpoint hosts such as Hugging Face (or a configured private route). If a KMS key is selected, both caller and job role need its S3 encryption permissions. Processing quotas may start at zero; request a per-type increase in Service Quotas. Quota availability does not reserve capacity.

```python
from tabtune_baseline import (
    BenchmarkConfig, SageMakerConfig, plan_sagemaker_baseline,
    run_sagemaker_baseline, resume_sagemaker_baseline,
)

cloud = SageMakerConfig(
    role_arn="arn:aws:iam::123456789012:role/your-processing-role",
    image_uri="123456789012.dkr.ecr.sa-east-1.amazonaws.com/your-repo@sha256:<64 hex characters>",
    staging_uri="s3://your-private-bucket/tabtune-runs/",
)
plan = plan_sagemaker_baseline(df, BenchmarkConfig("id", "target", split_col="split"), cloud)
for baseline, remote in zip(plan.baseline.models, plan.models):
    print(baseline.name, baseline.status, baseline.reason, remote.estimate,
          [(c.instance_type, c.estimated_cost_usd, c.price_source, c.price_at) for c in remote.candidates])
# After reviewing the plan and replacing the example AWS identifiers:
result = run_sagemaker_baseline(df, plan, "./baseline-runs")
# If the Studio process restarts, use the printed run URI:
result = resume_sagemaker_baseline(result.run_uri, "./baseline-runs")
```

Planning may query the public AWS Bulk Price List. It falls back to the bundled, dated São Paulo Processing price snapshot when pricing is unavailable. Candidates are ordered by on-demand USD/minute and filtered against estimated CPU RAM, **per-GPU** VRAM, and configured EBS; estimates are conservative but uncalibrated until the real smoke. No spending cap is imposed. The default is three concurrent jobs and one initial started job plus up to four OOM retries per model. The notebook reports status every 60 seconds. Quota/capacity rejections fall through to the next suitable priced type without consuming an attempt. OOM adaptations and every failure appear in `events.jsonl`, `results.json`, and `manifest.json`; an unknown failure is not treated as OOM.

Each run keeps state, attempt outputs, final artifacts and scores under the private S3 run URI. The staged input is deleted only after the run is terminal and the final S3 artifacts have been copied into Studio. A crash or blocked copy retains staged input for `resume_sagemaker_baseline`. A second active notebook monitors without submitting duplicate jobs. The local result is available as `result.local_result` after completion; inspect its `report_html` and `leaderboard()`.
