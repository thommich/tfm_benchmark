"""Public, optional SageMaker Processing API.

Importing this module never loads the SageMaker SDK or a model checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import BenchmarkConfig, CloudBenchmarkResult, SageMakerConfig, SageMakerPlan


def plan_sagemaker_baseline(df: pd.DataFrame, config: BenchmarkConfig,
                            cloud_config: SageMakerConfig) -> SageMakerPlan:
    """Build an inspectable CPU-only plan without uploading data or creating jobs."""
    from .cloud_planning import build_cloud_plan
    return build_cloud_plan(df, config, cloud_config)


def run_sagemaker_baseline(df: pd.DataFrame, cloud_plan: SageMakerPlan,
                            output_dir: str | Path) -> CloudBenchmarkResult:
    """Stage a reviewed snapshot and schedule GPU Processing jobs."""
    from .cloud_runtime import launch_run
    return launch_run(df, cloud_plan, Path(output_dir))


def resume_sagemaker_baseline(run_uri: str, output_dir: str | Path) -> CloudBenchmarkResult:
    """Reconcile a staged run and continue it without duplicate submissions."""
    from .cloud_runtime import resume_run
    return resume_run(run_uri, Path(output_dir))
