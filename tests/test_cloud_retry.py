from tabtune_baseline import BenchmarkConfig, plan_sagemaker_baseline
from tabtune_baseline.cloud_retry import Failure, adapt_for_retry, classify_failure
from test_baseline import frame
from test_sagemaker_contracts import settings


def fixture():
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    item = next(m for m in plan.baseline.models if m.status == "ready")
    candidates = next(m.candidates for m in plan.models if m.name == item.name)
    return item, candidates


def test_oom_phase_and_non_oom_failure_classification():
    assert classify_failure(worker={"error_type": "OutOfMemoryError", "phase": "predict"}).kind == "oom"
    assert classify_failure(worker={"error": "checkpoint download denied", "phase": "fit"}).kind == "checkpoint"
    assert classify_failure(description={"FailureReason": "Unknown container failure"}).kind == "unknown"
    assert classify_failure(submission_error="ResourceLimitExceeded: quota reached").kind == "quota"
    assert classify_failure(worker={"error_type": "ValueError", "error": "bad input"},
                            logs="Downloading checkpoint from HuggingFace; out of memory warning").kind == "unknown"
    assert classify_failure(description={"FailureReason": "AlgorithmError"},
                            logs="CUDA out of memory").kind == "oom"
    unrelated = classify_failure(description={"FailureReason": "AlgorithmError: container exited 1"},
                                 logs="Downloading checkpoint from HuggingFace\nAttributeError: bad method")
    assert unrelated.kind == "unknown" and "AttributeError" in unrelated.reason


def test_context_preserved_until_attempt_four_and_retry_limit():
    item, candidates = fixture()
    first = candidates[0]
    original_context = item.context_rows
    result = adapt_for_retry(item, Failure("oom", "predict", "CUDA out of memory"), 1, first, candidates)
    assert result and result[0].prediction_batch_rows < item.prediction_batch_rows
    assert result[0].context_rows == original_context
    second = adapt_for_retry(result[0], Failure("oom", "fit", "OOM"), 2, first, candidates)
    assert second and second[0].context_rows == original_context
    third = adapt_for_retry(second[0], Failure("oom", "fit", "OOM"), 3, second[1], candidates)
    assert third and third[0].context_rows < original_context and third[2]
    assert adapt_for_retry(third[0], Failure("oom", "fit", "OOM"), 5, third[1], candidates) is None
    assert adapt_for_retry(item, Failure("checkpoint", "fit", "denied"), 1, first, candidates) is None
