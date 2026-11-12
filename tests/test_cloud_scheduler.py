from tabtune_baseline.cloud_catalog import processing_catalog, suitable_candidates
from tabtune_baseline.cloud_preflight import AccountPreflight
from tabtune_baseline.cloud_scheduler import (CapacityTokens, dispatch_queue, initial_state,
                                               next_candidate, orchestrator_step)
from tabtune_baseline import BenchmarkConfig, plan_sagemaker_baseline
from datetime import datetime, timezone
from test_baseline import frame
from test_sagemaker_contracts import settings


def choices():
    catalog, source, at = processing_catalog(settings(), use_live=False)
    estimate = {"cpu_ram_gib": 10, "vram_per_gpu_gib": 13,
                "scratch_gib": 25, "reference_rows": 100, "validation_rows": 10}
    return suitable_candidates(estimate, settings(), catalog, source, at)


def test_concurrency_and_per_type_token_release_once():
    candidate = choices()[0]
    account = AccountPreflight("123456789012", {candidate.instance_type: 1})
    tokens = CapacityTokens(3, account)
    tokens.reserve("job-1", candidate)
    assert next_candidate([candidate], tokens) is None
    assert tokens.release("job-1") is True
    assert tokens.release("job-1") is False
    tokens.reserve("job-2", candidate)
    assert tokens.running == {"job-2": candidate.instance_type}


def test_global_concurrency_and_cheapest_available_fallback():
    candidates = choices()
    account = AccountPreflight("123456789012", {c.instance_type: 1 for c in candidates})
    tokens = CapacityTokens(1, account)
    first = next_candidate(candidates, tokens)
    tokens.reserve("job-1", first)
    assert next_candidate(candidates, tokens) is None
    tokens.release("job-1")
    assert next_candidate(candidates, tokens, {first.instance_type}).instance_type != first.instance_type


def test_queue_advances_after_terminal_release():
    candidate = choices()[0]
    account = AccountPreflight("123456789012", {candidate.instance_type: 2})
    tokens = CapacityTokens(1, account)
    queue = ["A", "B"]
    calls = []
    submit = lambda model, c, name: calls.append((model, c.instance_type, name))
    args = (queue, {"A": [candidate], "B": [candidate]}, tokens,
            lambda model: f"job-{model}", submit)
    assert dispatch_queue(*args) == ["job-A"] and queue == ["B"]
    tokens.release("job-A")
    assert dispatch_queue(*args) == ["job-B"] and queue == []
    assert [call[0] for call in calls] == ["A", "B"]


def test_quota_rejection_falls_back_without_started_attempt():
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    for item in plan.baseline.models:
        if item.status == "ready" and item.name != "ILTM":
            item.status, item.reason = "excluded", "fixture"
    state = initial_state(plan, "checksum")
    for name, record in state["models"].items():
        if name != "ILTM":
            record["status"] = "excluded"
    cloud = next(m for m in plan.models if m.name == "ILTM")
    account = AccountPreflight("123456789012", {c.instance_type: 1 for c in cloud.candidates})
    tokens = CapacityTokens(3, account)
    class Adapter:
        calls = []
        def submit(self, model, number, job, kind):
            self.calls.append(kind)
            if len(self.calls) == 1:
                raise RuntimeError("ResourceLimitExceeded: quota reached")
        def describe(self, job):
            class NotFound(Exception):
                response = {"Error": {"Code": "ResourceNotFound"}}
            raise NotFound()
    adapter = Adapter()
    orchestrator_step(plan, state, tokens, adapter, lambda *args: None,
                      lambda *args: {}, lambda _: None, lambda *args, **kwargs: None)
    record = state["models"]["ILTM"]
    assert len(adapter.calls) == 2 and adapter.calls[0] != adapter.calls[1]
    assert len(record["attempts"]) == 1 and record["attempts"][0]["number"] == 1
    assert len(record["rejections"]) == 1 and len(tokens.running) == 1


def test_ambiguous_submit_response_reconciles_existing_job():
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    state = initial_state(plan, "checksum")
    name = next(m.name for m in plan.baseline.models if m.status == "ready")
    for other, record in state["models"].items():
        if other != name:
            record["status"] = "excluded"
    choice = next(m.candidates[0] for m in plan.models if m.name == name)
    tokens = CapacityTokens(1, AccountPreflight("123456789012", {choice.instance_type: 1}))
    class Adapter:
        def submit(self, *args):
            raise TimeoutError("response lost")
        def describe(self, job):
            return {"ProcessingJobStatus": "InProgress"}
    orchestrator_step(plan, state, tokens, Adapter(), lambda *args: None,
                      lambda *args: {}, lambda _: None, lambda *args, **kwargs: None)
    assert state["models"][name]["status"] == "running"
    assert len(state["models"][name]["attempts"]) == 1


def test_own_job_wait_does_not_consume_external_capacity_timer():
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    name = next(m.name for m in plan.baseline.models if m.status == "ready")
    for remote in plan.models:
        if remote.name == name:
            remote.candidates = remote.candidates[:1]
    state = initial_state(plan, "checksum")
    for other, record in state["models"].items():
        if other != name:
            record["status"] = "excluded"
    record = state["models"][name]
    record["wait_since"] = "2020-01-01T00:00:00+00:00"
    candidate = next(m.candidates[0] for m in plan.models if m.name == name)
    tokens = CapacityTokens(3, AccountPreflight("123456789012", {candidate.instance_type: 1}))
    tokens.running["own-job"] = candidate.instance_type
    orchestrator_step(plan, state, tokens, object(), lambda *args: None,
                      lambda *args: {}, lambda _: None, lambda *args, **kwargs: None,
                      clock=datetime(2026, 11, 12, tzinfo=timezone.utc))
    assert record["status"] == "pending" and record["wait_since"] is None


def test_quota_rejection_is_remembered_on_later_attempt():
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    name = next(m.name for m in plan.baseline.models if m.status == "ready")
    state = initial_state(plan, "checksum")
    for other, record in state["models"].items():
        if other != name:
            record["status"] = "excluded"
    record = state["models"][name]
    choices = next(m.candidates for m in plan.models if m.name == name)
    record["attempts"] = [{"job_name": "first", "number": 1}]
    record["rejections"] = [{"number": 1, "instance_type": choices[0].instance_type,
                             "kind": "quota", "reason": "quota reached"}]
    tokens = CapacityTokens(3, AccountPreflight("123456789012",
                                                 {c.instance_type: 1 for c in choices}))
    class Adapter:
        selected = None
        def submit(self, model, number, job, kind):
            self.selected = kind
    adapter = Adapter()
    orchestrator_step(plan, state, tokens, adapter, lambda *args: None,
                      lambda *args: {}, lambda _: None, lambda *args, **kwargs: None)
    assert adapter.selected != choices[0].instance_type
    assert record["attempts"][-1]["number"] == 2
