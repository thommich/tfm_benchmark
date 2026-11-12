import io
import json
from dataclasses import asdict

from tabtune_baseline.cloud_store import S3RunStore
from tabtune_baseline.cloud_scheduler import (initial_state, reconcile_submission_intents,
                                               restore_tokens, orchestrator_step)
from tabtune_baseline.cloud_preflight import AccountPreflight
from tabtune_baseline import BenchmarkConfig, plan_sagemaker_baseline
from test_baseline import frame
from test_sagemaker_contracts import settings


class S3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.serial = 0

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise S3Error("NoSuchKey")
        body, tag = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": tag}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **kwargs):
        if IfNoneMatch == "*" and Key in self.objects:
            raise S3Error("PreconditionFailed")
        if IfMatch and (Key not in self.objects or self.objects[Key][1] != IfMatch):
            raise S3Error("PreconditionFailed")
        self.serial += 1
        tag = f'"{self.serial}"'
        self.objects[Key] = (Body, tag)
        return {"ETag": tag}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.put_object(Bucket=bucket, Key=key, Body=open(path, "rb").read())

    def download_file(self, bucket, key, path):
        with open(path, "wb") as stream:
            stream.write(self.objects[key][0])

    def delete_object(self, *, Bucket, Key):
        del self.objects[Key]


def test_conditional_lease_excludes_duplicate_scheduler():
    s3 = FakeS3()
    first = S3RunStore(s3, "s3://bucket/run/1/")
    second = S3RunStore(s3, "s3://bucket/run/1/")
    assert first.acquire_lease()
    assert not second.acquire_lease()
    first.renew_lease()
    first.release_lease()
    assert second.acquire_lease()


def test_resume_reconciles_name_before_dispatch_and_preserves_concurrency():
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    state = initial_state(plan, "checksum")
    model = next(m for m in plan.baseline.models if m.status == "ready")
    for name, record in state["models"].items():
        if name != model.name:
            record["status"] = "excluded"
    candidate = next(m.candidates[0] for m in plan.models if m.name == model.name)
    job = "known-job"
    state["submission_intents"][job] = {"model": model.name, "number": 1,
                                         "job_name": job, "instance_type": candidate.instance_type,
                                         "status": "submitting"}
    class Adapter:
        submits = 0
        def describe(self, name):
            return {"ProcessingJobStatus": "InProgress"}
        def submit(self, *args):
            self.submits += 1
    adapter = Adapter()
    persist = lambda _: None
    emit = lambda *args, **kwargs: None
    reconcile_submission_intents(state, adapter, persist, emit)
    account = AccountPreflight("123456789012", {candidate.instance_type: 1}, {candidate.instance_type: 1})
    tokens = restore_tokens(plan, state, account)
    orchestrator_step(plan, state, tokens, adapter, lambda *args: None,
                      lambda *args: {}, persist, emit)
    assert len(state["models"][model.name]["attempts"]) == 1
    assert adapter.submits == 0 and len(tokens.running) == 1


def test_terminal_report_includes_registry_and_deletes_input_after_copy(tmp_path):
    from tabtune_baseline.cloud_artifacts import finalize_run
    plan = plan_sagemaker_baseline(frame(), BenchmarkConfig("id", "target", split_col="split"), settings())
    state = initial_state(plan, "checksum")
    state["status"] = "terminal"
    for record in state["models"].values():
        record["status"] = "excluded"
        record["reason"] = "fixture"
    s3 = FakeS3()
    store = S3RunStore(s3, "s3://bucket/run/1/")
    data = tmp_path / "input.parquet"
    frame().to_parquet(data, index=False)
    store.upload_file("staged/input.parquet", data)
    result = finalize_run(plan, state, store, tmp_path / "results")
    assert len(result.local_result.models) == len(plan.baseline.models)
    assert (tmp_path / "results" / plan.run_id / "report.html").is_file()
    assert store.key("staged/input.parquet") not in s3.objects
    assert store.key("report.html") in s3.objects
