from types import SimpleNamespace

from tabtune_baseline.cloud_adapter import SageMakerAdapter
from tabtune_baseline.models import CloudOutputLayout
from test_sagemaker_contracts import settings


def test_script_processor_request_contract(monkeypatch):
    captured = {}

    class ScriptProcessor:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run(self, **kwargs):
            captured["run"] = kwargs

    def Input(**kwargs):
        return SimpleNamespace(**kwargs)

    def Output(**kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("tabtune_baseline.cloud_adapter._sdk",
                        lambda: (Input, Output, ScriptProcessor, object))
    adapter = SageMakerAdapter(settings(), SimpleNamespace(client=lambda _: object()), sdk_session=object())
    layout = CloudOutputLayout("s3://private-benchmark/runs/run123/")
    assert adapter.submit(layout, model="TabPFN", number=2,
                          job_name="tabtune-run123-tabpfn-2", instance_type="ml.g5.xlarge") == "tabtune-run123-tabpfn-2"
    assert captured["init"]["image_uri"] == settings().image_uri
    assert captured["init"]["instance_type"] == "ml.g5.xlarge"
    assert captured["init"]["instance_count"] == 1
    assert captured["init"]["max_runtime_in_seconds"] == 21600
    assert captured["run"]["job_name"] == "tabtune-run123-tabpfn-2"
    assert captured["run"]["wait"] is False and captured["run"]["logs"] is False
    assert len(captured["run"]["inputs"]) == 3
    assert captured["run"]["outputs"][0].destination == layout.attempt_output_uri("TabPFN", 2)


def test_kms_applies_to_outputs_and_only_ebs_volumes(monkeypatch):
    captured = []
    class ScriptProcessor:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def run(self, **kwargs):
            pass
    monkeypatch.setattr("tabtune_baseline.cloud_adapter._sdk",
                        lambda: (lambda **kwargs: kwargs, lambda **kwargs: kwargs,
                                 ScriptProcessor, object))
    from dataclasses import replace
    adapter = SageMakerAdapter(replace(settings(), kms_key_id="arn:aws:kms:sa-east-1:123456789012:key/test"),
                               SimpleNamespace(client=lambda _: object()), sdk_session=object())
    layout = CloudOutputLayout("s3://private-benchmark/runs/run123/")
    adapter.submit(layout, model="TabPFN", number=1, job_name="one", instance_type="ml.g5.xlarge")
    adapter.submit(layout, model="TabPFN", number=2, job_name="two", instance_type="ml.p5.4xlarge")
    assert captured[0]["output_kms_key"] and captured[0]["volume_kms_key"] is None
    assert captured[1]["volume_kms_key"] == captured[1]["output_kms_key"]
