# First SageMaker Processing run: validation record

Use this checklist for the first real run in `sa-east-1`. Keep AWS account IDs,
bucket names, dataset contents, credentials, and checkpoint files out of this
repository. Record non-sensitive evidence (job name, image digest, versions,
counts, statuses, and links to private AWS logs) in your private run notes.

## 1. Image and account gate

- Build and publish `cloud/Dockerfile` with `cloud/build_and_push.sh` in Linux
  CI/CodeBuild. Record the ECR digest and the successful `pip check` and
  `python -m tabtune_baseline.image_self_test` output. Confirm the running image
  has Python 3.11, TabTune 0.2.1, PyTorch 2.7.1 with CUDA 11.8, and the locked
  dependency versions. A local Windows virtual environment is not this test.
- In the Studio caller account, check the private staging bucket's Region,
  execution role ARN (including a possible `service-role/` path), ECR digest,
  applied Processing quotas, and currently active Processing jobs. A caller
  without `iam:GetRole` may continue with a warning; job submission must still
  prove `iam:PassRole` and the execution role's actual permissions.
- Confirm at least one suitable GPU type has a nonzero applied Processing quota.
  Compare the planned USD/minute, price source and publication date with the
  current São Paulo Processing price. Note whether the live price list or the
  bundled snapshot supplied the rate and how long planning took.
- If `kms_key_id` is configured, verify caller and execution role access to the
  key. The Processing output and S3 state/lease/artifacts must use that key,
  including after resume. A local-NVMe instance must **not** request a volume
  KMS key; an EBS-only instance may. Check the submitted job description.

## 2. One-model, small-data smoke

Use a small, non-sensitive binary dataset with an explicit reference/validation
split and known positive label. Plan on a CPU-only Studio kernel. For this
first smoke, retain only `TabPFN` as ready and mark other ready registry entries
excluded with reason `First-run single-model smoke`, then recompute
`plan.identity = plan.compute_identity()` and inspect the resulting plan before
launch. All registry entries must still appear in the final report. Do not run
the full benchmark before the one-model result passes these checks.

- Verify the plan uses only reference labels for any feature selection and only
  reference rows for context reduction. Record reference rows, validation rows,
  selected features, prediction batch, candidate type, estimated RAM/VRAM/disk,
  estimated cost, and any adaptations. The resource estimates are uncalibrated;
  compare them with observed peak usage and runtime after the job.
- Submit one Processing job and record its deterministic job name, image digest,
  instance type, job status, runtime limit, input/output S3 URIs, and actual
  runtime/cost estimate. There should be no access to an unrelated SageMaker
  default bucket and no duplicate submission for that attempt.
- In CloudWatch, verify CUDA sees the expected GPU and the worker reaches
  snapshot validation, fit, predict, and completion. Verify outbound checkpoint
  access from the **job role/network**, not merely from Studio. Confirm TabPFN
  loads v2 weights (`Prior-Labs/TabPFN-v2`), never v2.5, and records a local
  checkpoint SHA-256. If the checkpoint cannot be obtained, the job must fail;
  it must not report a completed result with random weights. The report must
  retain the v2 attribution requirement.
- Check that TabPFN and ILTM cache paths, when used, resolve under
  `/opt/ml/processing/checkpoints`. On a later TabICLv2 run with KV cache,
  confirm disk offload uses `/opt/ml/processing/scratch/tabiclv2_kv`, does not
  enter the uploaded output directory, and has enough free space.
- Compare `scores.parquet` with the original validation IDs and targets in
  order. There must be exactly one finite probability in `[0, 1]` per validation
  row, no missing or extra rows, and recomputing ROC AUC and
  `Gini = 2 * AUC - 1` must reproduce `result.json` and the leaderboard.
- Confirm `results.json`, `manifest.json`, `report.html`, `events.jsonl`, and
  every successful score file exist both in the private S3 run prefix and the
  requested Studio output directory. Confirm the staged input is deleted only
  after final artifacts are copied. Failed runs must retain it for resume.

## 3. Resume and failure evidence

After the basic smoke succeeds, use a separate controlled run to exercise
restart behavior while a job is active. Stop the Studio process, then call
`resume_sagemaker_baseline(run_uri, output_dir)`; also try a second notebook
against the same run. Record the original and resumed job names.

- The second notebook should wait and print read-only status until it acquires
  the lease. It must not create another job for an existing attempt. Resume must
  preserve the KMS key and count only this run's jobs that are still active when
  subtracting from account quota. A job that finished during the restart must
  reconcile as terminal before new dispatch.
- For any natural failure, compare `error.json`, SageMaker `FailureReason`,
  `cloudwatch.log`, state, events, and the final report. The short failure class
  and message must describe the actual exception; routine Hugging Face or
  memory-estimation startup messages must not turn an unrelated exception into
  a checkpoint or OOM failure. The full 20 KB log tail should be retained once
  under that attempt, not copied into state or HTML.
- If quota/capacity rejection occurs, verify the scheduler tries the next
  suitable priced type without consuming a started attempt and does not
  immediately retry a quota-rejected type on the next attempt. Time spent
  waiting for this run's own job should not consume the external-capacity timer.
- If an actual OOM occurs, record its phase, instance VRAM, batch/context sizes,
  selected next type, quota and every adaptation. A zero-quota larger GPU must
  never be selected for retry. Reference context must remain unchanged through
  started attempts 1–3; any reduction at attempt 4 or 5 needs a visible notice.
  If no permitted adaptation exists earlier, record the explicit terminal
  failure rather than claiming the full retry budget was used.

## Evidence to retain privately

| Gate | Pass/fail and evidence |
| --- | --- |
| Image build, self-test, ECR digest | |
| Account, role path, quota, pricing, KMS | |
| One GPU job, CUDA, checkpoint version/SHA | |
| Complete validation scores and recomputed Gini | |
| S3 and Studio artifact parity, staged-input lifecycle | |
| Resume, lease, quota accounting | |
| Any failure/retry classification and adaptation | |

The image build and real GPU smoke are required gates. Unit tests and mocked
SDK requests do not substitute for them.
