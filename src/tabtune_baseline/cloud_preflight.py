"""Read-only account checks and candidate quota filtering before job creation."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from urllib.request import Request, urlopen

from .models import InstanceCandidate, SageMakerConfig


@dataclass
class AccountPreflight:
    account_id: str
    quotas: dict[str, int] = field(default_factory=dict)
    active_jobs: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    active_job_names: set[str] = field(default_factory=set)

    def available(self, candidate: InstanceCandidate) -> int | None:
        quota = self.quotas.get(candidate.instance_type)
        if quota is None:
            return None
        return max(0, quota - self.active_jobs.get(candidate.instance_type, 0))


def _session(cloud: SageMakerConfig):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError('Install the optional cloud dependencies: pip install -e ".[cloud]"') from exc
    return boto3.Session(region_name=cloud.region)


def inspect_account(cloud: SageMakerConfig, *, session=None,
                    check_checkpoint_host: bool = True) -> AccountPreflight:
    """Validate the caller, bucket, IAM role, ECR digest and readable quotas.

    IAM/route checks from Studio cannot prove the Processing job role's runtime
    network access; the real GPU smoke remains the authority for that gate.
    """
    session = session or _session(cloud)
    try:
        caller = session.client("sts").get_caller_identity()
    except Exception as exc:
        raise RuntimeError(f"AWS account access failed: {exc}") from exc
    account = caller["Account"]
    if account != cloud.role_arn.split(":")[4] or account != cloud.image_uri.split(".")[0]:
        raise ValueError("IAM role, ECR image and caller must belong to the same AWS account")
    output = AccountPreflight(account_id=account)
    bucket = cloud.staging_uri[5:].split("/", 1)[0]
    try:
        s3 = session.client("s3")
        s3.head_bucket(Bucket=bucket)
        location = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
        if location != cloud.region:
            raise ValueError(f"Staging bucket is in {location!r}, expected {cloud.region}")
    except Exception as exc:
        raise RuntimeError(f"S3 staging bucket is inaccessible or in the wrong Region: {exc}") from exc
    role_name = cloud.role_arn.rsplit("/", 1)[-1]
    try:
        session.client("iam").get_role(RoleName=role_name)
    except Exception as exc:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in {"AccessDenied", "AccessDeniedException"}:
            output.warnings.append(f"Processing IAM role cannot be inspected: {exc}; submission will verify PassRole")
        else:
            raise RuntimeError(f"Processing IAM role cannot be inspected: {exc}") from exc
    repository = cloud.image_uri.split(".amazonaws.com/", 1)[1].split("@", 1)[0]
    digest = cloud.image_uri.rsplit("@", 1)[1]
    try:
        session.client("ecr").describe_images(
            repositoryName=repository, imageIds=[{"imageDigest": digest}])
    except Exception as exc:
        raise RuntimeError(f"Pinned ECR image digest is inaccessible: {exc}") from exc
    try:
        quotas = session.client("service-quotas")
        paginator = quotas.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode="sagemaker"):
            for entry in page.get("Quotas", []):
                name = entry.get("QuotaName", "").lower()
                if "processing job" not in name or "ml." not in name:
                    continue
                match = re.search(r"\bml\.[a-z0-9]+\.[a-z0-9]+\b", name)
                instance = match.group(0) if match else None
                if instance:
                    output.quotas[instance] = int(entry["Value"])
    except Exception as exc:
        output.warnings.append(f"Applied Processing quotas unavailable: {exc}; creation errors are authoritative")
    try:
        sm = session.client("sagemaker")
        paginator = sm.get_paginator("list_processing_jobs")
        for status in ("InProgress", "Stopping"):
            for page in paginator.paginate(StatusEquals=status):
                for item in page.get("ProcessingJobSummaries", []):
                    for retry in range(4):
                        try:
                            detail = sm.describe_processing_job(ProcessingJobName=item["ProcessingJobName"])
                            break
                        except Exception as exc:
                            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
                            if code not in {"Throttling", "ThrottlingException", "TooManyRequestsException"} or retry == 3:
                                raise
                            time.sleep(2 ** retry)
                    kind = detail["ProcessingResources"]["ClusterConfig"]["InstanceType"]
                    output.active_jobs[kind] = output.active_jobs.get(kind, 0) + 1
                    output.active_job_names.add(item["ProcessingJobName"])
    except Exception as exc:
        output.warnings.append(f"Active Processing jobs unavailable: {exc}; quota capacity may be overstated")
    if check_checkpoint_host:
        try:
            with urlopen(Request("https://huggingface.co", method="HEAD"), timeout=8):
                pass
        except Exception as exc:
            output.warnings.append(f"Checkpoint host unreachable from planner: {exc}; worker route is unverified")
    return output


def filter_quota(candidates: list[InstanceCandidate], preflight: AccountPreflight
                 ) -> tuple[list[InstanceCandidate], list[str]]:
    eligible: list[InstanceCandidate] = []
    reasons: list[str] = []
    for candidate in candidates:
        quota = preflight.quotas.get(candidate.instance_type)
        if quota == 0:
            reasons.append(f"{candidate.instance_type}: applied Processing quota is zero; request an increase in Service Quotas")
            continue
        eligible.append(candidate)
    return eligible, reasons
