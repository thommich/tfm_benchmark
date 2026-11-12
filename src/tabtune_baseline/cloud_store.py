"""Private S3 run storage and compare-and-swap scheduler lease."""

from __future__ import annotations

import json
from threading import RLock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4


class LeaseHeld(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class S3RunStore:
    def __init__(self, s3, run_uri: str, *, kms_key_id: str | None = None):
        parsed = urlparse(run_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("run_uri must be a private S3 run prefix")
        self.s3 = s3
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.run_uri = f"s3://{self.bucket}/{self.prefix}/"
        self.kms_key_id = kms_key_id
        self.owner = uuid4().hex
        self.lease_etag: str | None = None
        self.state_etag: str | None = None
        self.lease_failure: Exception | None = None
        self._lease_lock = RLock()

    def key(self, suffix: str) -> str:
        return f"{self.prefix}/{suffix.lstrip('/')}"

    def uri(self, suffix: str) -> str:
        return f"s3://{self.bucket}/{self.key(suffix)}"

    def _encryption(self) -> dict:
        return ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.kms_key_id}
                if self.kms_key_id else {"ServerSideEncryption": "AES256"})

    def put_bytes(self, suffix: str, data: bytes, **conditions) -> dict:
        return self.s3.put_object(Bucket=self.bucket, Key=self.key(suffix), Body=data,
                                  **self._encryption(), **conditions)

    def put_json(self, suffix: str, value: dict, **conditions) -> dict:
        return self.put_bytes(suffix, json.dumps(value, sort_keys=True, default=str,
                                                  allow_nan=False).encode(), **conditions)

    def get_object(self, suffix: str) -> dict:
        return self.s3.get_object(Bucket=self.bucket, Key=self.key(suffix))

    def get_json(self, suffix: str) -> dict:
        return json.loads(self.get_object(suffix)["Body"].read())

    def get_state(self) -> dict:
        current = self.get_object("state.json")
        self.state_etag = current["ETag"]
        return json.loads(current["Body"].read())

    def put_state(self, state: dict) -> None:
        self.assert_lease()
        condition = ({"IfMatch": self.state_etag} if self.state_etag else
                     {"IfNoneMatch": "*"})
        try:
            self.state_etag = self.put_json("state.json", state, **condition)["ETag"]
        except Exception as exc:
            if _conflict(exc):
                raise LeaseHeld("Run state changed under another scheduler") from exc
            raise

    def maybe_bytes(self, suffix: str) -> bytes | None:
        try:
            return self.get_object(suffix)["Body"].read()
        except Exception as exc:
            if _not_found(exc):
                return None
            raise

    def upload_file(self, suffix: str, path: Path) -> None:
        self.s3.upload_file(str(path), self.bucket, self.key(suffix),
                            ExtraArgs=self._encryption())

    def download_file(self, suffix: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, self.key(suffix), str(path))

    def delete(self, suffix: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self.key(suffix))

    def acquire_lease(self, *, seconds: int = 180) -> bool:
        now = _utcnow()
        lease = {"owner": self.owner, "expires_utc": (now + timedelta(seconds=seconds)).isoformat()}
        try:
            existing = self.get_object("lease.json")
        except Exception as exc:
            if not _not_found(exc):
                raise
            try:
                self.lease_etag = self.put_json("lease.json", lease, IfNoneMatch="*")["ETag"]
                return True
            except Exception as conflict:
                if _conflict(conflict):
                    return False
                raise
        current = json.loads(existing["Body"].read())
        if datetime.fromisoformat(current["expires_utc"]) > now:
            return False
        try:
            self.lease_etag = self.put_json("lease.json", lease, IfMatch=existing["ETag"])["ETag"]
            return True
        except Exception as conflict:
            if _conflict(conflict):
                return False
            raise

    def renew_lease(self, *, seconds: int = 180) -> None:
        with self._lease_lock:
            self.assert_lease()
            lease = {"owner": self.owner, "expires_utc": (_utcnow() + timedelta(seconds=seconds)).isoformat()}
            try:
                self.lease_etag = self.put_json("lease.json", lease, IfMatch=self.lease_etag)["ETag"]
            except Exception as exc:
                raise LeaseHeld("Scheduler lease renewal failed; stop submitting jobs") from exc

    def assert_lease(self) -> None:
        with self._lease_lock:
            if self.lease_failure is not None:
                raise LeaseHeld("Scheduler lease heartbeat failed; stop submitting jobs") from self.lease_failure
            if self.lease_etag is None:
                raise LeaseHeld("Scheduler does not hold the run lease")
            existing = self.get_object("lease.json")
            current = json.loads(existing["Body"].read())
            if (current["owner"] != self.owner or existing["ETag"] != self.lease_etag or
                    datetime.fromisoformat(current["expires_utc"]) <= _utcnow()):
                raise LeaseHeld("Scheduler lease changed or expired; stop submitting jobs")

    def release_lease(self) -> None:
        with self._lease_lock:
            if self.lease_etag is None:
                return
            existing = self.get_object("lease.json")
            current = json.loads(existing["Body"].read())
            if current["owner"] == self.owner and existing["ETag"] == self.lease_etag:
                self.put_json("lease.json", {"owner": self.owner,
                                             "expires_utc": _utcnow().isoformat()},
                              IfMatch=self.lease_etag)
            self.lease_etag = None


def _error_code(exc: Exception) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))


def _not_found(exc: Exception) -> bool:
    return _error_code(exc) in {"404", "NoSuchKey", "NotFound"}


def _conflict(exc: Exception) -> bool:
    return _error_code(exc) in {"409", "412", "PreconditionFailed", "ConditionalRequestConflict"}
