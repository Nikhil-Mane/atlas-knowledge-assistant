"""Raw file storage, content-addressed by SHA-256.

`local` keeps files in a folder (one machine, or a volume shared by all
containers); `azure` uses Azure Blob Storage and `s3` any S3-compatible
store, so API and workers on different machines see the same files. The same
file uploaded twice maps to the same key, so storage never holds duplicates.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Protocol

from rag.config import Settings, get_settings

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def blob_key(sha256: str, suffix: str) -> str:
    # Two-level fan-out keeps any one directory small (10K files/day -> ~3.6M/year).
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}{suffix.lower()}"


class BlobStore(Protocol):
    def put(self, path: Path) -> tuple[str, str]:
        """Store a file. Returns (sha256, key)."""

    def exists(self, key: str) -> bool: ...

    def fetch(self, key: str, dest: Path) -> Path:
        """Copy a stored file to `dest` and return it."""

    def delete(self, key: str) -> None:
        """Remove a stored file (no error if it's already gone)."""

    def check(self) -> None:
        """Raise if the store is not usable."""


class LocalBlobStore:
    def __init__(self, root: Path):
        self.root = root

    def put(self, path: Path) -> tuple[str, str]:
        sha = sha256_file(path)
        key = blob_key(sha, path.suffix)
        target = self.root / key
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(path, tmp)
            tmp.replace(target)   # atomic: readers never see a half-written file
        return sha, key

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def fetch(self, key: str, dest: Path) -> Path:
        shutil.copyfile(self.root / key, dest)
        return dest

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)

    def check(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / ".healthcheck"
        probe.write_text("ok")
        probe.unlink()


class S3BlobStore:
    def __init__(self, bucket: str, endpoint_url: str | None = None):
        import boto3  # optional dependency: pip install -e .[s3]

        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)

    def put(self, path: Path) -> tuple[str, str]:
        sha = sha256_file(path)
        key = blob_key(sha, path.suffix)
        if not self.exists(key):
            self.s3.upload_file(str(path), self.bucket, key)
        return sha, key

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def fetch(self, key: str, dest: Path) -> Path:
        self.s3.download_file(self.bucket, key, str(dest))
        return dest

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def check(self) -> None:
        self.s3.head_bucket(Bucket=self.bucket)


class AzureBlobStore:
    def __init__(self, connection_string: str, container: str):
        from azure.storage.blob import BlobServiceClient  # pip install -e .[azure]

        service = BlobServiceClient.from_connection_string(connection_string)
        self.container = service.get_container_client(container)

    def put(self, path: Path) -> tuple[str, str]:
        sha = sha256_file(path)
        key = blob_key(sha, path.suffix)
        blob = self.container.get_blob_client(key)
        if not blob.exists():
            with open(path, "rb") as f:
                blob.upload_blob(f, overwrite=True, max_concurrency=4)
        return sha, key

    def exists(self, key: str) -> bool:
        return self.container.get_blob_client(key).exists()

    def fetch(self, key: str, dest: Path) -> Path:
        with open(dest, "wb") as f:
            self.container.get_blob_client(key).download_blob(max_concurrency=4).readinto(f)
        return dest

    def delete(self, key: str) -> None:
        blob = self.container.get_blob_client(key)
        if blob.exists():
            blob.delete_blob(delete_snapshots="include")

    def check(self) -> None:
        if not self.container.exists():
            self.container.create_container()


def get_blob_store(settings: Settings | None = None) -> BlobStore:
    s = settings or get_settings()
    if s.blob_backend == "azure":
        if not s.azure_storage_connection_string:
            raise ValueError("BLOB_BACKEND=azure needs AZURE_STORAGE_CONNECTION_STRING")
        return AzureBlobStore(s.azure_storage_connection_string.get_secret_value(), s.azure_blob_container)
    if s.blob_backend == "s3":
        if not s.s3_bucket:
            raise ValueError("BLOB_BACKEND=s3 needs S3_BUCKET")
        return S3BlobStore(s.s3_bucket, s.s3_endpoint_url)
    return LocalBlobStore(s.blob_root)
