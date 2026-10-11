"""Small ADC-based GCS wrapper with explicit SHA-256 download verification.

Uploads create new objects only; downloads create new local files only. Failed
or mismatched downloads remain on disk for inspection. No operation deletes
files. Install the optional dependency with ``pip install -e '.[cloud]'``.
"""

import hashlib
import math
from pathlib import Path
import re

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage
from requests.exceptions import RequestException


class GCSOperationError(RuntimeError):
    """Authentication, network, permissions, or filesystem operation failed."""


class ChecksumMismatchError(GCSOperationError):
    """Downloaded bytes differ from the caller's expected SHA-256 digest."""


def sha256_file(path: str | Path) -> str:
    """Hash a file in 1 MiB blocks without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GCSStorage:
    """Reuse one client for one bucket; credentials come from ADC by default.

    ``client`` injection lets unit tests run without credentials or network.
    SHA-256 must come from a trusted source (e.g. ``upload_file``'s return value).
    """

    def __init__(self, project: str, bucket: str, *, client=None, upload_timeout: float = 600):
        if not math.isfinite(upload_timeout) or upload_timeout <= 0:
            raise ValueError("Upload timeout must be finite and positive.")
        self.upload_timeout = upload_timeout
        if not project.strip() or not bucket.strip():
            raise ValueError("Project and bucket names must be nonempty.")
        self.bucket_name = bucket
        try:
            self.client = client if client is not None else storage.Client(project=project)
            self.bucket = self.client.bucket(bucket)
        except (GoogleAuthError, GoogleAPIError) as exc:
            raise GCSOperationError(
                "Could not initialize GCS using ADC. Check local ADC configuration."
            ) from exc

    def _blob(self, object_name: str):
        if not object_name.strip():
            raise ValueError("Object name must be nonempty.")
        return self.bucket.blob(object_name)

    def upload_file(
        self, local_path: str | Path, object_name: str, *, metadata: dict[str, str] | None = None,
    ) -> str:
        """Create an object without overwriting; return the local SHA-256 digest.

        Automatic SDK checksum cleanup is disabled because it may delete a
        corrupt object. Downloads explicitly verify SHA-256 instead. If upload
        status is uncertain, inspect the object before attempting another run.
        """
        blob = self._blob(object_name)
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"Upload source is not a file: {path}")
        try:
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                stream.seek(0)
                if metadata is not None:
                    # Attach provenance during creation, without a later mutation.
                    # Always record the measured checksum, not a caller-supplied one.
                    blob.metadata = {**metadata, "sha256": digest.hexdigest()}
                blob.upload_from_file(
                    stream, if_generation_match=0, checksum=None,
                    timeout=self.upload_timeout, retry=None,
                )
            return digest.hexdigest()
        except (GoogleAPIError, GoogleAuthError, RequestException, OSError) as exc:
            raise GCSOperationError(
                f"Upload failed for gs://{self.bucket_name}/{object_name} "
                f"({type(exc).__name__}: {exc}); existing objects are not overwritten."
            ) from exc

    def object_metadata(self, object_name: str) -> dict[str, str]:
        """Read custom metadata through ADC; this does not modify the object."""
        blob = self._blob(object_name)
        try:
            blob.reload(timeout=60, retry=None)
            return dict(blob.metadata or {})
        except (GoogleAPIError, GoogleAuthError, RequestException) as exc:
            raise GCSOperationError(
                f"Metadata read failed for gs://{self.bucket_name}/{object_name} "
                f"({type(exc).__name__})."
            ) from exc

    def object_info(self, object_name: str) -> dict:
        """Read the current generation, byte size, and custom provenance."""
        blob = self._blob(object_name)
        try:
            blob.reload(timeout=60, retry=None)
            return {"generation": str(blob.generation), "size_bytes": blob.size,
                    "metadata": dict(blob.metadata or {})}
        except (GoogleAPIError, GoogleAuthError, RequestException) as exc:
            raise GCSOperationError(
                f"Object inspection failed for gs://{self.bucket_name}/{object_name} "
                f"({type(exc).__name__})."
            ) from exc

    def object_exists(self, object_name: str) -> bool:
        """Return false for missing objects; permissions/network errors raise."""
        blob = self._blob(object_name)
        try:
            return blob.exists(timeout=60, retry=None)
        except (GoogleAPIError, GoogleAuthError, RequestException) as exc:
            raise GCSOperationError(
                f"Existence check failed for gs://{self.bucket_name}/{object_name} "
                f"({type(exc).__name__})."
            ) from exc

    def download_file(
        self, object_name: str, local_path: str | Path, *, expected_sha256: str,
        generation: int | None = None,
    ) -> Path:
        """Download to a new file and check SHA-256; retain failures for inspection.

        The parent directory must exist. Existing destinations are never
        overwritten. ``download_to_file`` avoids SDK filename cleanup/deletion.
        """
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters.")
        blob = self._blob(object_name)
        conditions = {}
        if generation is not None:
            if generation <= 0:
                raise ValueError("GCS generation must be positive.")
            blob = self.bucket.blob(object_name, generation=generation)
            conditions["if_generation_match"] = generation
        path = Path(local_path)
        try:
            with path.open("xb") as stream:
                blob.download_to_file(stream, checksum=None, timeout=60, retry=None, **conditions)
            actual = sha256_file(path)
        except FileExistsError:
            raise
        except (GoogleAPIError, GoogleAuthError, RequestException, OSError) as exc:
            raise GCSOperationError(
                f"Download failed for gs://{self.bucket_name}/{object_name} "
                f"({type(exc).__name__}); any partial file is retained at {path}."
            ) from exc
        if actual != expected_sha256.lower():
            raise ChecksumMismatchError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256.lower()}, "
                f"got {actual}. File retained for inspection."
            )
        return path
