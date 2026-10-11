"""Publish one existing Parquet under 100 MiB, then verify a retained download."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from citypulse_ai.cloud.gcs import GCSOperationError, GCSStorage, sha256_file

MAX_ARTIFACT_BYTES = 100 * 1024**2


def parse_destination(destination: str) -> tuple[str, str]:
    """Require an exact gs://bucket/object name, never a directory destination."""
    parts = urlsplit(destination)
    name = parts.path[1:]
    if (parts.scheme != "gs" or not parts.netloc or not name.strip()
            or name.endswith("/") or parts.query or parts.fragment
            or "@" in parts.netloc or ":" in parts.netloc):
        raise ValueError("Destination must be gs://bucket/exact-object-name without query or fragment.")
    return parts.netloc, name


def parquet_details(path: Path) -> dict:
    """Read the footer, not the full dataset; hash the exact Arrow schema."""
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        return {
            "row_count": parquet.metadata.num_rows,
            "schema": str(schema),
            "schema_sha256": hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
        }


def verify_parquet(original: Path, downloaded: Path) -> dict:
    """Compare bytes, row counts, and schema including embedded metadata."""
    checksum_match = sha256_file(original) == sha256_file(downloaded)
    with pq.ParquetFile(original) as source, pq.ParquetFile(downloaded) as copy:
        row_count_match = source.metadata.num_rows == copy.metadata.num_rows
        schema_match = source.schema_arrow.equals(copy.schema_arrow, check_metadata=True)
    checks = {"sha256_match": checksum_match, "row_count_match": row_count_match,
              "schema_match": schema_match}
    if not all(checks.values()):
        raise GCSOperationError(f"Parquet verification failed: {checks}; download retained at {downloaded}.")
    return checks


def publish_parquet(
    local_path: str | Path, destination: str, *, project: str,
    report_path: str | Path, store: GCSStorage | None = None,
) -> dict:
    """Create one object and a new receipt; retain both on verification failure.

    Reports are opened exclusively before cloud operations. No ACL or IAM policy
    is changed: this command uses the destination bucket's existing permissions.
    An injected store is solely for testing; its bucket must match the destination.
    """
    bucket, object_name = parse_destination(destination)
    source = Path(local_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Artifact is not a file: {source}")
    if source.stat().st_size >= MAX_ARTIFACT_BYTES:
        raise ValueError("Artifact must be under 100 MiB; nothing was uploaded.")
    details = parquet_details(source)  # Invalid Parquet fails before authentication.
    digest = sha256_file(source)
    try:
        source_name = source.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_name = str(source)
    metadata = {
        "source_path": source_name, "source_filename": source.name,
        "sha256": digest, "size_bytes": str(source.stat().st_size),
        "row_count": str(details["row_count"]), "schema_sha256": details["schema_sha256"],
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Link existing validation evidence without uploading it or inventing a policy.
    for name in ("report.json", "verification.json"):
        evidence = source.parent / name
        if evidence.is_file():
            metadata[name.removesuffix(".json") + "_sha256"] = sha256_file(evidence)
    receipt = Path(report_path)
    result = {"project": project, "destination": destination, "source_path": source_name,
              "size_bytes": source.stat().st_size, "sha256": digest, **details,
              "object_metadata": metadata, "uploaded": False, "verified": False}
    # Existing receipt paths must never be overwritten, including after failures.
    with receipt.open("x", encoding="utf-8") as report:
        try:
            cloud = store if store is not None else GCSStorage(project, bucket)
            if cloud.bucket_name != bucket:
                raise ValueError("Client bucket does not match destination bucket.")
            uploaded_hash = cloud.upload_file(source, object_name, metadata=metadata)
            result["uploaded"] = True
            if uploaded_hash != digest:
                raise GCSOperationError("Source changed during upload; object retained, not verified.")
            result["object_exists"] = cloud.object_exists(object_name)
            remote = cloud.object_metadata(object_name)
            result["metadata_match"] = all(remote.get(k) == v for k, v in metadata.items())
            if not result["object_exists"] or not result["metadata_match"]:
                raise GCSOperationError("Uploaded object existence/provenance verification failed.")
            work = Path(tempfile.mkdtemp(prefix="citypulse-gcs-parquet-"))
            downloaded = work / source.name
            result["download_path"] = str(downloaded)
            cloud.download_file(object_name, downloaded, expected_sha256=digest)
            result["checks"] = verify_parquet(source, downloaded)
            result["verified"] = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            report.write(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", help="Exact gs://bucket/object destination")
    parser.add_argument("--project", required=True)
    parser.add_argument("--report", required=True, type=Path, help="New local JSON receipt; parent must exist")
    args = parser.parse_args(argv)
    try:
        result = publish_parquet(args.artifact, args.destination, project=args.project, report_path=args.report)
    except (GCSOperationError, OSError, ValueError, pa.ArrowException) as exc:
        print(f"Publication failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
