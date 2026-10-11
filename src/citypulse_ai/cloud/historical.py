"""Optional cloud publication of immutable, successfully processed source months.

Call under historical_cli.single_writer (or equivalent serialization). No source
partition, ETL report, raw file, or existing cloud object is rewritten or deleted.
"""

import hashlib
import json
import math
from pathlib import Path

from citypulse_ai.cloud.gcs import GCSStorage
from citypulse_ai.cloud.publish import MAX_ARTIFACT_BYTES, publish_verified_artifact
from citypulse_ai.etl.archives import validate_month, write_json
from citypulse_ai.etl.historical import load_manifest, now, verified_partition


def publish_month(
    month: str, output_root: Path, *, project: str, bucket: str,
    prefix: str = "curated/historical", max_file_bytes: int = MAX_ARTIFACT_BYTES,
    store: GCSStorage | None = None,
    upload_timeout: float = 600, upload_attempts: int = 3,
) -> dict:
    """Resume per-file verification; mark published only when every file verifies.

    Local ETL status stays 'processed'. Cloud failure/interruption is persisted
    separately and does not trigger reprocessing. Hard kills leave 'publishing'
    checkpoints which are recovered by the same command. Verified files are
    rechecked on retry so a changed/deleted remote object is never silently used.
    """
    validate_month(month)
    if (not math.isfinite(upload_timeout) or upload_timeout <= 0
            or not 1 <= upload_attempts <= 5):
        raise ValueError("Upload timeout must be finite/positive and attempts between 1 and 5.")
    root = Path(output_root).resolve()
    manifest = load_manifest(root)
    entry = manifest["archives"].get(month)
    if entry is None or not verified_partition(entry):
        raise ValueError("Cloud publication requires an unchanged, processed full source partition.")
    if not project.strip() or not bucket.strip() or not prefix.strip("/"):
        raise ValueError("Cloud project, bucket, and prefix must be nonempty.")
    if max_file_bytes <= 0:
        raise ValueError("Cloud maximum file size must be positive.")
    folder = Path(entry["output_dir"])
    hashes = {name: digest for name, digest in sorted(entry["output_sha256"].items())
              if Path(name).suffix in {".parquet", ".json"} or name == "rejected_trips.csv"}
    required = {"validated_trips.parquet", "hourly_rides.parquet", "rejected_trips.csv", "report.json"}
    if not required.issubset(hashes):
        raise ValueError("Partition lacks required validated Parquet or audit reports.")
    for name in hashes:
        if Path(name).name != name:
            raise ValueError("Publication artifacts must be direct source partition files.")
        if (folder / name).stat().st_size >= max_file_bytes:
            raise ValueError(f"{name} exceeds the explicit cloud size limit; nothing uploaded.")
    start_counts = entry.get("verification", {}).get("start_month_counts")
    if not isinstance(start_counts, dict) or sum(start_counts.values()) != entry["accepted_rows"]:
        raise ValueError("Reviewed NYC local start-month counts must reconcile with accepted rows.")
    specification = {"source_archive_month": month, "project": project, "bucket": bucket,
                     "prefix": prefix.strip("/"), "artifact_sha256": hashes,
                     "source": entry["source"], "local_start_month_counts": start_counts}
    revision = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    publication = entry.get("cloud_publication")
    if publication and publication.get("revision") != revision:
        raise ValueError("Publication source or destination changed; preserve the existing publication and investigate.")
    if publication is None:
        publication = {**specification, "revision": revision, "status": "pending",
                       "created_at_utc": now(), "files": {}}
        entry["cloud_publication"] = publication
    publication["status"] = "publishing"
    publication.pop("error", None)
    publication.pop("published_at_utc", None)
    write_json(root / "manifest.json", manifest)  # Persist identity before any cloud request.
    current = None
    try:
        cloud = store if store is not None else GCSStorage(project, bucket, upload_timeout=upload_timeout)
        if cloud.bucket_name != bucket:
            raise ValueError("Cloud client bucket differs from configured destination.")
        for name, digest in hashes.items():
            object_name = f"{specification['prefix']}/source_archive_month={month}/revision={revision}/{name}"
            current = publication["files"].setdefault(name, {})
            prior_generation = current.get("generation")
            current.update({"cloud_path": f"gs://{bucket}/{object_name}", "sha256": digest,
                            "verification_status": "publishing"})
            current.pop("error", None)
            write_json(root / "manifest.json", manifest)
            metadata = {"source_archive_month": month, "publication_revision": revision,
                        "sha256": digest, "source_filename": name,
                        "source_archive_url": entry["source"]["url"],
                        "source_archive_sha256_status": "available" if entry["source"].get("sha256") else "unavailable",
                        "local_start_month_counts": json.dumps(start_counts, sort_keys=True),
                        "coverage_status": "source_archive_only_not_complete_start_months"}
            if entry["source"].get("sha256"):
                metadata["source_archive_sha256"] = entry["source"]["sha256"]
            def checkpoint(info):
                # Record the inspected generation before an interruptible download.
                current.update(info)
                write_json(root / "manifest.json", manifest)
            receipt = publish_verified_artifact(folder / name, object_name, store=cloud,
                                                metadata=metadata, max_bytes=max_file_bytes,
                                                expected_generation=prior_generation,
                                                on_inspected=checkpoint, upload_attempts=upload_attempts)
            current.update(receipt)
            write_json(root / "manifest.json", manifest)
        publication["status"] = "published"
        publication["published_at_utc"] = now()
        write_json(root / "manifest.json", manifest)
        return publication
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        publication.update({"status": status, "error": f"{type(exc).__name__}: {exc}"})
        if current is not None:
            current.update({"verification_status": status, "error": publication["error"]})
        write_json(root / "manifest.json", manifest)
        raise
