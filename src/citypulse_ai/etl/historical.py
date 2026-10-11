"""Monthly orchestration around the unchanged Phase 1 chunked ETL."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

from citypulse_ai.etl.archives import (
    OFFICIAL_BUCKET, check_disk, discover_archive, download_archive, extract_archive, fingerprint,
    inspect_archive, validate_month, write_json,
)
from citypulse_ai.etl.pipeline import run_pipeline
from citypulse_ai.etl.trips import read_trips_csv

RESERVE_BYTES = 5 * 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "archives": {}, "modelling_specification": {
        "timezone": "America/New_York", "start_local": "2024-01-01",
        "end_local_exclusive": "2026-01-01", "train_end_local_exclusive": "2025-07-01",
        "validation_end_local_exclusive": "2025-10-01", "forecast_elapsed_hours": 24,
        "station_cohort_size": 100, "station_selection": "training history only; not implemented here",
    }}


def artifact_hashes(directory: Path) -> dict:
    return {p.name: fingerprint(p) for p in directory.iterdir() if p.is_file()}


def verified_partition(entry: dict) -> bool:
    if entry.get("status") != "processed":
        return False
    directory = Path(entry["output_dir"])
    return bool(entry.get("output_sha256")) and all(
        (directory / name).is_file() and fingerprint(directory / name) == digest
        for name, digest in entry["output_sha256"].items()
    )


def plan_months(months: list[str], raw_root: Path, output_root: Path,
                reserve_bytes: int = RESERVE_BYTES) -> dict:
    plans = []
    for month in months:
        source = discover_archive(month)
        layout = inspect_archive(source)
        plans.append({**source, **layout})
    # Deliberately conservative: ZIP + CSV + outputs + a disk-backed staging/spill allowance.
    retained = sum(p["bytes"] + 3 * p["csv_bytes"] for p in plans)
    staging = 6 * max(p["csv_bytes"] for p in plans)
    def filesystem(path):
        parent = path.resolve()
        while not parent.exists():
            parent = parent.parent
        return {"existing_parent": str(parent), "device": parent.stat().st_dev,
                "free_bytes": shutil.disk_usage(parent).free}
    raw_disk, output_disk = filesystem(raw_root), filesystem(output_root)
    free = output_disk["free_bytes"]
    same_volume = raw_disk["device"] == output_disk["device"]
    raw_estimate = sum(p["bytes"] + p["csv_bytes"] for p in plans)
    output_estimate = 2 * sum(p["csv_bytes"] for p in plans) + staging
    fits = (free >= retained + staging + reserve_bytes if same_volume else
            raw_disk["free_bytes"] >= raw_estimate + reserve_bytes and
            free >= output_estimate + reserve_bytes)
    return {"archives": plans, "archive_bytes": sum(p["bytes"] for p in plans),
            "csv_bytes": sum(p["csv_bytes"] for p in plans),
            "estimated_retained_and_staging_bytes": retained + staging,
            "disk_free_bytes": free, "reserve_bytes": reserve_bytes,
            "fits_current_filesystem": fits,
            "raw_filesystem": raw_disk, "output_filesystem": output_disk,
            "same_filesystem": same_volume,
            "note": "Conservative planning estimate, not measured ETL peak; checks the configured raw/output volumes."}


def verify_run(directory: Path, report: dict) -> dict:
    """Independent SQL reconciliation of Phase 1 outputs before publication."""
    import duckdb
    with duckdb.connect(str(directory / "citypulse.duckdb"), read_only=True) as connection:
        input_rows = report["input_rows"]
        accepted = connection.execute("SELECT count(*) FROM validated_trips").fetchone()[0]
        hourly_rows, starts, unique_keys = connection.execute("""
            SELECT count(*), coalesce(sum(ride_count),0),
                   count(DISTINCT (start_station_id,hour_start)) FROM hourly_rides
        """).fetchone()
        precision_errors = connection.execute("""
            SELECT count(*) FROM validated_trips
            WHERE epoch_us(started_at) <> started_at_utc_ns // 1000
               OR epoch_us(ended_at) <> ended_at_utc_ns // 1000
        """).fetchone()[0]
        months = connection.execute("""
            SELECT strftime(started_at AT TIME ZONE 'America/New_York','%Y-%m') AS start_month,
                   count(*) AS ride_starts FROM validated_trips GROUP BY start_month ORDER BY start_month
        """).fetchall()
    if input_rows != accepted + report["rejected_rows"] or accepted != report["accepted_rows"]:
        raise ValueError("Input/accepted/rejected reconciliation failed")
    if starts != accepted or hourly_rows != unique_keys or precision_errors:
        raise ValueError("Hourly count, key uniqueness, or UTC precision verification failed")
    return {"input_equals_accepted_plus_rejected": True, "ride_counts_equal_accepted": True,
            "duplicate_station_hour_keys": 0, "timestamp_epoch_mismatches": precision_errors,
            "start_month_counts": dict(months)}


def process_month(month: str, raw_root: Path, output_root: Path, *,
                  sample_rows: int | None = None, chunk_size: int = 50_000,
                  memory_limit: str = "512MB", reserve_bytes: int = RESERVE_BYTES) -> dict:
    """Resume raw steps; restart interrupted ETL atomically; never mix samples/full runs."""
    validate_month(month)
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("Sample rows must be positive")
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_root)
    entry = manifest["archives"].setdefault(month, {"month": month, "attempts": []})
    if verified_partition(entry):
        return {"month": month, "status": "skipped_verified", "output_dir": entry["output_dir"]}
    elif entry.get("status") == "processed":
        raise ValueError("Processed partition changed; preserve it and investigate before retrying")
    destination = output_root / (f"samples/{month}-n{sample_rows}" if sample_rows else f"source_month={month}")
    if destination.exists():
        # A process may stop after the atomic directory rename but before manifest publication.
        previous = entry["attempts"][-1] if entry["attempts"] else {}
        if previous.get("output_dir") != str(destination) or previous.get("sample_rows_per_file") != sample_rows:
            raise ValueError(f"Refusing to overwrite unrelated existing output: {destination}")
        report = json.loads((destination / "report.json").read_text())
        if report["processing"]["max_rows_per_file"] != sample_rows:
            raise ValueError("Existing run has a different sample scope")
        verification = verify_run(destination, report)
        hashes = artifact_hashes(destination)
        if previous.get("output_sha256") and previous["output_sha256"] != hashes:
            raise ValueError("Existing sample output changed")
        previous.update({"status": "sample_processed" if sample_rows else "processed",
                         "completed_at": now(), "output_sha256": hashes})
        if sample_rows is None:
            entry.update({"status": "processed", "output_dir": str(destination),
                          "output_sha256": hashes, "verification": verification,
                          "input_rows": report["input_rows"], "accepted_rows": report["accepted_rows"],
                          "rejected_rows": report["rejected_rows"]})
        write_json(output_root / "manifest.json", manifest)
        return {"month": month, "status": "resumed_verified", "output_dir": str(destination)}
    attempt = {"started_at": now(), "sample_rows_per_file": sample_rows, "status": "discovering",
               "output_dir": str(destination)}
    entry["attempts"].append(attempt)
    previous_status = entry.get("status")
    if previous_status != "processed":
        entry["status"] = "running"
    write_json(output_root / "manifest.json", manifest)
    started = perf_counter()
    free_before = shutil.disk_usage(output_root).free
    try:
        source = discover_archive(month)
        layout = inspect_archive(source)
        entry["source"] = {**source, **layout}
        # 6x CSV reserves staging, spill and output; memory limit is for DuckDB, not total RSS.
        raw_disk = check_disk(raw_root, source["bytes"] + layout["csv_bytes"], reserve_bytes)
        output_disk = check_disk(output_root, source["bytes"] + 7 * layout["csv_bytes"], reserve_bytes)
        entry["disk_preflight"] = {"raw": raw_disk, "output": output_disk}
        attempt["status"] = "downloading"
        write_json(output_root / "manifest.json", manifest)
        archive, digest = download_archive(source, raw_root / "archives")
        entry["source"]["sha256"] = digest
        attempt["status"] = "extracting"
        write_json(output_root / "manifest.json", manifest)
        files = extract_archive(archive, raw_root / f"source_month={month}")
        schemas = []
        for item in files:
            path = Path(item["path"])
            with path.open(newline="", encoding="utf-8-sig") as stream:
                columns = next(csv.reader(stream))
            preview = read_trips_csv(path, nrows=5)
            schemas.append({"member": item["member"], "columns": columns,
                            "started_at_examples": preview.started_at.tolist()})
        entry["csv_files"], entry["schemas"] = files, schemas
        attempt["status"] = "processing_sample" if sample_rows else "processing_full"
        write_json(output_root / "manifest.json", manifest)
        work_root = output_root / ".work"
        work_root.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=f"{month}-", dir=work_root) as work:
            run = Path(work) / "run"
            report = run_pipeline([item["path"] for item in files], run, chunk_size=chunk_size,
                                  max_rows_per_file=sample_rows, memory_limit=memory_limit)
            verification = verify_run(run, report)
            write_json(run / "historical_verification.json", verification)
            elapsed = perf_counter() - started
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics = {"elapsed_seconds_including_download": elapsed,
                       "process_peak_rss_bytes": max_rss if platform.system() == "Darwin" else max_rss * 1024,
                       "disk_free_before_bytes": free_before,
                       "disk_free_before_publication_bytes": shutil.disk_usage(output_root).free,
                       "output_file_bytes": sum(p.stat().st_size for p in run.iterdir() if p.is_file()),
                       "duckdb_memory_limit": memory_limit,
                       "note": "RSS is process lifetime high-water, not just DuckDB; disk free includes unrelated activity."}
            write_json(run / "historical_metrics.json", metrics)
            destination.parent.mkdir(parents=True, exist_ok=True)
            run.rename(destination)
        attempt.update({"status": "sample_processed" if sample_rows else "processed",
                        "completed_at": now(), "metrics": metrics, "verification": verification,
                        "output_dir": str(destination), "input_rows": report["input_rows"],
                        "accepted_rows": report["accepted_rows"], "rejected_rows": report["rejected_rows"]})
        attempt["output_sha256"] = artifact_hashes(destination)
        if not sample_rows:
            entry.update({"status": "processed", "output_dir": str(destination),
                          "output_sha256": attempt["output_sha256"], "verification": verification,
                          "input_rows": report["input_rows"], "accepted_rows": report["accepted_rows"],
                          "rejected_rows": report["rejected_rows"]})
        elif previous_status != "processed":
            entry["status"] = "sample_processed"
        write_json(output_root / "manifest.json", manifest)
        return {"month": month, **attempt}
    except BaseException as error:
        attempt.update({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                        "error": f"{type(error).__name__}: {error}", "completed_at": now()})
        if previous_status != "processed":
            entry["status"] = attempt["status"]
        write_json(output_root / "manifest.json", manifest)
        raise


def register_existing(month: str, directory: Path, output_root: Path) -> dict:
    """Reference a verified full source archive run without copying or modifying it."""
    validate_month(month)
    directory, output_root = Path(directory).resolve(), Path(output_root).resolve()
    manifest = load_manifest(output_root)
    if month in manifest["archives"]:
        raise ValueError("Month already registered; refusing a second source for the same archive")
    report = json.loads((directory / "report.json").read_text())
    if report["processing"]["max_rows_per_file"] is not None:
        raise ValueError("Cannot register a sampled run as a complete archive")
    verification = verify_run(directory, report)
    entry = {"month": month, "status": "processed", "origin": "registered_existing",
             "source": {"url": OFFICIAL_BUCKET + f"{month.replace('-', '')}-citibike-tripdata.zip", "sha256": None,
                        "note": "Archive ZIP not retained in original run; CSV SHA-256 values recorded instead"},
             "output_dir": str(directory), "output_sha256": artifact_hashes(directory),
             "input_rows": report["input_rows"], "accepted_rows": report["accepted_rows"],
             "rejected_rows": report["rejected_rows"], "csv_files": report["inputs"],
             "verification": verification, "attempts": []}
    manifest["archives"][month] = entry
    write_json(output_root / "manifest.json", manifest)
    return entry
