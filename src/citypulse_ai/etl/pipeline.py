"""Persist a single CSV's aggregates and validation audit for SQL analysis."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import duckdb
import pandas as pd
import pyarrow
import pytz

from citypulse_ai.etl.schema import TIMESTAMP_POLICY
from citypulse_ai.etl.trips import aggregate_hourly, read_trips_csv, validate_trips


def run_pipeline(input_csv: str | Path, output_dir: str | Path) -> dict:
    """Replace this run's four named outputs; never append counts across reruns."""
    source = Path(input_csv)
    destination = Path(output_dir)
    outputs = [destination / name for name in (
        "hourly_rides.parquet", "citypulse.duckdb", "rejected_trips.csv", "report.json"
    )]
    if source.resolve() in [path.resolve() for path in outputs]:
        raise ValueError("Input CSV must not be one of the output files")
    # Hash before and after ingestion so the report identifies the bytes read.
    def fingerprint() -> str:
        with source.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    digest = fingerprint()
    raw = read_trips_csv(source)
    if fingerprint() != digest:
        raise ValueError("Input CSV changed during ingestion")
    result = validate_trips(raw)
    hourly = aggregate_hourly(result.valid)
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path, database_path, rejected_path, report_path = outputs
    hourly.to_parquet(parquet_path, engine="pyarrow", index=False)
    result.rejected.to_csv(rejected_path, index=False)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(
            "CREATE OR REPLACE TABLE hourly_rides AS SELECT * FROM read_parquet(?)",
            [str(parquet_path.resolve())],
        )
    reason_counts = Counter(
        reason
        for reasons in result.rejected.rejection_reasons
        for reason in reasons.split(";")
    )
    report = {
        "input_file": str(source.resolve()),
        "input_sha256": digest,
        "input_rows": len(raw),
        "accepted_rows": len(result.valid),
        "rejected_rows": len(result.rejected),
        "hourly_rows": len(hourly),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "timestamp_policy": TIMESTAMP_POLICY,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "duckdb": duckdb.__version__,
            "pytz": pytz.__version__,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
