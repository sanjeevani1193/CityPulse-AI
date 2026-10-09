"""Chunked trip ETL with disk-backed duplicate checks and hourly aggregation."""

from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory
from zoneinfo import TZPATH

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz

from citypulse_ai.etl.schema import (
    LOCAL_TIMEZONE, RAW_COLUMNS, REQUIRED_VALUES, REVIEWED_STATION_IDS, STATION_ID_PATTERN,
    STATION_NULL_MARKERS, TIMESTAMP_POLICY,
)
from citypulse_ai.etl.trips import aggregate_hourly, iter_trip_chunks, read_trips_csv, validate_trips
from citypulse_ai.features.reporting import prepare_february_2025

AUDIT_COLUMNS = ["source_file", "source_row", *RAW_COLUMNS, "rejection_reasons"]
HOURLY_SCHEMA = pa.schema([
    ("start_station_id", pa.string()),
    ("hour_start", pa.timestamp("ns", tz="UTC")),
    ("ride_count", pa.int64()),
])
TRIP_SCHEMA = pa.schema([
    ("source_file", pa.string()), ("source_row", pa.int64()),
    ("ride_id", pa.string()), ("started_at", pa.timestamp("ns", tz="UTC")),
    ("ended_at", pa.timestamp("ns", tz="UTC")), ("start_station_id", pa.string()),
    ("end_station_id", pa.string()), ("duration_seconds", pa.float64()),
    # DuckDB TIMESTAMPTZ has microsecond precision; retain exact instants as well.
    ("started_at_utc_ns", pa.int64()), ("ended_at_utc_ns", pa.int64()),
])


def _fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _timezone_data() -> dict:
    """Record the system zoneinfo bytes used by the documented local-time policy."""
    for root in TZPATH:
        path = Path(root) / LOCAL_TIMEZONE
        if path.is_file():
            return {"source": "system zoneinfo", "sha256": _fingerprint(path)}
    from importlib.resources import files
    import tzdata
    data = files("tzdata.zoneinfo").joinpath(*LOCAL_TIMEZONE.split("/")).read_bytes()
    return {"source": "tzdata", "version": tzdata.__version__,
            "sha256": hashlib.sha256(data).hexdigest()}


def _stage_inputs(connection, sources, chunk_size, max_rows_per_file):
    manifests = []
    connection.execute("""
        CREATE TABLE raw_trips (
            source_file VARCHAR, source_row BIGINT, ride_id VARCHAR,
            started_at VARCHAR, ended_at VARCHAR, start_station_id VARCHAR,
            end_station_id VARCHAR, normalized_ride_id VARCHAR
        )
    """)
    for source in sources:
        digest = _fingerprint(source)
        rows = 0
        for chunk in iter_trip_chunks(source, chunk_size, max_rows_per_file):
            if chunk.empty:
                continue
            chunk.insert(0, "source_row", chunk.index + 2)
            chunk.insert(0, "source_file", str(source))
            chunk["normalized_ride_id"] = chunk.ride_id.str.strip()
            connection.register("incoming", chunk)
            connection.execute("INSERT INTO raw_trips SELECT * FROM incoming")
            connection.unregister("incoming")
            rows += len(chunk)
        if _fingerprint(source) != digest:
            raise ValueError(f"Input CSV changed during ingestion: {source}")
        manifests.append({"input_file": str(source), "input_sha256": digest, "input_rows": rows})
    connection.execute("""
        CREATE TABLE duplicate_ids AS
        SELECT normalized_ride_id FROM raw_trips
        WHERE normalized_ride_id <> ''
        GROUP BY normalized_ride_id HAVING COUNT(*) > 1
    """)
    return manifests


def _validate_batches(connection, audit_path, chunk_size, trips_path):
    """Validate bounded batches using duplicate flags computed over the whole run."""
    connection.execute("""
        CREATE TABLE hourly_parts (
            start_station_id VARCHAR, hour_start TIMESTAMPTZ, ride_count BIGINT
        )
    """)
    pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    # A separate cursor keeps the raw result stream intact while inserting counts.
    reader = connection.cursor()
    reader.execute("""
        SELECT r.*, d.normalized_ride_id IS NOT NULL AS duplicate_ride_id
        FROM raw_trips r LEFT JOIN duplicate_ids d USING (normalized_ride_id)
        ORDER BY source_file, source_row
    """)
    columns = [item[0] for item in reader.description]
    accepted = rejected = missing_ends = 0
    reasons = Counter()
    trip_writer = pq.ParquetWriter(trips_path, TRIP_SCHEMA)
    try:
        while records := reader.fetchmany(chunk_size):
            batch = pd.DataFrame.from_records(records, columns=columns)
            result = validate_trips(batch, duplicate_mask=batch.duplicate_ride_id)
            accepted += len(result.valid)
            rejected += len(result.rejected)
            missing_ends += int(result.valid.end_station_id.eq("").sum())
            if not result.valid.empty:
                rejected_positions = result.rejected.source_row.to_numpy() - 2
                valid_positions = batch.index.difference(rejected_positions)
                valid = result.valid.copy()
                valid["source_file"] = batch.source_file.iloc[valid_positions].to_numpy()
                valid["source_row"] = batch.source_row.iloc[valid_positions].to_numpy()
                valid["started_at_utc_ns"] = valid.started_at.array.asi8
                valid["ended_at_utc_ns"] = valid.ended_at.array.asi8
                trip_writer.write_table(pa.Table.from_pandas(
                    valid[TRIP_SCHEMA.names], schema=TRIP_SCHEMA, preserve_index=False,
                ))
            if not result.rejected.empty:
                positions = result.rejected.source_row.to_numpy() - 2
                result.rejected["source_row"] = batch.source_row.iloc[positions].to_numpy()
                result.rejected.insert(0, "source_file", batch.source_file.iloc[positions].to_numpy())
                result.rejected[AUDIT_COLUMNS].to_csv(audit_path, mode="a", header=False, index=False)
                reasons.update(reason for entry in result.rejected.rejection_reasons
                               for reason in entry.split(";"))
            hourly = aggregate_hourly(result.valid)
            if not hourly.empty:
                connection.register("incoming_hourly", hourly)
                connection.execute("INSERT INTO hourly_parts SELECT * FROM incoming_hourly")
                connection.unregister("incoming_hourly")
    finally:
        reader.close()
        trip_writer.close()
    return accepted, rejected, missing_ends, dict(sorted(reasons.items()))


def _write_hourly_parquet(connection, path, chunk_size):
    connection.execute("""
        CREATE TABLE hourly_rides AS
        SELECT start_station_id, hour_start, CAST(SUM(ride_count) AS BIGINT) AS ride_count
        FROM hourly_parts GROUP BY start_station_id, hour_start
    """)
    result = connection.execute(
        "SELECT * FROM hourly_rides ORDER BY start_station_id, hour_start"
    )
    batches = (
        result.to_arrow_reader(batch_size=chunk_size) if hasattr(result, "to_arrow_reader")
        else result.fetch_record_batch(rows_per_batch=chunk_size)
    )
    # Explicit schema preserves UTC nanoseconds, including for empty outputs.
    with pq.ParquetWriter(path, HOURLY_SCHEMA) as writer:
        for batch in batches:
            writer.write_batch(batch.cast(HOURLY_SCHEMA))


def run_pipeline(
    input_csv: str | Path | Sequence[str | Path], output_dir: str | Path, *,
    chunk_size: int = 50_000, max_rows_per_file: int | None = None,
    memory_limit: str = "512MB",
) -> dict:
    """Replace run outputs; duplicates cover every selected file/row in this run.

    Raw rows and duplicate IDs live in temporary DuckDB storage, not Python sets.
    Sampling limits validation/duplicate scope, while hashes cover full file bytes.
    """
    if chunk_size <= 0 or (max_rows_per_file is not None and max_rows_per_file <= 0):
        raise ValueError("Chunk size and optional row limit must be positive")
    paths = [input_csv] if isinstance(input_csv, (str, Path)) else list(input_csv)
    if not paths:
        raise ValueError("At least one input CSV is required")
    sources = sorted(Path(path).resolve() for path in paths)
    if len(set(sources)) != len(sources):
        raise ValueError("The same input file was supplied more than once")
    destination = Path(output_dir)
    names = ["hourly_rides.parquet", "validated_trips.parquet", "citypulse.duckdb",
             "february_2025_hourly_rides.parquet", "rejected_trips.csv", "report.json"]
    if any(source in [(destination / name).resolve() for name in names] for source in sources):
        raise ValueError("Input CSV must not be one of the output files")
    for source in sources:
        read_trips_csv(source, nrows=0)

    with TemporaryDirectory(prefix="citypulse-") as temporary:
        work = Path(temporary)
        with duckdb.connect(str(work / "staging.duckdb"), config={
            "memory_limit": memory_limit, "threads": 2, "temp_directory": str(work / "spill"),
        }) as connection:
            connection.execute("SET TimeZone = 'UTC'")
            manifests = _stage_inputs(connection, sources, chunk_size, max_rows_per_file)
            accepted, rejected, missing_ends, reasons = _validate_batches(
                connection, work / "rejected_trips.csv", chunk_size, work / "validated_trips.parquet",
            )
            _write_hourly_parquet(connection, work / "hourly_rides.parquet", chunk_size)
            report = {
                "inputs": manifests,
                "input_rows": sum(item["input_rows"] for item in manifests),
                "accepted_rows": accepted, "rejected_rows": rejected,
                "accepted_missing_end_station_rows": missing_ends,
                "hourly_rows": connection.execute("SELECT COUNT(*) FROM hourly_rides").fetchone()[0],
                "duplicate_ride_ids": connection.execute("SELECT COUNT(*) FROM duplicate_ids").fetchone()[0],
                "rejection_reason_counts": reasons,
                "timestamp_policy": TIMESTAMP_POLICY, "timezone_data": _timezone_data(),
                "validation_policy": {
                    "required_values": list(REQUIRED_VALUES),
                    "missing_end_station_id": "accepted",
                    "start_station_id_pattern": STATION_ID_PATTERN,
                    "start_station_null_markers": sorted(STATION_NULL_MARKERS),
                    "reviewed_station_id_exceptions": sorted(REVIEWED_STATION_IDS),
                    "station_id_mapping": "none; preserve distinct identifiers",
                    "duration": "strictly positive, representable elapsed UTC duration; no upper cutoff",
                    "duplicates": "reject all occurrences across selected rows/files",
                },
                "processing": {"chunk_size": chunk_size, "max_rows_per_file": max_rows_per_file,
                               "duckdb_memory_limit": memory_limit},
                "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "pyarrow": pa.__version__, "duckdb": duckdb.__version__,
                             "pytz": pytz.__version__},
            }
            if len(manifests) == 1:
                report.update({key: manifests[0][key] for key in ("input_file", "input_sha256")})
        for item in manifests:
            if _fingerprint(Path(item["input_file"])) != item["input_sha256"]:
                raise ValueError(f"Input CSV changed during processing: {item['input_file']}")
        destination.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(destination / "citypulse.duckdb")) as connection:
            connection.execute("SET TimeZone = 'UTC'")
            connection.execute(
                "CREATE OR REPLACE TABLE hourly_rides AS SELECT * FROM read_parquet(?)",
                [str(work / "hourly_rides.parquet")],
            )
            connection.execute(
                "CREATE OR REPLACE TABLE validated_trips AS SELECT * FROM read_parquet(?)",
                [str(work / "validated_trips.parquet")],
            )
            report["february_2025"] = prepare_february_2025(
                connection, work / "february_2025_hourly_rides.parquet",
            )
        (work / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for name in ("hourly_rides.parquet", "validated_trips.parquet", "rejected_trips.csv",
                     "february_2025_hourly_rides.parquet", "report.json"):
            shutil.move(str(work / name), str(destination / name))
    return report
