"""Cross-archive duplicate auditing and canonical NYC ride-start-month partitions."""

from contextlib import ExitStack
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import uuid

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from citypulse_ai.etl.archives import month_range, write_json
from citypulse_ai.etl.historical import load_manifest, verified_partition

# Order and types are explicit: neither file column order nor Hive directory keys
# define the catalogue. Exact epoch integers are required to retain nanoseconds.
TRIP_TYPES = {
    "source_file": "VARCHAR", "source_row": "BIGINT", "ride_id": "VARCHAR",
    "started_at": "TIMESTAMPTZ", "ended_at": "TIMESTAMPTZ",
    "start_station_id": "VARCHAR", "end_station_id": "VARCHAR",
    "duration_seconds": "DOUBLE", "started_at_utc_ns": "BIGINT",
    "ended_at_utc_ns": "BIGINT",
}
OPTIONAL_TRIP_COLUMNS = frozenset({"end_station_id", "duration_seconds"})


def sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def aligned_trip_query(path: Path, archive_month: str) -> tuple[str, dict]:
    """Inspect footer metadata only, then project a consistent, lossless contract.

    End-station and duration metadata may be unknown. Identity, provenance, UTC
    timestamps, and exact epoch integers cannot be invented from missing columns.
    Extra physical columns remain in the immutable source and are listed in the
    schema audit; catalogue rows retain the stable trip/provenance contract.
    """
    schema = pq.read_schema(path)
    names = set(schema.names)
    if len(names) != len(schema.names):
        raise ValueError(f"Duplicate Parquet column names: {path}")
    missing = set(TRIP_TYPES) - OPTIONAL_TRIP_COLUMNS - names
    if missing:
        raise ValueError(f"Missing required catalogue columns in {path}: {', '.join(sorted(missing))}")
    for name in ("source_file", "ride_id", "start_station_id", "end_station_id"):
        if name in names:
            kind = schema.field(name).type
            if not (pa.types.is_string(kind) or pa.types.is_large_string(kind)
                    or (name in OPTIONAL_TRIP_COLUMNS and pa.types.is_null(kind))):
                raise ValueError(f"{path}: {name} must be a string; numeric IDs cannot restore lost formatting")
    for name in ("source_row", "started_at_utc_ns", "ended_at_utc_ns"):
        if not pa.types.is_integer(schema.field(name).type):
            raise ValueError(f"{path}: {name} must contain exact integers")
    for name in ("started_at", "ended_at"):
        kind = schema.field(name).type
        if not pa.types.is_timestamp(kind) or kind.tz != "UTC":
            raise ValueError(f"{path}: {name} must be a UTC-aware timestamp; do not guess its timezone")
    if "duration_seconds" in names:
        kind = schema.field("duration_seconds").type
        if not (pa.types.is_floating(kind) or pa.types.is_integer(kind) or pa.types.is_null(kind)):
            raise ValueError(f"{path}: duration_seconds must be numeric")
    projection = [f'CAST("{name}" AS {kind}) AS "{name}"' if name in names
                  else f'NULL::{kind} AS "{name}"' for name, kind in TRIP_TYPES.items()]
    projection.append(f"{sql_literal(archive_month)}::VARCHAR AS source_archive_month")
    query = ("SELECT " + ", ".join(projection) +
             f" FROM read_parquet({sql_literal(path)}, hive_partitioning=false)")
    audit = {"source_archive_month": archive_month, "physical_columns": schema.names,
             "missing_optional_columns": sorted(OPTIONAL_TRIP_COLUMNS - names),
             "extra_columns_retained_in_source_only": sorted(names - set(TRIP_TYPES)),
             "hive_partitioning": False}
    return query, audit


def rejected_csv_columns(path: Path) -> list[str]:
    """Audit CSV values can be invalid/missing; their identity columns must exist."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream), [])
    if len(set(columns)) != len(columns) or any(not name for name in columns):
        raise ValueError(f"Invalid or duplicate rejection-audit column names: {path}")
    required = {"source_file", "source_row", "ride_id", "started_at", "ended_at",
                "start_station_id", "rejection_reasons"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Missing required rejection-audit columns in {path}: {', '.join(sorted(missing))}")
    return columns


def validate_required_values(connection) -> None:
    """Run only during catalogue execution, never during metadata inspection."""
    invalid = connection.execute("""
        SELECT count(*) FROM locally_valid_trips
        WHERE source_file IS NULL OR trim(source_file) = ''
           OR source_row IS NULL OR source_row < 2
           OR ride_id IS NULL OR trim(ride_id) = ''
           OR start_station_id IS NULL OR trim(start_station_id) = ''
           OR started_at IS NULL OR ended_at IS NULL
           OR started_at_utc_ns IS NULL OR ended_at_utc_ns IS NULL
           OR epoch_us(started_at) <> started_at_utc_ns // 1000
           OR epoch_us(ended_at) <> ended_at_utc_ns // 1000
           OR ended_at_utc_ns <= started_at_utc_ns
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"{invalid} locally valid trip rows lack required values or consistent exact UTC epochs")


def write_trip_partitions(connection, destination: Path, chunk_size: int) -> None:
    """DuckDB timestamps have microseconds; reconstruct UTC nanoseconds from exact ints."""
    reader = connection.execute("SELECT * FROM canonical_trips ORDER BY start_month, source_archive_month, source_file, source_row").to_arrow_reader(batch_size=chunk_size)
    writers = {}
    with ExitStack() as stack:
        for batch in reader:
            for column in ("started_at", "ended_at"):
                position = batch.schema.get_field_index(column)
                instants = pc.cast(batch.column(column + "_utc_ns"), pa.timestamp("ns", tz="UTC"))
                batch = batch.set_column(position, pa.field(column, instants.type), instants)
            for month in pc.unique(batch.column("start_month")).to_pylist():
                subset = batch.filter(pc.equal(batch.column("start_month"), month))
                if month not in writers:
                    path = destination / f"start_month={month}" / "validated_trips.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writers[month] = stack.enter_context(pq.ParquetWriter(path, subset.schema))
                writers[month].write_batch(subset)


def create_canonical_trip_view(connection, *, exclusion="correlated") -> None:
    """Preserve selection; temporary callers may supply a verified exclusion mode."""
    clauses = {
        "correlated": "WHERE NOT EXISTS (SELECT 1 FROM duplicate_ids d "
                      "WHERE d.normalized_ride_id = trim(t.ride_id))",
        "anti": "ANTI JOIN duplicate_ids d ON d.normalized_ride_id = trim(t.ride_id)",
        "verified_zero": "",
    }
    if exclusion not in clauses:
        raise ValueError("Unknown duplicate exclusion mode")
    connection.execute("CREATE VIEW canonical_trips AS SELECT t.*, "
        "strftime(started_at AT TIME ZONE 'America/New_York','%Y-%m') AS start_month "
        "FROM locally_valid_trips t " + clauses[exclusion])


def create_reconciliation_relations(connection, entries: list[dict], *, duplicate_builder=None,
                                    hourly_builder=None, canonical_builder=None) -> list[dict]:
    """Shared SQL contract; relation lifetime belongs to the caller connection."""
    valid_queries, raw_queries, schema_audit = [], [], []
    for entry in entries:
        source = Path(entry["output_dir"])
        month = sql_literal(entry["month"])
        rejected = sql_literal(source / "rejected_trips.csv")
        valid_query, audit = aligned_trip_query(source / "validated_trips.parquet", entry["month"])
        valid_queries.append(valid_query)
        schema_audit.append(audit)
        columns = rejected_csv_columns(source / "rejected_trips.csv")
        end_station = "end_station_id" if "end_station_id" in columns else "NULL::VARCHAR AS end_station_id"
        # all_varchar alone still runs DuckDB's CSV sniffer at view binding.
        # Each UNION branch can retain a ~30.5 MiB sniffing buffer. Supply the
        # verified header's full string schema and standard writer dialect so
        # binding many archives does not allocate history-wide sniffing buffers.
        # Do not reduce maximum line size or ignore malformed records.
        csv_schema = "{" + ", ".join(sql_literal(name) + ": 'VARCHAR'" for name in columns) + "}"
        raw_queries.extend([
            f"""SELECT {month} AS source_archive_month, source_file, source_row,
                ride_id, trim(ride_id) AS normalized_ride_id,
                CAST(started_at AS VARCHAR) AS started_at, CAST(ended_at AS VARCHAR) AS ended_at,
                started_at_utc_ns, ended_at_utc_ns, start_station_id, end_station_id,
                true AS locally_valid, '' AS original_rejection_reasons
                FROM ({valid_query}) AS aligned""",
            f"""SELECT {month} AS source_archive_month, source_file, CAST(source_row AS BIGINT),
                ride_id, trim(coalesce(ride_id,'')) AS normalized_ride_id,
                started_at, ended_at, NULL::BIGINT AS started_at_utc_ns, NULL::BIGINT AS ended_at_utc_ns,
                start_station_id, {end_station}, false AS locally_valid,
                rejection_reasons AS original_rejection_reasons
                FROM read_csv({rejected}, header=true, auto_detect=false,
                    columns={csv_schema}, delim=',', quote='"', escape='"')""",
        ])
    connection.execute("CREATE VIEW locally_valid_trips AS " + " UNION ALL ".join(valid_queries))
    validate_required_values(connection)
    connection.execute("CREATE VIEW raw_ride_occurrences AS " + " UNION ALL ".join(raw_queries))
    if duplicate_builder is not None:
        duplicate_builder(connection)
    else:
        connection.execute("""CREATE TABLE duplicate_ids AS
            SELECT normalized_ride_id, count(*) AS occurrences,
                   count(DISTINCT source_archive_month) AS archive_count
            FROM raw_ride_occurrences WHERE normalized_ride_id <> ''
            GROUP BY normalized_ride_id HAVING count(*) > 1""")
    if canonical_builder is None:
        create_canonical_trip_view(connection)
    else:
        canonical_builder(connection)
    connection.execute("""CREATE VIEW canonical_rejections AS
        SELECT r.*, d.normalized_ride_id IS NOT NULL AS duplicate_ride_id,
            coalesce(d.archive_count > 1,false) AS cross_archive_duplicate,
            CASE WHEN d.normalized_ride_id IS NOT NULL AND
                 NOT contains(coalesce(original_rejection_reasons,''),'duplicate_ride_id')
                 THEN concat_ws(';',nullif(original_rejection_reasons,''),'duplicate_ride_id')
                 ELSE original_rejection_reasons END AS rejection_reasons
        FROM raw_ride_occurrences r LEFT JOIN duplicate_ids d USING(normalized_ride_id)
        WHERE NOT r.locally_valid OR d.normalized_ride_id IS NOT NULL""")
    if hourly_builder is not None:
        hourly_builder(connection)
    else:
        connection.execute("""CREATE TABLE hourly_rides AS
            SELECT start_month,start_station_id,date_trunc('hour',started_at) AS hour_start,
                   count(*)::BIGINT AS ride_count FROM canonical_trips
            GROUP BY start_month,start_station_id,hour_start""")
    connection.execute("""CREATE VIEW model_window_hourly_rides AS
        SELECT * FROM hourly_rides
        WHERE hour_start >= TIMESTAMPTZ '2024-01-01 00:00:00 America/New_York'
          AND hour_start < TIMESTAMPTZ '2026-01-01 00:00:00 America/New_York'""")
    return schema_audit


def reconciliation_report(connection, entries: list[dict], manifest: dict,
                          schema_audit: list[dict], started: float, *, hourly_validation=None) -> dict:
    """Apply the same global reconciliation and NYC start-month accounting."""
    inputs = sum(entry["input_rows"] for entry in entries)
    accepted = connection.execute("SELECT count(*) FROM canonical_trips").fetchone()[0]
    rejected_count = connection.execute("SELECT count(*) FROM canonical_rejections").fetchone()[0]
    hourly_rows, starts, unique_keys = hourly_validation if hourly_validation is not None else connection.execute("""
        SELECT count(*),coalesce(sum(ride_count),0),count(DISTINCT(start_station_id,hour_start))
        FROM hourly_rides""").fetchone()
    if inputs != accepted + rejected_count or accepted != starts or hourly_rows != unique_keys:
        raise ValueError("Global reconciliation or station-hour uniqueness failed")
    start_counts = dict(connection.execute("SELECT start_month,count(*) FROM canonical_trips GROUP BY start_month ORDER BY start_month").fetchall())
    boundary_counts = connection.execute("""SELECT source_archive_month,start_month,count(*) AS ride_starts
        FROM canonical_trips WHERE source_archive_month <> start_month
        GROUP BY source_archive_month,start_month ORDER BY source_archive_month,start_month""").fetchall()
    duplicate_count, cross_count = connection.execute("SELECT count(*),count(*) FILTER(WHERE archive_count>1) FROM duplicate_ids").fetchone()
    cross_rejected = connection.execute("SELECT count(*) FROM canonical_rejections WHERE locally_valid AND cross_archive_duplicate").fetchone()[0]
    model_count = connection.execute("""SELECT count(*) FROM canonical_trips
        WHERE started_at >= TIMESTAMPTZ '2024-01-01 00:00:00 America/New_York'
          AND started_at < TIMESTAMPTZ '2026-01-01 00:00:00 America/New_York'""").fetchone()[0]
    expected = month_range("2024-01-01", "2026-01-01")
    registered = {entry["month"] for entry in entries}
    missing = sorted(set(expected) - registered)
    report = {"source_archives": [entry["month"] for entry in entries],
              "source_schema_alignment": schema_audit,
              "source_output_sha256": {entry["month"]: entry["output_sha256"] for entry in entries},
              "input_rows": inputs, "accepted_rows": accepted, "rejected_rows": rejected_count,
              "local_accepted_rows": sum(entry["accepted_rows"] for entry in entries),
              "additional_cross_archive_rejections": cross_rejected,
              "duplicate_ride_ids": duplicate_count, "cross_archive_duplicate_ride_ids": cross_count,
              "hourly_rows": hourly_rows, "hourly_ride_starts": starts,
              "duplicate_station_hour_keys": hourly_rows - unique_keys,
              "start_month_counts": start_counts,
              "boundary_start_counts": [{"source_archive_month": source, "start_month": month,
                                          "ride_starts": count} for source,month,count in boundary_counts],
              "modelling_specification": manifest["modelling_specification"],
              "in_window_ride_starts": model_count, "out_of_window_ride_starts": accepted-model_count,
              "missing_modelling_source_archives": missing,
              "january_2026_boundary_input_present": "2026-01" in registered,
              "historical_start_coverage_status": "unverified_partial" if missing or "2026-01" not in registered else "all_planned_sources_present_still_requires_boundary_review",
              "zero_filled": False, "elapsed_seconds": perf_counter()-started,
              "timestamp_policy": "Inherited Phase 1 NYC-naive/DST rejection; UTC ns reconstructed from exact epoch columns",
              "notes": ["Publication-month completeness is not proven start-month completeness.",
                        "Nonadjacent pilots are partial history, not complete modelling partitions.",
                        "No station cohort selection, features, forecast samples or model training in this phase."]}
    return report


def build_catalog(output_root: Path, *, memory_limit: str = "512MB", chunk_size: int = 50_000,
                  reserve_bytes: int = 5 * 1024**3, cloud_store=None,
                  restore_root: Path | None = None) -> dict:
    """Publish a new immutable snapshot; prior source partitions/snapshots stay intact.

    Duplicates reject every occurrence, including valid IDs also seen in rejected
    records. Source ETL reports are immutable; this catalogue adds a global audit.
    """
    from citypulse_ai.etl.archives import check_disk
    root = Path(output_root).resolve()
    manifest = load_manifest(root)
    entries = [entry for _, entry in sorted(manifest["archives"].items()) if entry.get("status") == "processed"]
    if not entries:
        raise ValueError("No complete source partitions registered")
    for index, entry in enumerate(entries):
        if not verified_partition(entry):
            if cloud_store is None:
                raise ValueError(f"Source partition changed: {entry['month']}")
            from citypulse_ai.cloud.retention import restore_partition
            cache = Path(restore_root) if restore_root is not None else root / "restored"
            destination = cache / entry['month'] / entry['cloud_publication']['revision']
            restored = restore_partition(entry, destination, store=cloud_store,
                                         names=['validated_trips.parquet', 'rejected_trips.csv'],
                                         reserve_bytes=reserve_bytes)
            entries[index] = {**entry, 'output_dir': str(restored)}
    source_bytes = sum((Path(entry["output_dir"]) / "validated_trips.parquet").stat().st_size for entry in entries)
    check_disk(root, 8 * source_bytes, reserve_bytes)
    work_root = root / ".work"
    work_root.mkdir(exist_ok=True)
    started = perf_counter()
    with TemporaryDirectory(prefix="catalog-", dir=work_root) as work:
        destination = Path(work) / "snapshot"
        destination.mkdir()
        with duckdb.connect(str(destination / "catalog.duckdb"), config={
            "memory_limit": memory_limit, "threads": 2, "temp_directory": str(Path(work) / "spill"),
        }) as connection:
            connection.execute("SET TimeZone = 'UTC'")
            schema_audit = create_reconciliation_relations(connection, entries)
            report = reconciliation_report(connection, entries, manifest, schema_audit, started)
            write_trip_partitions(connection, destination / "trips", chunk_size)
            connection.execute("COPY (SELECT * FROM hourly_rides ORDER BY start_month,start_station_id,hour_start) TO ? (FORMAT PARQUET, PARTITION_BY(start_month))", [str(destination / "hourly")])
            for table in ("canonical_rejections", "duplicate_ids"):
                connection.execute(f"COPY (SELECT * FROM {table}) TO ? (FORMAT PARQUET)", [str(destination / f"{table}.parquet")])
            # Persistent catalogs retain their original source-file dependencies.
            report["elapsed_seconds"] = perf_counter() - started
            # Auditable DB views refer to the immutable source partitions, not temporary files.
        write_json(destination / "report.json", report)
        snapshots = root / "canonical" / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        target = snapshots / uuid.uuid4().hex
        destination.rename(target)
    write_json(root / "canonical" / "CURRENT.json", {"snapshot": str(target), "report": str(target / "report.json")})
    manifest["canonical"] = {"snapshot": str(target), "input_rows": report["input_rows"], "accepted_rows": report["accepted_rows"]}
    write_json(root / "manifest.json", manifest)
    return {"snapshot": str(target), **report}
