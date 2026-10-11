"""Tiny Parquet fixtures reproduce the real mixed Hive/non-Hive folder layouts.

No downloads, existing warehouse scans, or raw-trip ETL runs are needed here.
"""

import csv
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from citypulse_ai.etl.archives import fingerprint, write_json
from citypulse_ai.etl.historical import load_manifest
from citypulse_ai.etl.history_catalog import TRIP_TYPES, aligned_trip_query, build_catalog
from citypulse_ai.etl.pipeline import AUDIT_COLUMNS, TRIP_SCHEMA


@pytest.fixture
def partition_factory(tmp_path):
    root = tmp_path / "historical"

    def create(month, rows, *, hive=True, missing=(), extra=False, rejection_rows=(), audit_missing_end=False):
        folder = root / (f"source_month={month}" if hive else f"reviewed-{month}")
        folder.mkdir(parents=True)
        records = []
        for index, row in enumerate(rows, start=2):
            start = pd.Timestamp(row.get("start", "2025-02-28 23:59:59.123456789"), tz="America/New_York").tz_convert("UTC")
            end = start + pd.Timedelta(minutes=10)
            records.append({"source_file": str(tmp_path / f"{month}.csv"), "source_row": index,
                            "ride_id": row["id"], "started_at": start, "ended_at": end,
                            "start_station_id": row.get("station", "001.01"), "end_station_id": "",
                            "duration_seconds": 600.0, "started_at_utc_ns": start.value,
                            "ended_at_utc_ns": end.value})
        table = pa.Table.from_pylist(records, schema=TRIP_SCHEMA)
        for name in missing:
            table = table.drop([name])
        if extra:
            table = table.append_column("start_station_name", pa.array(["Example station"] * len(rows), type=pa.string()))
            table = table.select(list(reversed(table.column_names)))
        parquet = folder / "validated_trips.parquet"
        pq.write_table(table, parquet)
        audit = folder / "rejected_trips.csv"
        columns = [name for name in AUDIT_COLUMNS if not (audit_missing_end and name == "end_station_id")]
        with audit.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rejection_rows:
                writer.writerow({name: row.get(name, "") for name in columns})
        manifest = load_manifest(root)
        manifest["archives"][month] = {
            "month": month, "status": "processed", "output_dir": str(folder),
            "input_rows": len(rows) + len(rejection_rows), "accepted_rows": len(rows),
            "rejected_rows": len(rejection_rows),
            "output_sha256": {p.name: fingerprint(p) for p in (parquet, audit)},
        }
        write_json(root / "manifest.json", manifest)
        return root, folder, table

    return create


def test_real_folder_layout_reproduces_old_failure_and_aligns(partition_factory):
    root, january, _ = partition_factory("2024-01", [{"id": "jan", "start": "2024-01-01"}])
    _, february, _ = partition_factory("2025-02", [{"id": "feb"}], hive=False)
    with duckdb.connect() as connection:
        first = str(january / "validated_trips.parquet")
        second = str(february / "validated_trips.parquet")
        assert len(connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [first]).fetchall()) == 11
        assert len(connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [second]).fetchall()) == 10
        with pytest.raises(duckdb.BinderException, match="same number"):
            connection.execute("SELECT *, '2024-01' AS source_archive_month FROM read_parquet(?) UNION ALL SELECT *, '2025-02' AS source_archive_month FROM read_parquet(?)", [first, second])
    result = build_catalog(root, reserve_bytes=0)
    assert result["input_rows"] == result["accepted_rows"] == result["hourly_ride_starts"] == 2
    assert result["rejected_rows"] == result["duplicate_station_hour_keys"] == 0


def test_reordered_extras_optional_columns_precision_and_provenance(partition_factory):
    root, january, _ = partition_factory("2024-01", [
        {"id": "000A", "station": "6569.09_", "start": "2024-01-01"},
        {"id": "000B", "station": "6569.09", "start": "2024-01-01"},
    ], extra=True)
    _, february, _ = partition_factory("2025-02", [{"id": "duplicate"}], hive=False)
    _, march, original = partition_factory("2025-03", [
        {"id": "duplicate"}, {"id": "boundary", "station": "001.01"},
    ], missing=("end_station_id", "duration_seconds"), audit_missing_end=True)
    sources = [p for folder in (january, february, march) for p in folder.iterdir()]
    hashes = {p: fingerprint(p) for p in sources}
    result = build_catalog(root, reserve_bytes=0, chunk_size=1)
    assert result["input_rows"] == 5 and result["accepted_rows"] == 3 and result["rejected_rows"] == 2
    assert result["cross_archive_duplicate_ride_ids"] == 1
    assert result["additional_cross_archive_rejections"] == 2
    assert result["start_month_counts"] == {"2024-01": 2, "2025-02": 1}
    assert result["boundary_start_counts"] == [{"source_archive_month": "2025-03", "start_month": "2025-02", "ride_starts": 1}]
    snapshot = Path(result["snapshot"])
    with duckdb.connect(str(snapshot / "catalog.duckdb"), read_only=True) as connection:
        names = [item[0] for item in connection.execute("DESCRIBE locally_valid_trips").fetchall()]
        assert names == [*TRIP_TYPES, "source_archive_month"]
        assert connection.execute("SELECT ride_id,start_station_id FROM canonical_trips WHERE source_archive_month='2024-01' ORDER BY ride_id").fetchall() == [("000A", "6569.09_"), ("000B", "6569.09")]
        row = connection.execute("SELECT source_archive_month,start_month,source_file,source_row,end_station_id,duration_seconds FROM canonical_trips WHERE ride_id='boundary'").fetchone()
        assert row == ("2025-03", "2025-02", str(march.parent.parent / "2025-03.csv"), 3, None, None)
        assert connection.execute("SELECT sum(ride_count) FROM hourly_rides WHERE start_month='2025-02'").fetchone()[0] == 1
    exported = pq.ParquetFile(snapshot / "trips/start_month=2025-02/validated_trips.parquet").read()
    for column in ("started_at", "ended_at"):
        assert exported.schema.field(column).type == pa.timestamp("ns", tz="UTC")
        assert exported.column(column).cast(pa.int64()).to_pylist() == [original.column(column + "_utc_ns")[1].as_py()]
    assert exported.column("ride_id").to_pylist() == ["boundary"]
    assert all(fingerprint(p) == digest for p, digest in hashes.items())
    audits = {a["source_archive_month"]: a for a in result["source_schema_alignment"]}
    assert audits["2024-01"]["extra_columns_retained_in_source_only"] == ["start_station_name"]
    assert audits["2025-03"]["missing_optional_columns"] == ["duration_seconds", "end_station_id"]
    assert all(a["hive_partitioning"] is False for a in audits.values())


@pytest.mark.parametrize("missing", ["ride_id", "start_station_id", "started_at", "started_at_utc_ns", "source_file", "source_row"])
def test_required_columns_are_never_filled_in(partition_factory, missing):
    root, _, _ = partition_factory("2025-03", [{"id": "one"}], missing=(missing,))
    with pytest.raises(ValueError, match=f"Missing required catalogue columns.*{missing}"):
        build_catalog(root, reserve_bytes=0)
    assert not (root / "canonical/CURRENT.json").exists()


@pytest.mark.parametrize("column,kind,value", [
    ("start_station_id", pa.int64(), 123),
    ("started_at", pa.timestamp("ns"), pd.Timestamp("2025-02-28 23:59:59")),
    ("started_at_utc_ns", pa.float64(), 1.0),
])
def test_lossy_identifier_or_timestamp_types_fail(partition_factory, column, kind, value):
    _, folder, table = partition_factory("2025-03", [{"id": "one"}])
    position = table.schema.get_field_index(column)
    changed = table.set_column(position, pa.field(column, kind), pa.array([value], type=kind))
    path = folder / "invalid-schema.parquet"
    pq.write_table(changed, path)
    with pytest.raises(ValueError):
        aligned_trip_query(path, "2025-03")


def test_present_but_null_optional_columns_are_safe(partition_factory):
    _, folder, table = partition_factory("2025-03", [{"id": "one"}])
    for column in ("end_station_id", "duration_seconds"):
        table = table.set_column(table.schema.get_field_index(column), column, pa.nulls(1))
    path = folder / "null-optionals.parquet"
    pq.write_table(table, path)
    query, _ = aligned_trip_query(path, "2025-03")
    with duckdb.connect() as connection:
        assert connection.execute(f"SELECT end_station_id,duration_seconds FROM ({query})").fetchone() == (None, None)


@pytest.mark.parametrize("column", ["source_file", "ride_id", "start_station_id", "started_at_utc_ns"])
def test_required_null_values_fail_without_publishing(partition_factory, column):
    root, folder, table = partition_factory("2025-03", [{"id": "one"}])
    field = table.schema.field(column)
    table = table.set_column(table.schema.get_field_index(column), field, pa.nulls(1, type=field.type))
    path = folder / "validated_trips.parquet"
    pq.write_table(table, path)
    manifest = load_manifest(root)
    manifest["archives"]["2025-03"]["output_sha256"][path.name] = fingerprint(path)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="lack required values"):
        build_catalog(root, reserve_bytes=0)
    assert not (root / "canonical/CURRENT.json").exists()


def test_missing_optional_audit_field_still_detects_cross_partition_duplicates(partition_factory):
    root, _, _ = partition_factory("2025-02", [{"id": "shared"}], hive=False)
    _, _, _ = partition_factory("2025-03", [{"id": "unique"}], audit_missing_end=True,
        rejection_rows=[{"source_file": "march.csv", "source_row": 3, "ride_id": "shared",
                         "started_at": "bad", "ended_at": "2025-03-01 01:00:00",
                         "start_station_id": "001.01", "rejection_reasons": "started_at:invalid_timestamp"}])
    result = build_catalog(root, reserve_bytes=0)
    assert (result["input_rows"], result["accepted_rows"], result["rejected_rows"]) == (3, 1, 2)
    assert result["cross_archive_duplicate_ride_ids"] == 1
    assert result["additional_cross_archive_rejections"] == 1
    with duckdb.connect(str(Path(result["snapshot"]) / "catalog.duckdb"), read_only=True) as connection:
        audit = connection.execute("SELECT source_archive_month,source_file,source_row,started_at,end_station_id,rejection_reasons FROM canonical_rejections WHERE NOT locally_valid").fetchone()
        assert audit == ("2025-03", "march.csv", 3, "bad", None, "started_at:invalid_timestamp;duplicate_ride_id")


def test_inconsistent_exact_epochs_fail_without_publishing(partition_factory):
    root, folder, table = partition_factory("2025-03", [{"id": "one"}])
    column = "started_at_utc_ns"
    original = table.column(column)[0].as_py()
    table = table.set_column(table.schema.get_field_index(column), column, pa.array([original + 1_000_000_000], type=pa.int64()))
    path = folder / "validated_trips.parquet"
    pq.write_table(table, path)
    manifest = load_manifest(root)
    manifest["archives"]["2025-03"]["output_sha256"][path.name] = fingerprint(path)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="consistent exact UTC epochs"):
        build_catalog(root, reserve_bytes=0)
    assert not (root / "canonical/CURRENT.json").exists()
