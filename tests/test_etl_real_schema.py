"""Synthetic fixtures with the real 13-column schema; no large CSVs needed."""

import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from citypulse_ai.etl.pipeline import run_pipeline
from citypulse_ai.etl.trips import aggregate_hourly, iter_trip_chunks, read_trips_csv, validate_trips

FIXTURES = Path(__file__).parent / "fixtures"
PARTS = [FIXTURES / f"trips_real_schema_part{part}.csv" for part in (1, 2)]


def test_real_schema_and_fractional_precision():
    raw = read_trips_csv(PARTS[0])
    assert raw.start_station_id.tolist() == ["0010.02"] * 3
    assert raw.end_station_id.iloc[0] == ""
    result = validate_trips(raw)
    assert len(result.valid) == 3
    assert result.valid.started_at.iloc[0] == pd.Timestamp("2025-02-01T13:00:00.123456789Z")
    assert result.valid.duration_seconds.iloc[0] == 600
    assert aggregate_hourly(result.valid).ride_count.tolist() == [2, 1]


@pytest.mark.parametrize("fraction", ["1", "123", "123456", "123456789"])
def test_fractional_timestamps(fraction):
    raw = read_trips_csv(PARTS[0]).iloc[[0]].copy()
    raw["started_at"] = f"2025-02-01 08:00:00.{fraction}"
    result = validate_trips(raw)
    assert result.rejected.empty
    assert result.valid.started_at.iloc[0] == pd.Timestamp(f"2025-02-01T13:00:00.{fraction}Z")


def test_missing_end_station_does_not_excuse_invalid_end_time():
    raw = read_trips_csv(PARTS[0]).iloc[[0]].copy()
    raw["ended_at"] = "not-a-date"
    result = validate_trips(raw)
    assert result.valid.empty
    assert result.rejected.rejection_reasons.iloc[0] == "ended_at:invalid_timestamp"


@pytest.mark.parametrize("station", ["NaN", "NULL", "N/A", "bad station", "!", "123..45"])
def test_invalid_start_station_ids(station):
    raw = read_trips_csv(PARTS[0]).iloc[[0]].copy()
    raw["start_station_id"] = station
    result = validate_trips(raw)
    assert result.valid.empty
    assert result.rejected.rejection_reasons.iloc[0] == "invalid_start_station_id"


@pytest.mark.parametrize("chunk_size", [1, 2, 50])
def test_cross_file_duplicates_aggregation_and_audit(chunk_size, tmp_path):
    report = run_pipeline(PARTS[::-1], tmp_path, chunk_size=chunk_size)
    assert report["input_rows"] == 6
    assert report["accepted_rows"] == 4
    assert report["rejected_rows"] == 2
    assert report["duplicate_ride_ids"] == 1
    assert report["accepted_missing_end_station_rows"] == 2
    assert report["rejection_reason_counts"] == {"duplicate_ride_id": 2}
    hourly = pd.read_parquet(tmp_path / "hourly_rides.parquet")
    assert list(hourly.itertuples(index=False, name=None)) == [
        ("0010.02", pd.Timestamp("2025-02-01T13:00:00Z"), 2),
        ("0010.02", pd.Timestamp("2025-02-01T14:00:00Z"), 1),
        ("0011.00", pd.Timestamp("2025-02-01T13:00:00Z"), 1),
    ]
    audit = pd.read_csv(tmp_path / "rejected_trips.csv", dtype={"ride_id": "string"})
    assert audit.ride_id.tolist() == ["shared", "shared"]
    assert audit.source_row.tolist() == [4, 3]
    assert audit.source_file.tolist() == [str(path.resolve()) for path in PARTS]
    assert report["inputs"] == sorted(report["inputs"], key=lambda item: item["input_file"])
    assert report == run_pipeline(PARTS, tmp_path, chunk_size=chunk_size)
    with duckdb.connect(str(tmp_path / "citypulse.duckdb")) as connection:
        assert connection.execute("SELECT SUM(ride_count) FROM hourly_rides").fetchone() == (4,)


def test_duplicates_across_chunks_normalize_whitespace(tmp_path):
    raw = read_trips_csv(PARTS[0])
    duplicate = raw.iloc[[0]].copy()
    duplicate["ride_id"] = " a "
    source = tmp_path / "trips.csv"
    pd.concat([raw, duplicate], ignore_index=True).to_csv(source, index=False)
    report = run_pipeline(source, tmp_path / "outputs", chunk_size=1)
    assert report["accepted_rows"] == 2
    assert report["rejected_rows"] == 2
    audit = pd.read_csv(tmp_path / "outputs" / "rejected_trips.csv")
    assert audit.source_row.tolist() == [2, 5]
    assert audit.ride_id.tolist() == ["a", " a "]


def test_chunk_reader_and_sampling_scope(tmp_path):
    chunks = list(iter_trip_chunks(PARTS[0], chunk_size=1, max_rows=2))
    assert [len(chunk) for chunk in chunks] == [1, 1]
    assert chunks[1].index.tolist() == [1]
    report = run_pipeline(PARTS, tmp_path, chunk_size=1, max_rows_per_file=1)
    assert report["input_rows"] == 2
    assert report["processing"]["max_rows_per_file"] == 1
    assert report["duplicate_ride_ids"] == 0
    assert [entry["input_rows"] for entry in report["inputs"]] == [1, 1]
    assert json.loads((tmp_path / "report.json").read_text()) == report


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"max_rows_per_file": 0}])
def test_bad_processing_limits(kwargs, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_pipeline(PARTS, tmp_path, **kwargs)


def test_repeated_input_file_is_not_double_counted(tmp_path):
    with pytest.raises(ValueError, match="more than once"):
        run_pipeline([PARTS[0], PARTS[0]], tmp_path)


def test_input_column_order_does_not_change_counts(tmp_path):
    raw = read_trips_csv(PARTS[0])
    source = tmp_path / "reordered.csv"
    raw.loc[:, raw.columns[::-1]].to_csv(source, index=False)
    report = run_pipeline(source, tmp_path / "outputs", chunk_size=1)
    assert report["accepted_rows"] == 3
    hourly = pd.read_parquet(tmp_path / "outputs" / "hourly_rides.parquet")
    assert hourly.start_station_id.tolist() == ["0010.02", "0010.02"]
    assert hourly.ride_count.tolist() == [2, 1]


def test_cli_accepts_multiple_files_and_sampling_limits(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(FIXTURES.parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "citypulse_ai.etl.cli", "--input", *map(str, PARTS),
         "--output-dir", str(tmp_path), "--chunk-size", "1", "--max-rows-per-file", "1"],
        env=environment, capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report["input_rows"] == 2
    assert report["accepted_rows"] == 2
    assert report["processing"]["chunk_size"] == 1


def test_input_change_aborts_before_publishing(tmp_path, monkeypatch):
    from citypulse_ai.etl import pipeline
    source = tmp_path / "source.csv"
    source.write_bytes(PARTS[0].read_bytes())
    original_reader = pipeline.iter_trip_chunks

    def changed_reader(*args):
        yield from original_reader(*args)
        with source.open("a") as stream:
            stream.write("\n")

    monkeypatch.setattr(pipeline, "iter_trip_chunks", changed_reader)
    with pytest.raises(ValueError, match="changed during ingestion"):
        run_pipeline(source, tmp_path / "outputs", chunk_size=1)
    assert not (tmp_path / "outputs").exists()
