"""Synthetic-data checks for the full trip ETL contract."""

import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from citypulse_ai.etl.pipeline import run_pipeline
from citypulse_ai.etl.schema import RAW_COLUMNS
from citypulse_ai.etl.trips import (
    SchemaError, aggregate_hourly, read_trips_csv, validate_trips,
)

FIXTURES = Path(__file__).parent / "fixtures"


def cli_environment():
    # pytest's pythonpath setting affects this process only, not subprocesses.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(FIXTURES.parents[1] / "src")
    return environment


def test_ingestion_preserves_ids_and_normalizes_timestamps():
    raw = read_trips_csv(FIXTURES / "trips_valid.csv")
    assert raw.start_station_id.tolist() == ["001", "001", "001", "002"]
    result = validate_trips(raw)
    assert len(result.valid) == 4
    assert result.rejected.empty
    assert result.valid.started_at.iloc[0] == pd.Timestamp("2024-01-15T13:05:00Z")
    assert result.valid.duration_seconds.tolist() == [600, 900, 1200, 300]


@pytest.mark.parametrize("column", RAW_COLUMNS)
def test_missing_columns_fail_before_output(column, tmp_path):
    path = tmp_path / "missing.csv"
    read_trips_csv(FIXTURES / "trips_valid.csv").drop(columns=column).to_csv(path, index=False)
    with pytest.raises(SchemaError, match=column):
        run_pipeline(path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_invalid_rows_have_audit_reasons():
    result = validate_trips(read_trips_csv(FIXTURES / "trips_invalid.csv"))
    assert result.valid.empty
    assert result.rejected.source_row.tolist() == list(range(2, 11))
    assert result.rejected.rejection_reasons.tolist() == [
        "missing_ride_id", "started_at:invalid_timestamp", "nonpositive_duration",
        "nonpositive_duration", "missing_start_station_id", "missing_end_station_id",
        "missing_started_at", "duplicate_ride_id", "duplicate_ride_id",
    ]


def test_duplicate_policy_is_independent_of_input_order():
    raw = read_trips_csv(FIXTURES / "trips_valid.csv")
    duplicate = raw.iloc[[0]].copy()
    duplicate["start_station_id"] = "999"
    raw = pd.concat([raw, duplicate], ignore_index=True)
    first = validate_trips(raw)
    reversed_result = validate_trips(raw.iloc[::-1])
    assert set(first.valid.ride_id) == set(reversed_result.valid.ride_id) == {"r2", "r3", "r4"}
    assert first.rejected.rejection_reasons.tolist() == ["duplicate_ride_id"] * 2


def test_hourly_aggregation():
    result = validate_trips(read_trips_csv(FIXTURES / "trips_valid.csv"))
    hourly = aggregate_hourly(result.valid)
    assert list(hourly.itertuples(index=False, name=None)) == [
        ("001", pd.Timestamp("2024-01-15T13:00:00Z"), 2),
        ("001", pd.Timestamp("2024-01-15T14:00:00Z"), 1),
        ("002", pd.Timestamp("2024-01-15T13:00:00Z"), 1),
    ]


def test_dst_is_not_guessed_and_elapsed_duration_is_in_utc():
    result = validate_trips(read_trips_csv(FIXTURES / "trips_dst.csv"))
    assert result.rejected.ride_id.tolist() == ["ambiguous", "nonexistent"]
    assert result.rejected.rejection_reasons.tolist() == [
        "started_at:ambiguous_or_nonexistent_local_time"
    ] * 2
    assert result.valid.duration_seconds.tolist() == [600, 600, 600]
    hourly = aggregate_hourly(result.valid)
    assert hourly.hour_start.tolist() == [
        pd.Timestamp("2024-03-10T06:00:00Z"),
        pd.Timestamp("2024-11-03T05:00:00Z"),
        pd.Timestamp("2024-11-03T06:00:00Z"),
    ]


def test_whitespace_and_missing_values():
    raw = read_trips_csv(FIXTURES / "trips_valid.csv")
    raw.loc[0, "ride_id"] = " r1 "
    raw.loc[0, "start_station_id"] = " NA "
    raw.loc[1, "start_station_id"] = "  "
    raw.loc[2, "ended_at"] = pd.NA
    result = validate_trips(raw)
    assert result.valid.ride_id.tolist() == ["r1", "r4"]
    assert result.valid.start_station_id.iloc[0] == "NA"
    assert result.rejected.rejection_reasons.tolist() == [
        "missing_start_station_id", "missing_ended_at"
    ]


@pytest.mark.parametrize("timestamp", ["01/15/2024 08:00", "2024-02-30 08:00:00", "9999-01-01 08:00:00"])
def test_invalid_date_formats_and_ranges(timestamp):
    raw = read_trips_csv(FIXTURES / "trips_valid.csv").iloc[[0]].copy()
    raw["started_at"] = timestamp
    result = validate_trips(raw)
    assert result.valid.empty
    assert result.rejected.rejection_reasons.iloc[0] == "started_at:invalid_timestamp"


def test_unrepresentable_duration_is_rejected():
    raw = read_trips_csv(FIXTURES / "trips_valid.csv").iloc[[0]].copy()
    raw["started_at"] = "1700-01-01T00:00:00Z"
    raw["ended_at"] = "2200-01-01T00:00:00Z"
    result = validate_trips(raw)
    assert result.valid.empty
    assert result.rejected.rejection_reasons.iloc[0] == "invalid_duration"


def test_outputs_and_reproducible_rerun(tmp_path):
    source = FIXTURES / "trips_valid.csv"
    report = run_pipeline(source, tmp_path)
    assert report == run_pipeline(source, tmp_path)
    assert report["accepted_rows"] == 4
    assert report["rejected_rows"] == 0
    assert report["hourly_rows"] == 3
    assert len(report["input_sha256"]) == 64
    assert json.loads((tmp_path / "report.json").read_text()) == report
    parquet = pd.read_parquet(tmp_path / "hourly_rides.parquet")
    assert str(parquet.hour_start.dtype) == "datetime64[ns, UTC]"
    assert parquet.ride_count.sum() == 4
    assert pd.read_csv(tmp_path / "rejected_trips.csv").empty
    with duckdb.connect(str(tmp_path / "citypulse.duckdb"), read_only=True) as connection:
        assert connection.execute(
            "SELECT start_station_id, SUM(ride_count) FROM hourly_rides GROUP BY 1 ORDER BY 1"
        ).fetchall() == [("001", 3), ("002", 1)]
        connection.execute("SET TimeZone = 'UTC'")
        hours = connection.execute(
            "SELECT hour_start FROM hourly_rides ORDER BY start_station_id, hour_start"
        ).fetchall()
        assert pd.Timestamp(hours[0][0]) == pd.Timestamp("2024-01-15T13:00:00Z")


@pytest.mark.parametrize("fixture", ["trips_empty.csv", "trips_invalid.csv"])
def test_empty_and_all_rejected_inputs_produce_typed_outputs(fixture, tmp_path):
    report = run_pipeline(FIXTURES / fixture, tmp_path)
    assert report["accepted_rows"] == report["hourly_rows"] == 0
    assert report["input_rows"] == report["rejected_rows"]
    frame = pd.read_parquet(tmp_path / "hourly_rides.parquet")
    assert frame.empty
    assert str(frame.hour_start.dtype) == "datetime64[ns, UTC]"
    assert str(frame.ride_count.dtype) == "int64"
    with duckdb.connect(str(tmp_path / "citypulse.duckdb")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hourly_rides").fetchone() == (0,)


def test_headerless_empty_input_is_schema_error(tmp_path):
    source = tmp_path / "blank.csv"
    source.write_text("")
    with pytest.raises(SchemaError, match="no header"):
        read_trips_csv(source)


def test_cli(tmp_path):
    # The module entry point also works before reinstalling the console script.
    process = subprocess.run(
        [sys.executable, "-m", "citypulse_ai.etl.cli", "--input",
         str(FIXTURES / "trips_valid.csv"), "--output-dir", str(tmp_path)],
        capture_output=True, text=True, check=True, env=cli_environment(),
    )
    assert json.loads(process.stdout)["accepted_rows"] == 4


def test_cli_schema_failure(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("ride_id\nr1\n")
    process = subprocess.run(
        [sys.executable, "-m", "citypulse_ai.etl.cli", "--input", str(source),
         "--output-dir", str(tmp_path / "output")],
        capture_output=True, text=True, env=cli_environment(),
    )
    assert process.returncode == 2
    assert "Missing required columns" in process.stderr
    assert not (tmp_path / "output").exists()
