"""Regression checks for reviewed IDs, UTC preservation, and local month bounds."""

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from citypulse_ai.etl.pipeline import run_pipeline
from citypulse_ai.etl.trips import validate_trips


def trip(ride_id, start, station="6569.09"):
    return {"ride_id": ride_id, "started_at": start,
            "ended_at": str(pd.Timestamp(start) + pd.Timedelta(minutes=10)),
            "start_station_id": station, "end_station_id": ""}


def test_exact_reviewed_exception_preserves_distinct_ids():
    raw = pd.DataFrame([
        trip("reviewed", "2025-02-01 12:00:00.123456789", "6569.09_"),
        trip("base", "2025-02-01 12:00:00", "6569.09"),
    ])
    result = validate_trips(raw)
    assert result.rejected.empty
    assert result.valid.start_station_id.tolist() == ["6569.09_", "6569.09"]


@pytest.mark.parametrize("station", ["6569.09__", "6569.08_", "_6569.09", "Shop Morgan", "NA"])
def test_other_malformed_identifiers_still_rejected(station):
    result = validate_trips(pd.DataFrame([trip("bad", "2025-02-01 12:00:00", station)]))
    assert result.valid.empty
    assert result.rejected.rejection_reasons.tolist() == ["invalid_start_station_id"]


def test_exception_does_not_bypass_duration_validation():
    raw = pd.DataFrame([trip("bad-duration", "2025-02-01 12:00:00", "6569.09_")])
    raw["ended_at"] = "2025-02-01 11:59:00"
    assert validate_trips(raw).rejected.rejection_reasons.tolist() == ["nonpositive_duration"]


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_february_window_canonical_preservation_and_conservation(chunk_size, tmp_path):
    raw = pd.DataFrame([
        trip("jan-last", "2025-01-31 23:59:59.999999999"),
        trip("feb-first", "2025-02-01 00:00:00"),
        trip("utc-before", "2025-02-01T04:59:59Z"),
        trip("utc-first", "2025-02-01T05:00:00Z"),
        trip("feb-last", "2025-02-28 23:59:59.999999999"),
        trip("mar-first", "2025-03-01 00:00:00"),
        trip("utc-end", "2025-03-01T05:00:00Z"),
        trip("utc-inside", "2025-03-01T04:59:59Z"),
        trip("reviewed", "2025-02-02 12:00:00", "6569.09_"),
        trip("base", "2025-02-02 12:00:00", "6569.09"),
        trip("invalid", "2025-02-02 12:00:00", "6569.09__"),
    ])
    source = tmp_path / "boundaries.csv"
    raw.to_csv(source, index=False)
    output = tmp_path / "output"
    report = run_pipeline(source, output, chunk_size=chunk_size)
    assert report["input_rows"] == report["accepted_rows"] + report["rejected_rows"] == 11
    assert report["accepted_rows"] == 10
    assert report["rejected_rows"] == 1
    assert report["rejection_reason_counts"] == {"invalid_start_station_id": 1}
    assert report["validation_policy"]["reviewed_station_id_exceptions"] == ["6569.09_"]
    assert report["february_2025"]["ride_starts"] == 6
    assert report["february_2025"]["excluded_ride_starts"] == 4
    canonical = pd.read_parquet(output / "validated_trips.parquet")
    assert len(canonical) == 10
    assert canonical.ride_id.tolist() == raw.ride_id.tolist()[:-1]
    first = canonical.iloc[0]
    assert first.started_at == pd.Timestamp("2025-02-01T04:59:59.999999999Z")
    assert first.started_at_utc_ns == first.started_at.value
    assert first.source_row == 2
    assert str(canonical.started_at.dtype) == "datetime64[ns, UTC]"
    hourly = pd.read_parquet(output / "hourly_rides.parquet")
    february = pd.read_parquet(output / "february_2025_hourly_rides.parquet")
    assert hourly.ride_count.sum() == 10
    assert february.ride_count.sum() == 6
    assert not february.duplicated(["start_station_id", "hour_start"]).any()
    assert february.ride_count.gt(0).all()  # No zero-filling.
    assert february.hour_start.min() == pd.Timestamp("2025-02-01T05:00:00Z")
    assert february.hour_start.max() == pd.Timestamp("2025-03-01T04:00:00Z")
    with duckdb.connect(str(output / "citypulse.duckdb")) as connection:
        connection.execute("SET TimeZone = 'Asia/Tokyo'")
        # Explicit NYC boundary literals work regardless of SQL session timezone.
        assert connection.execute("SELECT SUM(ride_count) FROM february_2025_hourly_rides").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM validated_trips").fetchone() == (10,)
        exact = connection.execute(
            "SELECT started_at_utc_ns FROM validated_trips WHERE ride_id='jan-last'"
        ).fetchone()[0]
        assert exact == first.started_at.value
        assert connection.execute(
            "SELECT SUM(ride_count) FROM hourly_rides WHERE start_station_id='6569.09_'"
        ).fetchone() == (1,)
    assert json.loads((output / "report.json").read_text()) == report


def test_empty_canonical_and_february_outputs(tmp_path):
    source = Path(__file__).parent / "fixtures" / "trips_empty.csv"
    report = run_pipeline(source, tmp_path)
    assert report["february_2025"]["ride_starts"] == 0
    assert pd.read_parquet(tmp_path / "validated_trips.parquet").empty
    assert pd.read_parquet(tmp_path / "february_2025_hourly_rides.parquet").empty
