"""Forecasting preparation checks use small counts, not the raw trip dataset."""

import hashlib

import duckdb
import pandas as pd
import pytest

from citypulse_ai.features.preparation import (
    END, START, TRAIN_END, VALIDATION_END, prepare_hourly, run_preparation,
)


def observations(*rows):
    return pd.DataFrame(
        [(station, START + pd.Timedelta(hours=hour), count) for station, hour, count in rows],
        columns=["start_station_id", "hour_start", "ride_count"],
    )


def test_complete_calendar_and_conservation():
    prepared = prepare_hourly(observations(("6569.09_", 0, 2), ("6569.09", 0, 3), ("6569.09", 2, 1)))
    assert len(prepared.citywide) == 672
    assert len(prepared.station_hourly) == 1344
    assert prepared.citywide.ride_count.sum() == prepared.station_hourly.ride_count.sum() == 6
    assert prepared.citywide.ride_count.isna().sum() == 670
    assert not prepared.station_hourly.is_zero_filled.any()
    assert not prepared.station_hourly.duplicated(["start_station_id", "hour_start"]).any()
    assert set(prepared.coverage.start_station_id) == {"6569.09", "6569.09_"}


def test_citywide_zero_requires_explicit_source_confirmation():
    prepared = prepare_hourly(observations(("001.01", 1, 7)), source_complete=True)
    assert prepared.citywide.ride_count.sum() == 7
    assert prepared.citywide.is_zero_filled.sum() == 671
    assert prepared.station_hourly.ride_count.isna().sum() == 671
    assert prepared.citywide.loc[0, "coverage_status"] == "zero_complete_source"
    assert prepared.citywide.loc[0, "ride_count_observed"] is pd.NA


def test_local_calendar_and_chronological_boundaries():
    prepared = prepare_hourly(observations(("A", 0, 1)))
    city = prepared.citywide.set_index("hour_start")
    assert city.index[0] == START and city.index[-1] == END - pd.Timedelta(hours=1)
    assert city.loc[START, "hour"] == 0
    assert city.loc[START, "weekday"] == 5 and city.loc[START, "is_weekend"]
    assert city.loc[TRAIN_END - pd.Timedelta(hours=1), "split"] == "train"
    assert city.loc[TRAIN_END, "split"] == "validation"
    assert city.loc[VALIDATION_END, "split"] == "test"
    assert city.groupby("split").size().to_dict() == {"test": 96, "train": 480, "validation": 96}
    assert prepared.station_hourly["split"].tolist() == prepared.citywide["split"].tolist()


def test_optional_station_zeros_are_flagged_past_only_and_bounded():
    prepared = prepare_hourly(
        observations(("A", 2, 3), ("A", 10, 1), ("NEW", 500, 2)),
        source_complete=True, station_zero_horizon=2,
    )
    grid = prepared.station_hourly
    station = grid[grid.start_station_id.eq("A")].reset_index(drop=True)
    assert station.ride_count.iloc[:2].isna().all()  # Never back-fill before first observation.
    assert station.ride_count.iloc[3:5].tolist() == [0, 0]
    assert pd.isna(station.ride_count.iloc[5])  # Long silence remains unknown.
    assert station.is_zero_filled.sum() == 4
    assert station.loc[3, "coverage_status"] == "assumed_zero_recent_activity"
    new = grid[grid.start_station_id.eq("NEW")]
    assert not new.training_station.any() and not new.is_zero_filled.any()
    assert grid.ride_count.sum() == 6


def test_future_observation_does_not_change_training_labels():
    early = observations(("A", 2, 3))
    future = observations(("A", 600, 4), ("NEW", 600, 2))
    before = prepare_hourly(early, source_complete=True, station_zero_horizon=24).station_hourly
    after = prepare_hourly(pd.concat([early, future]), source_complete=True, station_zero_horizon=24).station_hourly
    before = before[before["split"].eq("train")].reset_index(drop=True)
    after = after[after["split"].eq("train") & after.start_station_id.eq("A")].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_gap_audit_and_activity_episodes_do_not_claim_closure():
    prepared = prepare_hourly(observations(("A", 1, 2), ("A", 2, 3), ("A", 30, 4)))
    assert prepared.gaps.hours.sum() == 669
    assert set(prepared.gaps.position) == {"before_first_observation", "between_observations", "after_last_observation"}
    assert set(prepared.gaps.activity_status) == {"no_record_unknown_operation"}
    assert prepared.activity_periods.observed_hours.tolist() == [2, 1]
    assert prepared.activity_periods.ride_starts.sum() == 9
    assert prepared.coverage.longest_no_record_hours.iloc[0] == 641


@pytest.mark.parametrize("change", ["duplicate", "naive", "outside", "fraction", "zero", "blank", "off_hour", "missing_count"])
def test_invalid_hourly_inputs_fail(change):
    data = observations(("A", 0, 2))
    if change == "duplicate":
        data = pd.concat([data, data])
    elif change == "naive":
        data["hour_start"] = data.hour_start.dt.tz_localize(None)
    elif change == "outside":
        data.loc[0, "hour_start"] = END
    elif change == "fraction":
        data["ride_count"] = 1.5
    elif change == "zero":
        data["ride_count"] = 0
    elif change == "blank":
        data["start_station_id"] = " "
    elif change == "off_hour":
        data.loc[0, "hour_start"] = START + pd.Timedelta(seconds=1)
    else:
        data["ride_count"] = float("nan")
    with pytest.raises(ValueError):
        prepare_hourly(data)


def test_empty_and_missing_schema_fail():
    with pytest.raises(ValueError):
        prepare_hourly(observations(("A", 0, 1)).iloc[:0])
    with pytest.raises(ValueError):
        prepare_hourly(pd.DataFrame())


@pytest.mark.parametrize("horizon,complete", [(0, True), (-1, True), (24, False)])
def test_invalid_zero_policy_fails(horizon, complete):
    with pytest.raises(ValueError):
        prepare_hourly(observations(("A", 0, 1)), source_complete=complete, station_zero_horizon=horizon)


def test_read_only_reproducible_exports(tmp_path):
    warehouse = tmp_path / "phase1" / "source.duckdb"
    warehouse.parent.mkdir()
    observed = observations(("001.01", 0, 7), ("001.01", 2, 3))
    with duckdb.connect(str(warehouse)) as connection:
        connection.register("observed", observed)
        connection.execute("CREATE TABLE february_2025_hourly_rides AS SELECT * FROM observed")
    original = hashlib.sha256(warehouse.read_bytes()).hexdigest()
    report = run_preparation(warehouse, tmp_path / "prepared", source_complete=True)
    assert report["source_sha256"] == original == hashlib.sha256(warehouse.read_bytes()).hexdigest()
    assert report["observed_ride_starts"] == 10
    assert report["station_policy"] == "unknown_unless_observed"
    repeated = run_preparation(warehouse, tmp_path / "repeated", source_complete=True)
    assert report == repeated
    for name in ["citywide_hourly", "station_hourly", "station_coverage", "coverage_gaps", "activity_periods"]:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "prepared" / f"{name}.parquet"),
                                      pd.read_parquet(tmp_path / "repeated" / f"{name}.parquet"))
    with pytest.raises(ValueError, match="separate"):
        run_preparation(warehouse, warehouse.parent)
