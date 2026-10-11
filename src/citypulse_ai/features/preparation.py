"""February forecasting-data preparation with explicit missing-hour policies."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

LOCAL_ZONE = "America/New_York"
START = pd.Timestamp("2025-02-01", tz=LOCAL_ZONE).tz_convert("UTC")
TRAIN_END = pd.Timestamp("2025-02-21", tz=LOCAL_ZONE).tz_convert("UTC")
VALIDATION_END = pd.Timestamp("2025-02-25", tz=LOCAL_ZONE).tz_convert("UTC")
END = pd.Timestamp("2025-03-01", tz=LOCAL_ZONE).tz_convert("UTC")


@dataclass
class PreparedData:
    citywide: pd.DataFrame
    station_hourly: pd.DataFrame
    coverage: pd.DataFrame
    gaps: pd.DataFrame
    activity_periods: pd.DataFrame


def calendar_and_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar values are known in advance; assign identical dates to all stations."""
    result = frame.copy()
    local = result.hour_start.dt.tz_convert(LOCAL_ZONE)
    result["hour"] = local.dt.hour.astype("int8")
    result["weekday"] = local.dt.dayofweek.astype("int8")  # Monday=0.
    result["is_weekend"] = result.weekday.ge(5)
    result["split"] = "train"
    result.loc[result.hour_start.ge(TRAIN_END), "split"] = "validation"
    result.loc[result.hour_start.ge(VALIDATION_END), "split"] = "test"
    return result


def validate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail on corrupt or mismatched input instead of silently dropping records."""
    required = ["start_station_id", "hour_start", "ride_count"]
    if not set(required).issubset(frame):
        raise ValueError("Expected start_station_id, hour_start, and ride_count")
    data = frame[required].copy()
    if data.empty:
        raise ValueError("No observed February station-hours")
    data["start_station_id"] = data.start_station_id.astype("string")
    if data.start_station_id.isna().any() or data.start_station_id.str.strip().eq("").any():
        raise ValueError("Station identifiers must be nonblank strings")
    if not isinstance(data.hour_start.dtype, pd.DatetimeTZDtype):
        raise ValueError("hour_start must be timezone-aware; do not guess its timezone")
    data["hour_start"] = data.hour_start.dt.tz_convert("UTC")
    if data.hour_start.isna().any() or not data.hour_start.eq(data.hour_start.dt.floor("h")).all():
        raise ValueError("Timestamps must be nonmissing UTC hour boundaries")
    if not data.hour_start.ge(START).all() or not data.hour_start.lt(END).all():
        raise ValueError("Input contains hours outside the February NYC reporting window")
    if data.duplicated(["start_station_id", "hour_start"]).any():
        raise ValueError("Duplicate station-hour keys")
    counts = pd.to_numeric(data.ride_count, errors="raise")
    if counts.isna().any() or not counts.gt(0).all() or not counts.mod(1).eq(0).all():
        raise ValueError("Observed ride counts must be positive integers")
    data["ride_count"] = counts.astype("int64")
    return data.sort_values(["start_station_id", "hour_start"]).reset_index(drop=True)


def coverage_intervals(station_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrospective audit only: no-record runs and episodes separated by 24h silence.

    These intervals describe recorded activity, not verified operating schedules.
    Never use full-month episode bounds to choose training rows or features.
    """
    gaps, episodes = [], []
    for station, rows in station_grid.groupby("start_station_id", sort=True):
        missing = rows.ride_count_observed.isna()
        runs = missing.ne(missing.shift()).cumsum()
        first, last = rows.loc[~missing, "hour_start"].agg(["min", "max"])
        for _, run in rows.loc[missing].groupby(runs[missing]):
            left, right = run.hour_start.iloc[0], run.hour_start.iloc[-1]
            position = "before_first_observation" if right < first else (
                "after_last_observation" if left > last else "between_observations"
            )
            gaps.append((station, left, right + pd.Timedelta(hours=1), len(run), position,
                         "no_record_unknown_operation"))
        observed = rows.loc[~missing]
        groups = observed.hour_start.diff().gt(pd.Timedelta(hours=24)).cumsum()
        for _, episode in observed.groupby(groups):
            episodes.append((station, episode.hour_start.iloc[0],
                             episode.hour_start.iloc[-1] + pd.Timedelta(hours=1),
                             len(episode), int(episode.ride_count_observed.sum())))
    gap_columns = ["start_station_id", "start_utc", "end_utc_exclusive", "hours",
                   "position", "activity_status"]
    period_columns = ["start_station_id", "start_utc", "end_utc_exclusive",
                      "observed_hours", "ride_starts"]
    return pd.DataFrame(gaps, columns=gap_columns), pd.DataFrame(episodes, columns=period_columns)


def prepare_hourly(
    observed: pd.DataFrame, *, source_complete: bool = False,
    station_zero_horizon: int | None = None,
) -> PreparedData:
    """Build calendar grids. Station zeros are opt-in assumptions, never observations.

    The optional horizon is a causal, conservative heuristic: a training-seen
    station can receive an assumed zero only within N hours after a recorded
    start. Future observations never back-fill earlier missing labels. Longer
    gaps and stations first seen after training remain unknown. This is not
    evidence of operating status or unconstrained demand.
    """
    data = validate_hourly(observed)
    if station_zero_horizon is not None and station_zero_horizon <= 0:
        raise ValueError("Optional station zero horizon must be positive")
    if station_zero_horizon is not None and not source_complete:
        raise ValueError("Station zero assumptions require complete-source confirmation")
    hours = pd.date_range(START, END, freq="h", inclusive="left")
    city = data.groupby("hour_start").ride_count.sum().reindex(hours).rename_axis("hour_start").reset_index()
    city["ride_count_observed"] = city.ride_count.astype("Int64")
    city["is_zero_filled"] = city.ride_count.isna() & source_complete
    city["ride_count"] = city.ride_count.fillna(0).astype("Int64") if source_complete else city.ride_count.astype("Int64")
    city["coverage_status"] = "observed"
    city.loc[city.is_zero_filled, "coverage_status"] = "zero_complete_source"
    city.loc[city.ride_count.isna(), "coverage_status"] = "unknown"

    # Full grid is an audit universe; station model membership is frozen on train.
    index = pd.MultiIndex.from_product([sorted(data.start_station_id.unique()), hours],
                                      names=["start_station_id", "hour_start"])
    grid = data.set_index(["start_station_id", "hour_start"]).reindex(index).reset_index()
    grid["ride_count_observed"] = grid.ride_count.astype("Int64")
    grid["ride_count"] = grid.ride_count.astype("Int64")
    training_ids = set(data.loc[data.hour_start.lt(TRAIN_END), "start_station_id"])
    grid["training_station"] = grid.start_station_id.isin(training_ids)
    grid["is_zero_filled"] = False
    grid["coverage_status"] = "observed"
    grid.loc[grid.ride_count.isna(), "coverage_status"] = "unknown"
    if station_zero_horizon is not None:
        last_observed = grid.hour_start.where(grid.ride_count_observed.notna())
        last_observed = last_observed.groupby(grid.start_station_id).ffill()
        age = (grid.hour_start - last_observed).dt.total_seconds() / 3600
        fill = grid.training_station & grid.ride_count.isna() & age.le(station_zero_horizon)
        grid.loc[fill, "ride_count"] = 0
        grid.loc[fill, "is_zero_filled"] = True
        grid.loc[fill, "coverage_status"] = "assumed_zero_recent_activity"
    gaps, periods = coverage_intervals(grid)
    coverage = data.groupby("start_station_id").agg(
        observed_hours=("hour_start", "size"), ride_starts=("ride_count", "sum"),
        first_observed_utc=("hour_start", "min"), last_observed_utc=("hour_start", "max"),
    ).reset_index()
    coverage["total_hours"] = len(hours)
    coverage["missing_hours"] = len(hours) - coverage.observed_hours
    coverage["observed_fraction"] = coverage.observed_hours / len(hours)
    coverage["training_station"] = coverage.start_station_id.isin(training_ids)
    gap_max = gaps.groupby("start_station_id").hours.max() if not gaps.empty else pd.Series(dtype="int64")
    coverage["longest_no_record_hours"] = coverage.start_station_id.map(gap_max).fillna(0).astype("int64")
    assert int(city.ride_count.sum()) == int(data.ride_count.sum())
    assert int(grid.ride_count.sum()) == int(data.ride_count.sum())
    return PreparedData(calendar_and_splits(city), calendar_and_splits(grid), coverage, gaps, periods)


def run_preparation(
    warehouse: str | Path, output_dir: str | Path, *, source_complete: bool = False,
    station_zero_horizon: int | None = None,
) -> dict:
    """Read Phase 1 DuckDB without writes; export a separate reproducible dataset."""
    source, destination = Path(warehouse).resolve(), Path(output_dir).resolve()
    if source.parent == destination:
        raise ValueError("Use a new preparation directory, separate from Phase 1 outputs")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with duckdb.connect(str(source), read_only=True) as connection:
        connection.execute("SET TimeZone = 'UTC'")
        observed = connection.execute("SELECT * FROM february_2025_hourly_rides").df()
    prepared = prepare_hourly(observed, source_complete=source_complete,
                              station_zero_horizon=station_zero_horizon)
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Source warehouse changed during preparation")
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in (("citywide_hourly", prepared.citywide),
                        ("station_hourly", prepared.station_hourly),
                        ("station_coverage", prepared.coverage),
                        ("coverage_gaps", prepared.gaps), ("activity_periods", prepared.activity_periods)):
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    stations = prepared.station_hourly
    report = {
        "source_warehouse": str(source), "source_sha256": digest,
        "timezone": LOCAL_ZONE, "window_start_utc": START.isoformat(), "window_end_utc_exclusive": END.isoformat(),
        "source_complete_confirmed": source_complete,
        "station_policy": "unknown_unless_observed" if station_zero_horizon is None else "assumed_zero_recent_activity",
        "station_zero_horizon_hours": station_zero_horizon,
        "observed_ride_starts": int(observed.ride_count.sum()),
        "citywide_hours": len(prepared.citywide), "citywide_zero_filled_hours": int(prepared.citywide.is_zero_filled.sum()),
        "stations": len(prepared.coverage), "training_stations": int(prepared.coverage.training_station.sum()),
        "station_hour_rows": len(stations), "observed_station_hours": int(stations.ride_count_observed.notna().sum()),
        "unknown_station_hours": int(stations.ride_count.isna().sum()),
        "zero_filled_station_hours": int(stations.is_zero_filled.sum()),
        "no_record_intervals": len(prepared.gaps), "no_record_intervals_at_least_24h": int(prepared.gaps.hours.ge(24).sum()),
        "observed_activity_periods": len(prepared.activity_periods),
        "citywide_split_hours": prepared.citywide.groupby("split", sort=True).size().to_dict(),
        "split_boundaries_local": {"train": "[2025-02-01, 2025-02-21)", "validation": "[2025-02-21, 2025-02-25)", "test": "[2025-02-25, 2025-03-01)"},
        "versions": {"pandas": pd.__version__, "duckdb": duckdb.__version__},
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
