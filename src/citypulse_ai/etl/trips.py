"""CSV ingestion, deterministic row validation, and hourly ride-start counts."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from citypulse_ai.etl.schema import LOCAL_TIMEZONE, RAW_COLUMNS


class SchemaError(ValueError):
    """The source does not contain the required raw columns."""


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame


def read_trips_csv(path: str | Path) -> pd.DataFrame:
    """Read identifiers as strings, preserving leading zeros and literal 'NA'."""
    try:
        raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError("CSV has no header; required columns are missing") from exc
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}")
    return raw.loc[:, list(RAW_COLUMNS)]


def _parse_timestamp(value: str) -> tuple[pd.Timestamp, str | None]:
    # Require an explicit date/time shape; do not accept inferred day/month formats.
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
        r"(?:Z|[+-]\d{2}:\d{2})?", value
    ):
        return pd.NaT, "invalid_timestamp"
    try:
        stamp = pd.Timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(LOCAL_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
            if pd.isna(stamp):
                return pd.NaT, "ambiguous_or_nonexistent_local_time"
        stamp = stamp.tz_convert("UTC")
        # Enforce the output's nanosecond range before building a typed DataFrame.
        stamp = stamp.as_unit("ns")
        return stamp, None
    except (ValueError, OverflowError):
        return pd.NaT, "invalid_timestamp"


def validate_trips(raw: pd.DataFrame) -> ValidationResult:
    """Reject invalid rows and all occurrences of duplicated, nonblank ride IDs.

    Source row numbers include the header (first data record is row 2). Reasons
    are accumulated so no arbitrary duplicate survivor or timestamp is selected.
    """
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}")
    cleaned = raw.loc[:, list(RAW_COLUMNS)].astype("string").fillna("")
    cleaned = cleaned.apply(lambda column: column.str.strip()).reset_index(drop=True)
    duplicates = cleaned.ride_id.duplicated(keep=False) & cleaned.ride_id.ne("")
    reasons: list[str] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for index, row in cleaned.iterrows():
        errors = [f"missing_{column}" for column in RAW_COLUMNS if not row[column]]
        if duplicates.iloc[index]:
            errors.append("duplicate_ride_id")
        parsed = []
        for column in ("started_at", "ended_at"):
            stamp, error = _parse_timestamp(row[column]) if row[column] else (pd.NaT, None)
            parsed.append(stamp)
            if error:
                errors.append(f"{column}:{error}")
        start, end = parsed
        if pd.notna(start) and pd.notna(end):
            try:
                duration = end - start
                if duration <= pd.Timedelta(0):
                    errors.append("nonpositive_duration")
            except (ValueError, OverflowError):
                errors.append("invalid_duration")
        starts.append(start)
        ends.append(end)
        reasons.append(";".join(errors))

    accepted = pd.Series([not reason for reason in reasons], dtype=bool)
    rejected = raw.loc[:, list(RAW_COLUMNS)].reset_index(drop=True).loc[~accepted].copy()
    rejected.insert(0, "source_row", [i + 2 for i in rejected.index])
    rejected["rejection_reasons"] = [reasons[i] for i in rejected.index]
    cleaned["started_at"] = pd.Series(starts, dtype="datetime64[ns, UTC]")
    cleaned["ended_at"] = pd.Series(ends, dtype="datetime64[ns, UTC]")
    valid = cleaned.loc[accepted].reset_index(drop=True)
    valid["duration_seconds"] = (valid.ended_at - valid.started_at).dt.total_seconds()
    return ValidationResult(valid=valid, rejected=rejected.reset_index(drop=True))


def aggregate_hourly(trips: pd.DataFrame) -> pd.DataFrame:
    """Count accepted ride starts per station and UTC hour; omit unobserved bins."""
    starts = trips.assign(hour_start=trips.started_at.dt.floor("h"))
    return (
        starts.groupby(["start_station_id", "hour_start"], sort=True)
        .size()
        .rename("ride_count")
        .reset_index()
    )
