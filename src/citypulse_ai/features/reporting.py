"""Derived reporting datasets; canonical validated records remain unfiltered."""

from pathlib import Path

FEBRUARY_VIEW = "february_2025_hourly_rides"
FEBRUARY_START = "2025-02-01 00:00:00 America/New_York"
FEBRUARY_END = "2025-03-01 00:00:00 America/New_York"


def prepare_february_2025(connection, parquet_path: str | Path) -> dict:
    """Select observed February NYC station-hours without changing UTC instants.

    This calendar window starts/ends at hour boundaries, so filtering canonical
    hourly counts is equivalent to filtering individual ride starts before
    aggregation. No zeros are added and canonical tables are never updated.
    """
    connection.execute(f"""
        CREATE OR REPLACE VIEW {FEBRUARY_VIEW} AS
        SELECT start_station_id, hour_start, ride_count FROM hourly_rides
        WHERE hour_start >= TIMESTAMPTZ '{FEBRUARY_START}'
          AND hour_start < TIMESTAMPTZ '{FEBRUARY_END}'
    """)
    connection.execute(
        f"COPY (SELECT * FROM {FEBRUARY_VIEW} ORDER BY start_station_id, hour_start) "
        "TO ? (FORMAT PARQUET)", [str(parquet_path)],
    )
    rows, starts = connection.execute(
        f"SELECT COUNT(*), COALESCE(SUM(ride_count), 0) FROM {FEBRUARY_VIEW}"
    ).fetchone()
    canonical_starts = connection.execute(
        "SELECT COALESCE(SUM(ride_count), 0) FROM hourly_rides"
    ).fetchone()[0]
    return {"view": FEBRUARY_VIEW, "timezone": "America/New_York",
            "start_inclusive": FEBRUARY_START, "end_exclusive": FEBRUARY_END,
            "hourly_rows": rows, "ride_starts": starts,
            "excluded_ride_starts": canonical_starts - starts, "zero_filled": False}
