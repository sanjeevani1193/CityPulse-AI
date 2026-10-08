"""Raw CSV contract and shared timestamp policy."""

RAW_COLUMNS = (
    "ride_id",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
)
LOCAL_TIMEZONE = "America/New_York"
TIMESTAMP_POLICY = (
    "ISO-8601 timestamps with seconds; naive timestamps are America/New_York. "
    "Reject ambiguous or nonexistent naive DST times; explicit offsets identify "
    "instants. Normalize to UTC and aggregate by UTC hour."
)
