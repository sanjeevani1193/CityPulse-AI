"""Raw CSV contract and shared timestamp policy."""

RAW_COLUMNS = (
    "ride_id",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
)
LOCAL_TIMEZONE = "America/New_York"
REQUIRED_VALUES = tuple(column for column in RAW_COLUMNS if column != "end_station_id")
STATION_ID_PATTERN = r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
STATION_NULL_MARKERS = {"na", "nan", "null", "none", "n/a", "<na>"}
# Reviewed from February source names/coordinates. Preserve as a distinct ID.
REVIEWED_STATION_IDS = frozenset({"6569.09_"})
TIMESTAMP_POLICY = (
    "ISO-8601 timestamps with seconds and up to 9 fractional digits; naive "
    "timestamps are assumed America/New_York (publisher confirmation pending). "
    "Reject ambiguous or nonexistent naive DST times; explicit offsets identify "
    "instants. Normalize to UTC and aggregate by UTC hour."
)
