"""Read-only January–February 2024 benchmark inputs, including boundary archives."""

from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import combinations

import duckdb
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.historical import load_manifest
from citypulse_ai.etl.history_catalog import sql_literal
from citypulse_ai.forecasting.baselines import LOCAL_ZONE, prepare_citywide

START = pd.Timestamp("2024-01-01", tz=LOCAL_ZONE)
END = pd.Timestamp("2024-03-01", tz=LOCAL_ZONE)
SOURCE_MONTHS = ("2024-01", "2024-02", "2024-03")


def load_development_citywide(output_root: str | Path) -> tuple[pd.DataFrame, dict]:
    """Reuse immutable hourly outputs; fail rather than double-count duplicate rides.

    Boundary contributions are selected by NYC-local start month, not archive
    name. Ride-ID checks include rejection audits. Only compact citywide sums
    leave DuckDB; no raw CSV ingestion, warehouse writes or GCS access occurs.
    """
    root = Path(output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest_digest = fingerprint(manifest_path)
    manifest = load_manifest(root)
    sources, hourly_queries, id_queries = [], [], []
    expected_months = {"2024-01": 0, "2024-02": 0}
    checked_files = {}
    for month in SOURCE_MONTHS:
        entry = manifest["archives"].get(month, {})
        if entry.get("status") != "processed":
            raise ValueError(f"Complete processed source partition required: {month}")
        if entry["input_rows"] != entry["accepted_rows"] + entry["rejected_rows"]:
            raise ValueError(f"Source reconciliation failed: {month}")
        folder = Path(entry["output_dir"])
        for name in ("hourly_rides.parquet", "validated_trips.parquet", "rejected_trips.csv"):
            path = folder / name
            digest = entry["output_sha256"][name]
            if fingerprint(path) != digest:
                raise ValueError(f"Source artifact changed: {path}")
            checked_files[path] = digest
        counts = entry["verification"]["start_month_counts"]
        if sum(counts.values()) != entry["accepted_rows"]:
            raise ValueError(f"Source start-month counts do not reconcile: {month}")
        for start_month in expected_months:
            expected_months[start_month] += counts.get(start_month, 0)
        sources.append({"source_archive_month": month, "source_url": entry["source"]["url"],
                        "source_archive_sha256": entry["source"].get("sha256"), "accepted_rows": entry["accepted_rows"],
                        "start_month_counts": counts, "hourly_sha256": entry["output_sha256"]["hourly_rides.parquet"]})
        tag = sql_literal(month)
        hourly_queries.append(f"SELECT {tag} AS source_archive_month,* FROM read_parquet({sql_literal(folder / 'hourly_rides.parquet')},hive_partitioning=false)")
        id_queries.extend([
            f"SELECT {tag} AS source_archive_month,trim(ride_id) AS ride_id FROM read_parquet({sql_literal(folder / 'validated_trips.parquet')},hive_partitioning=false)",
            f"SELECT {tag} AS source_archive_month,trim(coalesce(ride_id,'')) AS ride_id FROM read_csv({sql_literal(folder / 'rejected_trips.csv')},header=true,all_varchar=true)",
        ])
    with TemporaryDirectory(prefix="citypulse-baseline-spill-") as spill:
        with duckdb.connect(config={"memory_limit": "512MB", "threads": 1, "temp_directory": spill}) as connection:
            connection.execute("SET TimeZone='UTC'")
            connection.execute("SET preserve_insertion_order=false")
            connection.execute("CREATE VIEW source_ids AS " + " UNION ALL ".join(id_queries))
            # Grouping millions of mostly unique IDs exhausted 512 MB on real
            # data. Bounded hash joins can spill; only matching IDs are retained.
            pair_queries = [f"""SELECT a.ride_id FROM source_ids a JOIN source_ids b USING(ride_id)
                WHERE a.source_archive_month={sql_literal(left_month)}
                  AND b.source_archive_month={sql_literal(right_month)} AND a.ride_id<>''"""
                for left_month, right_month in combinations(SOURCE_MONTHS, 2)]
            # Execute pairs sequentially: a UNION plan kept multiple hash tables
            # alive concurrently and still exceeded the real-data memory cap.
            for pair_query in pair_queries:
                if connection.execute(pair_query + " LIMIT 1").fetchone() is not None:
                    raise ValueError("Found cross-archive duplicate IDs: use a duplicate-audited catalogue before aggregating")
            duplicate_count = 0
            connection.execute("CREATE VIEW source_hourly AS " + " UNION ALL ".join(hourly_queries))
            invalid = connection.execute("""SELECT count(*) FROM source_hourly WHERE
                hour_start IS NULL OR hour_start<>date_trunc('hour',hour_start)
                OR ride_count IS NULL OR ride_count<=0 OR ride_count<>floor(ride_count)""").fetchone()[0]
            duplicates = connection.execute("""SELECT count(*) FROM (SELECT source_archive_month,start_station_id,hour_start
                FROM source_hourly GROUP BY ALL HAVING count(*)>1)""").fetchone()[0]
            if invalid or duplicates:
                raise ValueError("Invalid hourly counts or duplicate source station-hour keys")
            totals = dict(connection.execute("SELECT source_archive_month,sum(ride_count) FROM source_hourly GROUP BY 1").fetchall())
            if any(totals[s["source_archive_month"]] != s["accepted_rows"] for s in sources):
                raise ValueError("Source hourly counts differ from accepted counts")
            left, right = START.tz_convert("UTC"), END.tz_convert("UTC")
            observed = connection.execute("""SELECT hour_start,sum(ride_count)::BIGINT AS ride_count
                FROM source_hourly WHERE hour_start>=? AND hour_start<? GROUP BY 1 ORDER BY 1""", [left, right]).df()
    citywide = prepare_citywide(observed.set_index("hour_start").ride_count, START, END)
    local_month = citywide.hour_start.dt.tz_convert(LOCAL_ZONE).dt.strftime("%Y-%m")
    actual_months = {k: int(v) for k, v in citywide.groupby(local_month).ride_count.sum().items()}
    if actual_months != expected_months or len(citywide) != 1440:
        raise ValueError("Calendar grid or monthly accepted-start reconciliation failed")
    if fingerprint(manifest_path) != manifest_digest or any(fingerprint(p) != d for p, d in checked_files.items()):
        raise ValueError("Source changed during benchmark preparation")
    report = {"manifest_sha256": manifest_digest, "sources": sources, "timezone": LOCAL_ZONE,
              "hours": len(citywide), "observed_ride_starts": int(citywide.ride_count.sum()),
              "local_start_month_counts": actual_months, "cross_archive_duplicate_ids": duplicate_count,
              "missing_citywide_hours": int(citywide.ride_count.isna().sum()), "zero_filled_hours": 0,
              "coverage_limit": "Observed accepted starts from three processed archives; archive coverage is not proof of all underlying trips."}
    return citywide, report
