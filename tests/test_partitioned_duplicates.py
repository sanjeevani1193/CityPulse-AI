"""Partition routing must never replace full-ID equality or historical scope."""

import duckdb
import pytest

from citypulse_ai.etl.partitioned_duplicates import build_partitioned_duplicates


@pytest.mark.parametrize("partitions", [1, 2, 16])
def test_exact_ids_nonadjacent_archives_and_rejected_occurrences(tmp_path, partitions):
    with duckdb.connect(config={"memory_limit": "512MB"}) as connection:
        connection.execute("CREATE TABLE raw_ride_occurrences "
                           "(normalized_ride_id VARCHAR,source_archive_month VARCHAR)")
        connection.executemany("INSERT INTO raw_ride_occurrences VALUES (?,?)", [
            ("same", "2024-01"), ("same", "2025-03"),
            ("within", "2024-06"), ("within", "2024-06"),
            ("unique", "2024-05"), ("UNIQUE", "2024-06"),
            ("", "2024-01"), ("", "2025-03"),
        ])
        expected = connection.execute("SELECT normalized_ride_id,count(*),"
            "count(DISTINCT source_archive_month) FROM raw_ride_occurrences "
            "WHERE normalized_ride_id <> '' GROUP BY 1 HAVING count(*)>1 ORDER BY 1").fetchall()
        execution = build_partitioned_duplicates(connection, tmp_path / "ids",
            partition_count=partitions, disk_budget_bytes=4 * 1024**2,
            reserve_bytes=0, batch_size=2)
        assert connection.execute("SELECT * FROM duplicate_ids ORDER BY 1").fetchall() == expected
        assert expected == [("same", 2, 2), ("within", 2, 1)]
        assert execution["projected_occurrences"] == 6
        assert execution["partition_bytes"] < 4 * 1024**2


def test_disk_budget_failure_cannot_produce_duplicate_results(tmp_path):
    with duckdb.connect() as connection:
        connection.execute("CREATE VIEW raw_ride_occurrences AS "
                           "SELECT 'ride' normalized_ride_id,'2024-06' source_archive_month")
        with pytest.raises(ValueError, match="disk budget"):
            build_partitioned_duplicates(connection, tmp_path / "ids",
                partition_count=1, disk_budget_bytes=1, reserve_bytes=0)
        assert connection.execute("SELECT count(*) FROM duckdb_tables() "
                                  "WHERE table_name='duplicate_ids'").fetchone()[0] == 0
