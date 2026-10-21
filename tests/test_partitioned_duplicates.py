"""Partition routing must never replace full-ID equality or historical scope."""

import duckdb
import pytest
import weakref

from citypulse_ai.etl.partitioned_duplicates import build_partitioned_duplicates
from citypulse_ai.etl import partitioned_duplicates


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


@pytest.mark.parametrize('archives', [1, 8, 24])
@pytest.mark.parametrize('duplicates', [False, True])
def test_projection_writers_released_before_exact_aggregation(tmp_path, monkeypatch, archives, duplicates):
    """Closed writer metadata must not overlap the next phase as history grows."""
    writers = []
    constructor = partitioned_duplicates.pq.ParquetWriter
    def observe_writer(*args, **kwargs):
        writer = constructor(*args, **kwargs)
        writers.append(weakref.ref(writer))
        return writer
    monkeypatch.setattr(partitioned_duplicates.pq, 'ParquetWriter', observe_writer)
    class Connection:
        def __init__(self, native):
            self.native, self.partitions = native, 0
        def execute(self, sql, *args):
            if sql.lstrip().startswith('INSERT INTO duplicate_ids'):
                self.partitions += 1
                assert writers and all(ref() is None for ref in writers)
            return self.native.execute(sql, *args)
    with duckdb.connect(config={'memory_limit': '512MB', 'threads': 2}) as native:
        native.execute("CREATE TABLE raw_ride_occurrences AS SELECT "
            "lpad(i::VARCHAR,16,'0') normalized_ride_id, "
            "'source-'||(i%?)::VARCHAR source_archive_month FROM range(?) r(i)",
            [archives, archives*100])
        if duplicates:
            native.execute("INSERT INTO raw_ride_occurrences VALUES "
                "('0000000000000000','source-last'),('within','source-0'),('within','source-0')")
        expected = native.execute("SELECT normalized_ride_id,count(*),count(DISTINCT source_archive_month) "
            "FROM raw_ride_occurrences GROUP BY 1 HAVING count(*)>1 ORDER BY 1").fetchall()
        connection = Connection(native)
        result = build_partitioned_duplicates(connection, tmp_path/'ids', partition_count=8,
            disk_budget_bytes=4*1024**2, reserve_bytes=0, batch_size=64)
        assert connection.partitions == 8
        assert result['projected_occurrences'] == archives*100 + (3 if duplicates else 0)
        assert native.execute('SELECT * FROM duplicate_ids ORDER BY 1').fetchall() == expected


@pytest.mark.parametrize('buffer_limit', [256, 64*1024**2])
def test_tiny_batches_coalesce_with_bounded_buffers_and_exact_counts(tmp_path, monkeypatch, buffer_limit):
    monkeypatch.setattr(partitioned_duplicates, 'ID_BUFFER_BYTES', buffer_limit)
    with duckdb.connect(config={'memory_limit': '512MB', 'threads': 2}) as c:
        c.execute("CREATE VIEW raw_ride_occurrences AS SELECT "
            "lpad((i%1499)::VARCHAR,16,'0') normalized_ride_id, "
            "'source-'||(i%24)::VARCHAR source_archive_month FROM range(3000) r(i)")
        expected = c.execute("SELECT normalized_ride_id,count(*),count(DISTINCT source_archive_month) "
            "FROM raw_ride_occurrences GROUP BY 1 HAVING count(*)>1 ORDER BY 1").fetchall()
        result = build_partitioned_duplicates(c, tmp_path/'ids', partition_count=8,
            disk_budget_bytes=4*1024**2, reserve_bytes=0, batch_size=17)
        assert result['projected_occurrences'] == 3000
        assert c.execute('SELECT * FROM duplicate_ids ORDER BY 1').fetchall() == expected
        metadata = [partitioned_duplicates.pq.read_metadata(p) for p in (tmp_path/'ids').glob('*.parquet')]
        assert sum(m.num_rows for m in metadata) == 3000
        if buffer_limit == 64*1024**2:
            assert sum(m.num_row_groups for m in metadata) == 8
