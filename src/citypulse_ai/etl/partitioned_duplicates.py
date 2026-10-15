"""Exact global duplicate aggregation with bounded, disposable ID projections."""

from contextlib import ExitStack
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from citypulse_ai.etl.archives import check_disk
from citypulse_ai.etl.history_catalog import sql_literal


def build_partitioned_duplicates(connection, workspace: Path, *, partition_count: int,
                                 disk_budget_bytes: int, reserve_bytes: int,
                                 batch_size: int = 50_000) -> dict:
    """MD5 routes IDs only; full trimmed strings define equality, never hashes.

    Include accepted and rejected occurrences from all archives. Project once,
    then aggregate one partition at a time with precisely the original SQL.
    Oversized/skewed partitions fail safely at the existing DuckDB memory limit.
    No persistent inputs, manifests or catalogues are modified.
    """
    if partition_count < 1 or batch_size < 1 or disk_budget_bytes < 1:
        raise ValueError("Partition count, batch size and disk budget must be positive")
    workspace.mkdir()
    check_disk(workspace, disk_budget_bytes, reserve_bytes)
    reader = connection.execute(f"""
        SELECT normalized_ride_id, source_archive_month,
               CAST('0x' || substr(md5(normalized_ride_id),1,16) AS UBIGINT)
                   % {partition_count} AS bucket
        FROM raw_ride_occurrences WHERE normalized_ride_id <> ''
    """).to_arrow_reader(batch_size=batch_size)
    writers, paths, rows = {}, {}, 0
    with ExitStack() as stack:
        stack.callback(reader.close)
        for batch in reader:
            rows += batch.num_rows
            for bucket in pc.unique(batch.column("bucket")).to_pylist():
                subset = batch.filter(pc.equal(batch.column("bucket"), bucket)).select(
                    ["normalized_ride_id", "source_archive_month"])
                if bucket not in writers:
                    paths[bucket] = workspace / f"ids-{bucket}.parquet"
                    writers[bucket] = stack.enter_context(pq.ParquetWriter(
                        paths[bucket], subset.schema, compression="zstd"))
                writers[bucket].write_batch(subset)
            # Leave room for Parquet footers and check the reserve every batch.
            used = sum(path.stat().st_size for path in paths.values())
            if used + len(paths) * 64 * 1024 > disk_budget_bytes:
                raise ValueError("Duplicate partition scratch exceeded its disk budget")
            check_disk(workspace, 0, reserve_bytes)
    used = sum(path.stat().st_size for path in paths.values())
    if used > disk_budget_bytes:
        raise ValueError("Duplicate partition scratch exceeded its disk budget")
    connection.execute("CREATE TABLE duplicate_ids (normalized_ride_id VARCHAR, "
                       "occurrences BIGINT, archive_count BIGINT)")
    for bucket, path in sorted(paths.items()):
        check_disk(workspace, 0, reserve_bytes)
        connection.execute(f"""INSERT INTO duplicate_ids
            SELECT normalized_ride_id,count(*),count(DISTINCT source_archive_month)
            FROM read_parquet({sql_literal(path)}, hive_partitioning=false)
            GROUP BY normalized_ride_id HAVING count(*) > 1""")
    return {"strategy": "md5_routed_full_id_grouping", "partition_count": partition_count,
            "projected_occurrences": rows, "partition_bytes": used,
            "partition_disk_budget_bytes": disk_budget_bytes}
