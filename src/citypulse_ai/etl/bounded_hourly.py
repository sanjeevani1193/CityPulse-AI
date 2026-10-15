"""Exact disjoint UTC-day aggregation with durable, checked restart points."""

from datetime import datetime, timedelta

from citypulse_ai.etl.history_catalog import sql_literal


def build_hourly(connection, checkpoints) -> tuple[int, int, int]:
    stage = checkpoints.load('canonical-subtotals')
    if stage is None:
        days = connection.execute("SELECT CAST(date_trunc('day',started_at) AS VARCHAR),count(*) "
                                  "FROM canonical_trips GROUP BY 1 ORDER BY 1").fetchall()
        local = connection.execute("SELECT source_archive_month,start_month,count(*) "
                                   "FROM canonical_trips GROUP BY 1,2 ORDER BY 1,2").fetchall()
        stage = checkpoints.save('canonical-subtotals', {
            'days': days, 'source_start_month_counts': local,
            'status': 'verified_subtotals_not_final_canonical_results'})
    paths, total_rows, total_starts = [], 0, 0
    for index, (start, expected_starts) in enumerate(stage['data']['days']):
        stop = (datetime.fromisoformat(start) + timedelta(days=1)).isoformat()
        lower, upper = sql_literal(start), sql_literal(stop)
        name = f'hourly-day-{index:05d}'
        saved = checkpoints.load(name)
        if saved is None:
            path = checkpoints.asset(name)
            connection.execute(f"""COPY (
                SELECT start_month,start_station_id,date_trunc('hour',started_at) AS hour_start,
                       count(*)::BIGINT AS ride_count
                FROM canonical_trips
                WHERE started_at >= TIMESTAMPTZ {lower} AND started_at < TIMESTAMPTZ {upper}
                GROUP BY start_month,start_station_id,hour_start
            ) TO {sql_literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        else:
            if saved['data']['start'] != start or len(saved['files']) != 1:
                raise ValueError('Hourly partition identity differs')
            path = checkpoints.root / saved['files'][0]['name']
        # Exact uniqueness is bounded to one UTC day. Distinct days cannot share
        # an elapsed UTC hour, even on 23/25-hour NYC calendar days.
        rows, starts, unique, invalid = connection.execute(f"""
            SELECT count(*),coalesce(sum(ride_count),0),
                   count(DISTINCT(start_station_id,hour_start)),
                   count(*) FILTER(WHERE hour_start IS NULL OR start_station_id IS NULL
                     OR trim(start_station_id)='' OR ride_count <= 0
                     OR hour_start < TIMESTAMPTZ {lower} OR hour_start >= TIMESTAMPTZ {upper}
                     OR hour_start <> date_trunc('hour',hour_start)
                     OR start_month IS DISTINCT FROM
                        strftime(hour_start AT TIME ZONE 'America/New_York','%Y-%m'))
            FROM read_parquet({sql_literal(path)}, hive_partitioning=false)
        """).fetchone()
        if rows != unique or starts != expected_starts or invalid:
            raise ValueError('Hourly partition count, key or UTC-day reconciliation failed')
        data = {'start': start, 'stop': stop, 'rows': rows, 'ride_starts': starts}
        if saved is None:
            checkpoints.save(name, data, [path])
        elif saved['data'] != data:
            raise ValueError('Hourly partition metadata differs')
        paths.append(sql_literal(path))
        total_rows += rows
        total_starts += starts
        checkpoints.guard()
    if paths:
        connection.execute('CREATE VIEW hourly_rides AS SELECT * FROM read_parquet(['
                           + ','.join(paths) + '], hive_partitioning=false)')
    else:
        connection.execute("CREATE VIEW hourly_rides AS SELECT NULL::VARCHAR start_month, "
                           "NULL::VARCHAR start_station_id,NULL::TIMESTAMPTZ hour_start,"
                           "0::BIGINT ride_count WHERE false")
    observed = connection.execute('SELECT count(*),coalesce(sum(ride_count),0) FROM hourly_rides').fetchone()
    if observed != (total_rows, total_starts):
        raise ValueError('Hourly partition union reconciliation failed')
    return total_rows, total_starts, total_rows
