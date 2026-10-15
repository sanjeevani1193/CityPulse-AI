"""Small exact comparisons, DST boundaries and restartable daily aggregation."""

import duckdb
import pytest

from citypulse_ai.etl.bounded_hourly import build_hourly
from citypulse_ai.etl.reconciliation_checkpoints import Checkpoints


def seed(connection):
    connection.execute("SET TimeZone='UTC'")
    connection.execute("CREATE TABLE canonical_trips(started_at TIMESTAMPTZ,"
                       "start_station_id VARCHAR,source_archive_month VARCHAR)")
    connection.executemany("INSERT INTO canonical_trips VALUES (?,?,?)", [
        ('2024-06-01T03:59:59.123456Z', '001.01_', '2024-05'),
        ('2024-06-01T03:59:59.654321Z', '001.01_', '2024-06'),
        ('2024-06-01T04:00:00Z', '001.01_', '2024-06'),
        ('2024-03-10T06:30:00Z', '007', '2024-03'),
        ('2024-03-10T07:30:00Z', '007', '2024-03'),
        ('2024-11-03T05:30:00Z', '007', '2024-11'),
        ('2024-11-03T06:30:00Z', '007', '2024-11'),
    ])
    connection.execute("ALTER TABLE canonical_trips ADD COLUMN start_month VARCHAR")
    connection.execute("UPDATE canonical_trips SET start_month="
                       "strftime(started_at AT TIME ZONE 'America/New_York','%Y-%m')")


def test_boundaries_dst_exact_schema_and_original_algorithm(tmp_path):
    with duckdb.connect(config={'memory_limit': '512MB'}) as connection:
        seed(connection)
        expected = connection.execute("SELECT start_month,start_station_id,"
            "date_trunc('hour',started_at) hour_start,count(*)::BIGINT ride_count "
            "FROM canonical_trips GROUP BY 1,2,3 ORDER BY 1,2,3").fetchall()
        checkpoints = Checkpoints(tmp_path / 'checkpoints', 'fixture', 16*1024**2, 0)
        summary = build_hourly(connection, checkpoints)
        assert summary == (6, 7, 6)
        assert connection.execute('SELECT * FROM hourly_rides ORDER BY 1,2,3').fetchall() == expected
        assert connection.execute("SELECT sum(ride_count) FROM hourly_rides "
                                  "WHERE start_month='2024-05'").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM hourly_rides WHERE "
            "strftime(hour_start AT TIME ZONE 'America/New_York','%Y-%m-%d %H')="
            "'2024-11-03 01'").fetchone()[0] == 2
        assert [r[1] for r in connection.execute('DESCRIBE hourly_rides').fetchall()] == [
            'VARCHAR', 'VARCHAR', 'TIMESTAMP WITH TIME ZONE', 'BIGINT']


def test_representative_daily_keys_and_exact_merging(tmp_path):
    with duckdb.connect(config={'memory_limit': '512MB', 'threads': 2}) as connection:
        connection.execute("SET TimeZone='UTC'")
        connection.execute("CREATE VIEW canonical_trips AS SELECT "
            "TIMESTAMPTZ '2024-06-01' + (i%120)*INTERVAL '1 hour' started_at,"
            "'station-' || CAST(i%2003 AS VARCHAR) start_station_id,"
            "'2024-06' source_archive_month,'2024-06' start_month FROM range(120000) t(i) "
            "WHERE i%120>=4")
        checkpoints = Checkpoints(tmp_path / 'checkpoints', 'fixture', 16*1024**2, 0)
        summary = build_hourly(connection, checkpoints)
        assert summary[1] == connection.execute('SELECT count(*) FROM canonical_trips').fetchone()[0]
        assert connection.execute("SELECT count(*) FROM ((SELECT start_month,start_station_id,"
            "date_trunc('hour',started_at) hour_start,count(*)::BIGINT ride_count "
            "FROM canonical_trips GROUP BY 1,2,3) EXCEPT ALL SELECT * FROM hourly_rides)").fetchone()[0] == 0


def test_corrupt_checkpoint_is_refused_without_overwrite(tmp_path):
    with duckdb.connect() as connection:
        seed(connection)
        checkpoints = Checkpoints(tmp_path / 'checkpoints', 'fixture', 16*1024**2, 0)
        build_hourly(connection, checkpoints)
        saved = checkpoints.load('hourly-day-00000')
        asset = checkpoints.root / saved['files'][0]['name']
        asset.write_bytes(b'corrupt; preserve for investigation')
        connection.execute('DROP VIEW hourly_rides')
        with pytest.raises(ValueError, match='checkpoint asset'):
            build_hourly(connection, checkpoints)
        assert asset.read_bytes() == b'corrupt; preserve for investigation'


def test_checkpoint_quota_and_identity(tmp_path):
    checkpoints = Checkpoints(tmp_path / 'checkpoints', 'fixture', 4096, 0)
    checkpoints.save('stage', {'verified': True})
    with pytest.raises(ValueError, match='Changed reconciliation checkpoint'):
        Checkpoints(checkpoints.root, 'other-inputs', 4096, 0).load('stage')
    (checkpoints.root / 'abandoned-scratch').write_bytes(b'x'*4096)
    with pytest.raises(ValueError, match='disk budget'):
        checkpoints.guard()


def test_resume_after_interrupted_day_preserves_completed_partitions(tmp_path, monkeypatch):
    with duckdb.connect(config={'memory_limit': '512MB'}) as connection:
        seed(connection)
        checkpoints = Checkpoints(tmp_path / 'checkpoints', 'fixture', 16*1024**2, 0)
        original = checkpoints.save
        def interrupted(name, data, files=()):
            if name == 'hourly-day-00001':
                raise KeyboardInterrupt('Stop between partition write and checkpoint')
            return original(name, data, files)
        monkeypatch.setattr(checkpoints, 'save', interrupted)
        with pytest.raises(KeyboardInterrupt):
            build_hourly(connection, checkpoints)
        saved = checkpoints.load('hourly-day-00000')
        path = checkpoints.root / saved['files'][0]['name']
        before = (path.stat().st_mtime_ns, path.read_bytes())
        monkeypatch.setattr(checkpoints, 'save', original)
        assert build_hourly(connection, checkpoints) == (6, 7, 6)
        assert (path.stat().st_mtime_ns, path.read_bytes()) == before
        files_before = set(checkpoints.root.iterdir())
        connection.execute('DROP VIEW hourly_rides')
        assert build_hourly(connection, checkpoints) == (6, 7, 6)
        assert set(checkpoints.root.iterdir()) == files_before
