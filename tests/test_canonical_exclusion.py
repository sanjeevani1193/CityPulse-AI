"""Equivalent exclusion and plans for exact full-ID lookup, without history grouping."""

import duckdb
import pytest

from citypulse_ai.etl.history_catalog import create_canonical_trip_view
from citypulse_ai.etl.temporary_reconciliation import validate_duplicate_result


@pytest.mark.parametrize('duplicates', [[], [('same', 3, 2), ('within', 2, 1)]])
def test_exact_precedence_schema_and_explain(duplicates):
    with duckdb.connect(config={'memory_limit': '512MB'}) as c:
        c.execute("CREATE TABLE locally_valid_trips(ride_id VARCHAR,start_station_id VARCHAR,"
                  "started_at TIMESTAMPTZ,source_archive_month VARCHAR)")
        c.executemany('INSERT INTO locally_valid_trips VALUES (?,?,?,?)', [
            ('same','001.01_','2024-06-01T03:59:59Z','2024-05'),
            (' same ','001.01_','2024-06-01T03:59:59Z','2024-06'),
            ('same','001.01_','2024-06-01T04:00:00Z','2024-06'),
            ('within','007','2024-06-02T04:00:00Z','2024-06'),
            ('within','007','2024-06-02T04:00:00Z','2024-06'),
            ('SAME','007','2024-06-02T04:00:00Z','2024-06'),
            ('unique','007','2024-11-03T05:30:00Z','2024-11'),
            ('unique-2','007','2024-11-03T06:30:00Z','2024-11'),
        ])
        c.execute('CREATE TABLE duplicate_ids(normalized_ride_id VARCHAR,occurrences BIGINT,archive_count BIGINT)')
        if duplicates:c.executemany('INSERT INTO duplicate_ids VALUES (?,?,?)',duplicates)
        validate_duplicate_result(c, {'duplicate_ride_ids':len(duplicates),
                                      'cross_archive_duplicate_ride_ids':int(bool(duplicates))})
        create_canonical_trip_view(c)
        query="SELECT start_month,start_station_id,date_trunc('hour',started_at),count(*) FROM canonical_trips GROUP BY 1,2,3 ORDER BY 1,2,3"
        expected=c.execute(query).fetchall()
        original_plan=c.execute('EXPLAIN '+query).fetchone()[1]
        assert 'DELIM' in original_plan
        schema=c.execute('DESCRIBE canonical_trips').fetchall()
        for mode in (['anti'] if duplicates else ['anti','verified_zero']):
            c.execute('DROP VIEW canonical_trips')
            create_canonical_trip_view(c,exclusion=mode)
            assert c.execute(query).fetchall()==expected
            assert c.execute('DESCRIBE canonical_trips').fetchall()==schema
            plan=c.execute('EXPLAIN '+query).fetchone()[1]
            assert 'DELIM' not in plan
            # Only station-hour GROUP BY remains; there is no ride-ID GROUP BY.
            assert plan.count('HASH_GROUP_BY')==1
            if duplicates:
                assert 'ANTI' in plan
                assert c.execute('SELECT ride_id FROM canonical_trips ORDER BY ride_id').fetchall()==[
                    ('SAME',),('unique',),('unique-2',)]


def test_false_zero_result_claim_is_rejected():
    with duckdb.connect() as c:
        c.execute("CREATE TABLE duplicate_ids AS SELECT 'id'::VARCHAR normalized_ride_id,"
                  "2::BIGINT occurrences,2::BIGINT archive_count")
        with pytest.raises(ValueError,match='integrity differ'):
            validate_duplicate_result(c, {'duplicate_ride_ids':0,'cross_archive_duplicate_ride_ids':0})
