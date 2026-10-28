"""Hand-authored SQL execution evaluation, explicitly NOT LLM accuracy."""
import statistics
from citypulse_ai.sqlpilot.core import SafeDatabase, connect, QueryRejected

# Reference SQL is fixed trusted evaluation code, not supplied by a caller/model.
CASES = [
    ('Total accepted sample starts', 'SELECT sum(ride_count) AS starts FROM hourly_rides',
     'SELECT count(*) FROM validated_trips'),
    ('Three busiest sample start stations', 'SELECT start_station_id,sum(ride_count) AS rides FROM hourly_rides GROUP BY 1 ORDER BY rides DESC,start_station_id LIMIT 3',
     'SELECT start_station_id,count(*) AS n FROM validated_trips GROUP BY 1 ORDER BY n DESC,start_station_id LIMIT 3'),
    ('Average sample trip duration in seconds', 'SELECT avg(duration_seconds) FROM validated_trips',
     'SELECT sum(duration_seconds)/count(*) FROM validated_trips'),
    ('Starts by NYC-local weekday', "SELECT date_part('dow',hour_start AT TIME ZONE 'America/New_York') AS weekday,sum(ride_count) FROM hourly_rides GROUP BY 1 ORDER BY 1",
     "SELECT date_part('dow',started_at AT TIME ZONE 'America/New_York'),count(*) FROM validated_trips GROUP BY 1 ORDER BY 1"),
    ('Trips joined to their exact station-hour', "SELECT count(*) FROM hourly_rides h JOIN validated_trips t ON h.start_station_id=t.start_station_id AND h.hour_start=date_trunc('hour',t.started_at)",
     'SELECT count(*) FROM validated_trips'),
]
UNSAFE = ['DELETE FROM hourly_rides','SELECT 1; SELECT 2',
          "SELECT * FROM read_csv('/etc/passwd')",'SELECT table_name FROM information_schema.tables',
          'SELECT source_file AS source_file FROM validated_trips',"COPY hourly_rides TO '/tmp/not-allowed.csv'"]


def evaluate(path):
    db = SafeDatabase(path); details = []; durations = []
    for question, sql, reference in CASES:
        try:
            result = db.query(sql); durations.append(result['latency_seconds'])
            with connect(path) as connection: expected = [list(r) for r in connection.execute(reference).fetchall()]
            details.append({'question': question, 'execution_success': True, 'correct': result['rows'] == expected,
                            'latency_seconds': result['latency_seconds']})
        except QueryRejected:
            details.append({'question': question, 'execution_success': False, 'correct': False})
    rejected = 0
    for sql in UNSAFE:
        try: db.query(sql)
        except QueryRejected: rejected += 1
    return {'evaluation_type': 'hand-authored SQL vs independent fixed reference SQL; no LLM exercised',
            'cases': details, 'execution_successes': sum(x['execution_success'] for x in details),
            'correct_results': sum(x['correct'] for x in details), 'total_cases': len(CASES),
            'unsafe_rejected': rejected, 'unsafe_total': len(UNSAFE),
            'median_query_seconds': statistics.median(durations) if durations else None,
            'max_query_seconds': max(durations) if durations else None}
