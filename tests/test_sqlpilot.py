"""Synthetic sample, native DuckDB parser and real isolated readonly execution."""
import json
from pathlib import Path
import subprocess
import sys

import duckdb
import pytest
from citypulse_ai.sqlpilot.core import SafeDatabase, QueryRejected, connect, schema, validate
from citypulse_ai.sqlpilot.agent import SQLPilot
from citypulse_ai.sqlpilot.mcp_server import Server


@pytest.fixture
def sample(tmp_path):
    path=tmp_path/'sample.duckdb'
    with duckdb.connect(str(path), config={'threads':1}) as c:
        c.execute("CREATE TABLE hourly_rides(start_station_id VARCHAR,hour_start TIMESTAMPTZ,ride_count BIGINT)")
        c.execute("INSERT INTO hourly_rides VALUES ('001','2024-01-01 05:00:00+00',2),('002','2024-01-01 06:00:00+00',3),('001','2024-01-02 05:00:00+00',1)")
        c.execute('CREATE TABLE validated_trips(started_at TIMESTAMPTZ,ended_at TIMESTAMPTZ,start_station_id VARCHAR,end_station_id VARCHAR,duration_seconds DOUBLE,source_file VARCHAR)')
        c.execute("INSERT INTO validated_trips VALUES ('2024-01-01 05:10:00+00','2024-01-01 05:15:00+00','001','002',300,'private-path')")
    return path


@pytest.mark.parametrize('sql,expected',[
    ("SELECT sum(ride_count) AS rides FROM hourly_rides",[[6]]),
    ("SELECT start_station_id,sum(ride_count) AS rides FROM hourly_rides GROUP BY 1 ORDER BY rides DESC,start_station_id LIMIT 1",[['001',3]]),
    ("SELECT ride_count FROM hourly_rides WHERE start_station_id='001' ORDER BY hour_start",[[2],[1]]),
    ("SELECT date_part('dow',hour_start AT TIME ZONE 'America/New_York') AS weekday,sum(ride_count) FROM hourly_rides GROUP BY 1 ORDER BY 1",[[1,5],[2,1]]),
    ("SELECT h.ride_count,t.duration_seconds FROM hourly_rides h JOIN validated_trips t ON h.start_station_id=t.start_station_id AND h.hour_start=date_trunc('hour',t.started_at)",[[2,300.]]),
])
def test_reference_queries(sample,sql,expected):
    result=SafeDatabase(sample).query(sql)
    assert result['rows']==expected and not result['truncated'] and result['latency_seconds']>0


@pytest.mark.parametrize('sql',[
    'DROP TABLE hourly_rides','DELETE FROM hourly_rides','UPDATE hourly_rides SET ride_count=0',
    'SELECT 1; SELECT 2',"COPY hourly_rides TO '/tmp/forbidden.csv'",'INSTALL httpfs','LOAD httpfs',
    "ATTACH '/tmp/other.duckdb' AS x",'PRAGMA version',"SET enable_external_access=true",
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM read_parquet('https://example.com/data')",
    'SELECT table_name FROM information_schema.tables','SELECT * FROM hourly_rides',
    'SELECT source_file FROM validated_trips',
    'SELECT source_file AS source_file FROM validated_trips',
    'SELECT source_file AS start_station_id FROM validated_trips',
    'SELECT start_station_id AS SOURCE_FILE FROM validated_trips ORDER BY lower(SOURCE_FILE)',
    'SELECT start_station_id FROM validated_trips t(a,b,c,d,e,start_station_id)',"SELECT current_setting('access_mode') FROM hourly_rides",
    'SELECT query(\'DELETE FROM hourly_rides\') FROM hourly_rides',
    'WITH x AS (SELECT * FROM hourly_rides) SELECT * FROM x',
    'SELECT ride_count FROM other.hourly_rides',
    'SELECT (SELECT source_file FROM validated_trips LIMIT 1) FROM hourly_rides',
    "SELECT repeat('x',1000000000) FROM hourly_rides",'SELECT ride_count FROM hourly_rides UNION SELECT 1',
])
def test_unsafe_queries_fail_closed(sample,sql):
    with pytest.raises(QueryRejected): SafeDatabase(sample).query(sql)
    with connect(sample) as c: assert c.execute('SELECT sum(ride_count) FROM hourly_rides').fetchone()[0]==6


def test_invalid_sql_unknown_columns_and_readonly(sample):
    for sql in ['SELECT missing FROM hourly_rides','not SQL','SELECT ride_count FROM missing','x'*4097]:
        with pytest.raises(QueryRejected): SafeDatabase(sample).query(sql)
    with connect(sample) as c:
        with pytest.raises(duckdb.Error): c.execute('CREATE TABLE forbidden(x INT)')
        with pytest.raises(duckdb.Error): c.execute("SELECT * FROM read_csv('/etc/passwd')")
        assert schema(c)['hourly_rides']['sample_rows']==3


def test_timeout_kills_worker(sample):
    with pytest.raises(QueryRejected,match='timeout'): SafeDatabase(sample,timeout=.001).query('SELECT sum(ride_count) FROM hourly_rides')


def test_row_limit(sample):
    with duckdb.connect(str(sample)) as c:
        c.execute("INSERT INTO hourly_rides SELECT '003',TIMESTAMPTZ '2024-01-01 00:00:00+00',1 FROM range(200)")
    result=SafeDatabase(sample).query('SELECT ride_count FROM hourly_rides')
    assert result['truncated'] and len(result['rows'])==100


def test_macro_shadowing_and_views_rejected(sample):
    with duckdb.connect(str(sample)) as c:c.execute('CREATE MACRO sum(x) AS 0')
    with pytest.raises(QueryRejected,match='builtin'): SafeDatabase(sample).query('SELECT sum(ride_count) FROM hourly_rides')
    with duckdb.connect(str(sample)) as c:
        c.execute('DROP TABLE hourly_rides');c.execute('CREATE VIEW hourly_rides AS SELECT 1 AS ride_count')
    with pytest.raises(QueryRejected,match='physical'): SafeDatabase(sample).discover()


def test_agent_no_llm_no_fabricated_answer(sample):
    agent=SQLPilot(SafeDatabase(sample))
    assert agent.ask('How many starts?')['status']=='backend_unavailable'
    class Clarification:
        def generate(self,q,s):return {'clarification':'Which period?'}
    assert SQLPilot(SafeDatabase(sample),Clarification()).ask('busiest?')['status']=='clarification'
    class Unsafe:
        def generate(self,q,s):return {'sql':'DELETE FROM hourly_rides'}
    with pytest.raises(QueryRejected):SQLPilot(SafeDatabase(sample),Unsafe()).ask('malicious')


def test_mcp_lifecycle_and_unsafe_tool(sample):
    server=Server(SafeDatabase(sample))
    def call(method,params={}):return server.handle({'jsonrpc':'2.0','id':1,'method':method,'params':params})
    assert 'error' in call('tools/list')
    assert call('initialize',{'protocolVersion':'2025-03-26','clientInfo':{'name':'test','version':'1'},'capabilities':{}})['result']['protocolVersion']=='2025-03-26'
    assert server.handle({'jsonrpc':'2.0','method':'notifications/initialized'}) is None
    assert len(call('tools/list')['result']['tools'])==2
    assert call('tools/call',{'name':'sqlpilot_query','arguments':{'sql':'DELETE FROM hourly_rides'}})['result']['isError']
    r=call('tools/call',{'name':'sqlpilot_query','arguments':{'sql':'SELECT sum(ride_count) FROM hourly_rides'}})
    assert json.loads(r['result']['content'][0]['text'])['rows']==[[6]]
    assert 'error' in call('tools/call',{'name':'sqlpilot_query','arguments':{'sql':'x','database':'/etc/passwd'}})


def test_stdio_mcp_real_process(sample):
    messages=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-03-26','clientInfo':{'name':'test'},'capabilities':{}}},
              {'jsonrpc':'2.0','method':'notifications/initialized'}, {'jsonrpc':'2.0','id':2,'method':'tools/list'}]
    p=subprocess.run([sys.executable,'-m','citypulse_ai.sqlpilot.mcp_server','--database',str(sample)],input='\n'.join(map(json.dumps,messages))+'\n',text=True,capture_output=True,timeout=10)
    assert p.returncode==0 and len(p.stdout.splitlines())==2
    assert json.loads(p.stdout.splitlines()[1])['result']['tools'][0]['name']=='sqlpilot_schema'


def test_dst_fold_and_utc_order(sample):
    with duckdb.connect(str(sample)) as c:
        c.execute("INSERT INTO hourly_rides VALUES ('001','2024-11-03 05:00:00+00',2),('001','2024-11-03 06:00:00+00',5)")
    r=SafeDatabase(sample).query("SELECT date_part('hour',hour_start AT TIME ZONE 'America/New_York') AS local_hour,sum(ride_count) FROM hourly_rides WHERE hour_start>=TIMESTAMPTZ '2024-11-03 00:00:00+00' GROUP BY 1")
    assert r['rows']==[[1,7]]


def test_execution_evaluation_is_not_llm_accuracy(sample):
    from citypulse_ai.sqlpilot.evaluation import evaluate
    result=evaluate(sample)
    # The synthetic hourly fixture intentionally has more rides than its one-trip
    # validated table: independent references must expose that inconsistency.
    assert result['execution_successes']==5 and result['correct_results']<5
    assert result['unsafe_rejected']==6 and 'no LLM' in result['evaluation_type']


def test_parser_version_fails_closed(sample,monkeypatch):
    monkeypatch.setattr(duckdb,'__version__','unknown')
    with pytest.raises(QueryRejected,match='parser contract'):connect(sample)


def test_database_symlink_is_not_resolved_around_policy(sample,tmp_path):
    link=tmp_path/'alias.duckdb';link.symlink_to(sample)
    with pytest.raises(QueryRejected,match='trusted sample'):SafeDatabase(link).discover()
