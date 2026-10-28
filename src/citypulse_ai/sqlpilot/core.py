"""Small, fail-closed DuckDB analytical query core; no model framework imports."""
import json
from pathlib import Path
import subprocess
import signal
import sys
import time

import duckdb

TABLES = {
    'hourly_rides': ('start_station_id', 'hour_start', 'ride_count'),
    'validated_trips': ('started_at', 'ended_at', 'start_station_id', 'end_station_id', 'duration_seconds'),
}
FUNCTIONS = {'count', 'count_star', 'sum', 'avg', 'min', 'max', 'round', 'abs',
             'date_part', 'date_trunc', 'timezone', 'strftime', 'coalesce', 'lower', 'upper'}
CLASSES = {'COLUMN_REF', 'CONSTANT', 'FUNCTION', 'COMPARISON', 'CONJUNCTION',
           'BETWEEN', 'OPERATOR', 'CASE', 'CAST'}
CONFIG = {'memory_limit': '64MB', 'threads': 1, 'enable_external_access': False,
          'autoload_known_extensions': False, 'autoinstall_known_extensions': False,
          'max_temp_directory_size': '0B'}


class QueryRejected(ValueError):
    pass


def connect(path):
    if duckdb.__version__ != '1.5.6':
        raise QueryRejected('SQLPilot requires the tested DuckDB 1.5.6 parser contract')
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 32*1024**2:
        raise QueryRejected('Require an existing trusted sample database, at most 32 MiB')
    connection = duckdb.connect(str(path), read_only=True, config=CONFIG)
    connection.execute("SET TimeZone='UTC'")
    return connection


def schema(connection):
    result = {}
    for table, columns in TABLES.items():
        kind = connection.execute("SELECT table_type FROM information_schema.tables WHERE table_catalog=current_database() AND table_schema='main' AND table_name=?", [table]).fetchall()
        if kind != [('BASE TABLE',)]:
            raise QueryRejected('Allowlisted inputs must be physical main-schema tables, not external views')
        rows = connection.execute("SELECT column_name,data_type FROM information_schema.columns WHERE table_catalog=current_database() AND table_schema='main' AND table_name=?", [table]).fetchall()
        found = dict(rows)
        if not set(columns) <= found.keys(): raise QueryRejected('Approved sample schema differs')
        count = connection.execute(f'SELECT count(*) FROM main."{table}"').fetchone()[0]
        if count > 10000: raise QueryRejected('Use an approved small sample, at most 10,000 rows/table')
        result[table] = {'columns': {c: found[c] for c in columns}, 'sample_rows': count}
    return result


def validate(sql, connection):
    if not isinstance(sql, str) or not sql.strip() or len(sql.encode()) > 4096:
        raise QueryRejected('SQL must be nonempty and at most 4096 bytes')
    try:
        parsed = json.loads(connection.execute('SELECT json_serialize_sql(?)', [sql]).fetchone()[0])
    except duckdb.Error as error:
        raise QueryRejected('DuckDB could not parse SQL') from error
    if parsed.get('error') or len(parsed.get('statements', [])) != 1:
        raise QueryRejected('Require exactly one DuckDB SELECT statement')
    statement = parsed['statements'][0]; node = statement['node']
    if node['type'] != 'SELECT_NODE' or node.get('cte_map', {}).get('map'):
        raise QueryRejected('Only plain SELECT is supported; no CTEs or set operations')
    if statement.get('named_param_map') or len(node['select_list']) > 16:
        raise QueryRejected('No parameters; at most sixteen projected expressions')
    aliases = {e['alias'] for e in node['select_list'] if e.get('alias')}
    used = set()

    def table(ref):
        if ref['type'] == 'BASE_TABLE':
            name = ref['table_name']
            if (name not in TABLES or ref.get('schema_name') not in ('', 'main')
                    or ref.get('catalog_name') or ref.get('at_clause') or ref.get('sample') or ref.get('column_name_alias')):
                raise QueryRejected('Table, catalog or source not allowlisted')
            used.add(name)
        elif ref['type'] == 'JOIN':
            if ref.get('condition') is None: raise QueryRejected('Joins require an explicit condition')
            table(ref['left']); table(ref['right'])
        else:
            raise QueryRejected('No table functions, subqueries, external sources or implicit tables')
    table(node['from_table'])
    allowed_columns = {c for t in used for c in TABLES[t]}
    for name in used:
        actual_columns = {r[0].casefold() for r in connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_catalog=current_database() AND table_schema='main' AND table_name=?", [name]).fetchall()}
        if {a.casefold() for a in aliases} & (actual_columns - set(TABLES[name])):
            raise QueryRejected('Output aliases cannot shadow hidden source columns')

    def walk(value, allow_alias=False):
        if isinstance(value, list):
            for part in value: walk(part, allow_alias)
        elif isinstance(value, dict):
            kind = value.get('class')
            if kind and kind not in CLASSES:
                raise QueryRejected('Unsupported expression: '+kind)
            node_type = value.get('type')
            if isinstance(node_type, str) and node_type.endswith('_NODE') and value is not node:
                raise QueryRejected('Nested query nodes are not supported')
            if kind == 'COLUMN_REF' and value['column_names'][-1] not in (allowed_columns | aliases if allow_alias else allowed_columns):
                raise QueryRejected('Column is not allowlisted')
            if kind == 'FUNCTION':
                name = value['function_name']
                if name not in FUNCTIONS or value.get('catalog') or value.get('schema') not in ('', 'main', 'pg_catalog') or value.get('export_state'):
                    raise QueryRejected('Function not allowlisted')
                # A trusted source still must not shadow a builtin with a SQL macro.
                internal = connection.execute('SELECT DISTINCT internal FROM duckdb_functions() WHERE function_name=?', [name]).fetchall()
                if not internal or any(not row[0] for row in internal):
                    raise QueryRejected('Non-builtin functions are forbidden')
            for key, part in value.items():
                walk(part, allow_alias or (key == 'modifiers'))
    walk(node)
    canonical = connection.execute('SELECT json_deserialize_sql(?)', [json.dumps(parsed)]).fetchone()[0]
    return canonical.rstrip().rstrip(';'), sorted(used)


def worker(database, request):
    with connect(database) as connection:
        if request.get('action') == 'schema': return {'schema': schema(connection), 'scope': 'sample only, not complete historical totals'}
        schema(connection)
        sql, tables = validate(request['sql'], connection)
        cursor = connection.execute('SELECT * FROM (\n'+sql+'\n) AS approved_result LIMIT 101')
        rows = cursor.fetchall(); truncated = len(rows) > 100
        clean = [[v.isoformat() if hasattr(v, 'isoformat') else v for v in row] for row in rows[:100]]
        result = {'sql': sql, 'columns': [d[0] for d in cursor.description], 'rows': clean,
                  'truncated': truncated, 'tables': tables, 'row_limit': 100,
                  'explanation': 'Returned '+str(min(len(rows),100))+' rows from the approved sample. Source: accepted-record sample only; not complete historical or physical demand totals. Output units follow the selected columns.'}
        if len(json.dumps(result, default=str).encode()) > 128*1024:
            raise QueryRejected('Result exceeds the 128 KiB response budget')
        return result


class SafeDatabase:
    def __init__(self, path, timeout=5):
        self.path = str(Path(path).absolute())
        if not 0 < timeout <= 10: raise ValueError('Timeout must be in (0,10] seconds')
        self.timeout = timeout

    def _call(self, request):
        started = time.monotonic()
        process = subprocess.Popen([sys.executable, '-m', 'citypulse_ai.sqlpilot.core', self.path],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, _ = process.communicate(json.dumps(request), timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            process.kill(); process.communicate()
            raise QueryRejected('Query process exceeded its wall-clock timeout') from error
        except BaseException:
            process.kill(); process.communicate()
            raise
        if process.returncode: raise QueryRejected('Query worker failed safely')
        value = json.loads(out)
        if 'error' in value: raise QueryRejected(value['error'])
        value['latency_seconds'] = time.monotonic()-started
        return value

    def discover(self): return self._call({'action': 'schema'})
    def query(self, sql): return self._call({'action': 'query', 'sql': sql})


if __name__ == '__main__':
    try:
        signal.alarm(5)  # independent hard deadline even if the MCP parent exits
        result = worker(sys.argv[1], json.loads(sys.stdin.read(8192)))
        print(json.dumps(result, default=str, allow_nan=False))
    except QueryRejected as error: print(json.dumps({'error': str(error)}))
    except Exception: print(json.dumps({'error': 'SQL binding/execution failed; correct the question or SQL against the permitted schema'}))
