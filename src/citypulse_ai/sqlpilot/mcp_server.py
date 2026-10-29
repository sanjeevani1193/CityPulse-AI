"""Bounded newline-delimited JSON-RPC stdio subset of MCP 2025-03-26.

No SDK dependency or network listener. A client owns its LLM; these are safe,
read-only database tools, not a claim of a configured text-to-SQL model.
"""
import argparse
import json
import sys
from citypulse_ai.sqlpilot.core import QueryRejected, SafeDatabase

TOOLS = [
    {'name': 'sqlpilot_schema', 'description': 'Discover permitted sample tables/columns and scope.',
     'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'name': 'sqlpilot_query', 'description': 'Execute one validated sample-only SELECT, capped at 100 rows.',
     'inputSchema': {'type': 'object', 'properties': {'sql': {'type': 'string', 'maxLength': 4096}},
                     'required': ['sql'], 'additionalProperties': False}},
]
for tool in TOOLS:
    tool['annotations'] = {'readOnlyHint': True, 'destructiveHint': False,
                           'idempotentHint': True, 'openWorldHint': False}


class Server:
    def __init__(self, database): self.database = database; self.initialized = False; self.ready = False

    def handle(self, message):
        if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
        identity = message.get('id'); method = message.get('method'); params = message.get('params', {})
        def error(code, text): return {'jsonrpc': '2.0', 'id': identity, 'error': {'code': code, 'message': text}}
        if 'id' not in message:
            if method == 'notifications/initialized' and self.initialized: self.ready = True
            return None
        if not isinstance(params, dict): return error(-32602, 'Parameters must be an object')
        if method == 'initialize':
            if self.initialized: return error(-32600, 'Already initialized')
            if not isinstance(params.get('clientInfo'), dict) or not isinstance(params.get('capabilities'), dict) or not isinstance(params.get('protocolVersion'), str):
                return error(-32602, 'Initialization metadata required')
            self.initialized = True
            result = {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'citypulse-sqlpilot-core', 'version': '0.1.0'}}
        elif method == 'ping': result = {}
        elif not self.ready: return error(-32002, 'Initialize and send notifications/initialized first')
        elif method == 'tools/list': result = {'tools': TOOLS}
        elif method == 'tools/call':
            name = params.get('name'); args = params.get('arguments', {})
            if not isinstance(args, dict): return error(-32602, 'Arguments must be an object')
            if name == 'sqlpilot_schema' and args == {}: function = self.database.discover; values = ()
            elif name == 'sqlpilot_query' and set(args) == {'sql'} and isinstance(args['sql'], str): function = self.database.query; values = (args['sql'],)
            else: return error(-32602, 'Unknown tool or invalid arguments')
            try:
                value = function(*values); failed = False
            except QueryRejected as exc:
                value = {'error': str(exc)}; failed = True
            result = {'content': [{'type': 'text', 'text': json.dumps(value)}], 'isError': failed}
        else: return error(-32601, 'Method not supported')
        return {'jsonrpc': '2.0', 'id': identity, 'result': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, help='Operator-owned trusted small DuckDB snapshot')
    server = Server(SafeDatabase(parser.parse_args().database))
    for _ in range(128):  # bounded session; clients reconnect explicitly
        line = sys.stdin.readline(8193)
        if not line: break
        if len(line.encode()) > 8192: break  # fail closed; never accumulate an unbounded message
        try: reply = server.handle(json.loads(line))
        except (ValueError, TypeError): reply = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid JSON'}}
        if reply is not None: print(json.dumps(reply), flush=True)


if __name__ == '__main__': main()
