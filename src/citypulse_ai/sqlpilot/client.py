"""Official MCP SDK client; tools always cross a real stdio boundary."""
from contextlib import asynccontextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolError(ValueError):
    pass


class Tools:
    def __init__(self, session):
        self.session = session; self.calls = 0; self.audit = []

    async def call(self, name, arguments):
        if name not in {'sqlpilot_schema', 'sqlpilot_query'} or self.calls >= 3:
            raise ToolError('Tool call budget exceeded or tool not permitted')
        self.calls += 1; started = time.monotonic()
        response = await self.session.call_tool(name, arguments=arguments)
        self.audit.append({'method': 'tools/call', 'tool': name, 'is_error': response.isError,
                           'latency_seconds': time.monotonic()-started})
        if len(response.content) != 1 or response.content[0].type != 'text':
            raise ToolError('Unexpected MCP content')
        text = response.content[0].text
        if len(text.encode()) > 128*1024: raise ToolError('MCP result too large')
        value = json.loads(text)
        if response.isError: raise ToolError(value.get('error', 'Query rejected'))
        if not isinstance(value, dict): raise ToolError('MCP result must be an object')
        return value


@asynccontextmanager
async def connect_tools(database):
    # Do not forward provider credentials, tracing keys or arbitrary subprocess config.
    root = Path(__file__).resolve().parents[2]
    interpreter = os.environ.get('SQLPILOT_MCP_PYTHON', sys.executable)
    if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
        raise ToolError('Operator MCP interpreter must be an existing absolute path')
    parameters = StdioServerParameters(command=interpreter,
        args=['-m', 'citypulse_ai.sqlpilot.mcp', '--database', str(Path(database).absolute())],
        env={'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1',
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=8)) as session:
            initialized = await session.initialize()
            listing = await session.list_tools()
            if {tool.name for tool in listing.tools} != {'sqlpilot_schema', 'sqlpilot_query'}:
                raise ToolError('Required MCP tools unavailable')
            tools = Tools(session)
            tools.audit.extend([{'method': 'initialize', 'protocol': initialized.protocolVersion},
                                {'method': 'tools/list', 'tools': sorted(tool.name for tool in listing.tools)}])
            yield tools
