"""Container-only HTTP/UI/MCP check with controlled generation, never an LLM API.

Run through `docker compose exec -T sqlpilot python - < docker/verify_sqlpilot.py`.
No keys are printed; the check refuses a configured backend before opening a UI.
"""
import asyncio
import json
import os
import urllib.request

from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.asyncio.client import connect

from citypulse_ai.sqlpilot.client import connect_tools, ToolError
from citypulse_ai.sqlpilot.graph import run_agent
from citypulse_ai.sqlpilot.providers import configured_backend


class ControlledGeneration:
    is_live = False

    def __init__(self, *decisions):
        self.decisions = iter(decisions)

    def generate(self, question, schema):
        return next(self.decisions)


async def browser_check():
    async with connect('ws://127.0.0.1:8501/_stcore/stream',
                       origin='http://127.0.0.1:8501', subprotocols=['streamlit']) as ws:
        async def rerun(button_id=None):
            request = BackMsg()
            request.rerun_script.query_string = ''
            if button_id:
                widget = request.rerun_script.widget_states.widgets.add()
                widget.id = button_id
                widget.trigger_value = True
            await ws.send(request.SerializeToString())
            elements = []
            for _ in range(150):
                raw = await asyncio.wait_for(ws.recv(), timeout=25)
                message = ForwardMsg()
                message.ParseFromString(raw)
                if message.HasField('delta') and message.delta.HasField('new_element'):
                    element = message.delta.new_element
                    assert not element.HasField('exception'), 'Streamlit application exception'
                    elements.append(element)
                if message.HasField('script_finished'):
                    return elements
            raise AssertionError('Streamlit did not finish rendering')

        first = await rerun()
        assert any(e.HasField('heading') and e.heading.body == 'CityPulse SQLPilot' for e in first)
        button = next(e.button.id for e in first if e.HasField('button') and
                      e.button.label == 'Inspect permitted schema')
        second = await rerun(button)
        schemas = [json.loads(e.json.body) for e in second if e.HasField('json')]
        assert any(set(value.get('schema', {})) == {'hourly_rides', 'validated_trips'} for value in schemas)
        return 'HTTP WebSocket UI rendered and schema button executed through MCP'


async def main():
    assert os.getuid() != 0, 'Container must run as non-root'
    assert configured_backend() is None, 'Do not run automated smoke checks with an enabled provider'
    database = os.environ['SQLPILOT_DATABASE']
    for path in ('/', '/_stcore/health'):
        with urllib.request.urlopen('http://127.0.0.1:8501'+path, timeout=3) as response:
            assert response.status == 200
    ui = await browser_check()
    async with connect_tools(database) as tools:
        schema = await tools.call('sqlpilot_schema', {})
        expected = await tools.call('sqlpilot_query', {'sql': 'SELECT sum(ride_count) FROM hourly_rides'})
        assert set(schema['schema']) == {'hourly_rides', 'validated_trips'}
        try:
            await tools.call('sqlpilot_query', {'sql': 'DELETE FROM hourly_rides'})
        except ToolError:
            pass
        else:
            raise AssertionError('Unsafe SQL was not blocked')
        protocol = tools.audit[0]['protocol']
    result = await run_agent('Sum accepted recorded starts in this sample', database,
                            ControlledGeneration({'sql': 'SELECT sum(ride_count) FROM hourly_rides'}))
    assert result['status'] == 'executed' and result['result']['rows'] == expected['rows']
    assert not result['live_llm_exercised']
    clarification = await run_agent('Undefined ranking metric', database,
                                   ControlledGeneration({'clarification': 'Which ranking metric and time period?'}))
    assert clarification['status'] == 'clarification'
    print(json.dumps({'http_status': 200, 'ui': ui, 'mcp_protocol': protocol,
                      'mcp_tools': ['sqlpilot_schema', 'sqlpilot_query'],
                      'unsafe_sql_rejected': True, 'langgraph_execution': result['status'],
                      'clarification': clarification['status'], 'live_llm_exercised': False,
                      'generation_kind': 'controlled_test_double', 'uid': os.getuid()}))


if __name__ == '__main__':
    asyncio.run(main())
