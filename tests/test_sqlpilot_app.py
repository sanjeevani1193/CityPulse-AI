"""Real SDK/MCP/LangGraph tests with explicitly controlled generation doubles."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from citypulse_ai.sqlpilot.client import connect_tools, ToolError
from citypulse_ai.sqlpilot.graph import run_agent
from citypulse_ai.sqlpilot.providers import ProviderConfig, HostedJSONBackend, schema_identity
from tests.test_sqlpilot import sample


class Scripted:
    is_live = False
    def __init__(self,*decisions):self.decisions=list(decisions);self.calls=[]
    def generate(self,question,schema):
        self.calls.append((question,schema))
        return self.decisions.pop(0)


def ask(sample,backend,question='How many starts in this synthetic sample?'):
    return asyncio.run(run_agent(question,sample,backend))


def test_real_sdk_tool_discovery_and_calls(sample):
    async def run():
        async with connect_tools(sample) as tools:
            schema=await tools.call('sqlpilot_schema',{})
            assert set(schema['schema'])=={'hourly_rides','validated_trips'}
            result=await tools.call('sqlpilot_query',{'sql':'SELECT sum(ride_count) FROM hourly_rides'})
            assert result['rows']==[[6]]
            with pytest.raises(ToolError):await tools.call('sqlpilot_query',{'sql':'DELETE FROM hourly_rides'})
            with pytest.raises(ToolError,match='budget'):await tools.call('sqlpilot_query',{'sql':'SELECT sum(ride_count) FROM hourly_rides'})
            assert tools.audit[0]['protocol']=='2025-03-26'
    asyncio.run(run())


def test_langgraph_actual_mcp_result_and_sanitized_trace(sample):
    result=ask(sample,Scripted({'sql':'SELECT sum(ride_count) AS starts FROM hourly_rides'}))
    assert result['status']=='executed' and result['result']['rows']==[[6]]
    assert 'starts = 6' in result['explanation'] and not result['live_llm_exercised']
    assert [x['node'] for x in result['trace']]==['discover_schema','generate_sql','execute_sql','explain']
    calls=[x['tool'] for x in result['mcp_trace'] if x['method']=='tools/call']
    assert calls==['sqlpilot_schema','sqlpilot_query']
    assert 'SELECT' not in json.dumps(result['trace']) and 'question' not in json.dumps(result['trace'])


def test_one_repair_and_fixed_limits(sample):
    backend=Scripted({'sql':'SELECT missing FROM hourly_rides'}, {'sql':'SELECT sum(ride_count) FROM hourly_rides'})
    result=ask(sample,backend)
    assert result['status']=='executed' and result['attempts']==2 and result['tool_calls']==3
    assert any(x['node']=='repair' for x in result['trace']) and len(backend.calls)==2
    bad=Scripted({'sql':'DELETE FROM hourly_rides'},{'sql':"SELECT * FROM read_csv('/etc/passwd')"})
    result=ask(sample,bad)
    assert result['status']=='error' and result['attempts']==2 and result['tool_calls']==3


def test_clarification_backend_failure_and_invalid_decision(sample):
    for decision in [{'clarification':'Which sample period?'}, {'clarification':'Physical demand is not observed here.'}]:
        result=ask(sample,Scripted(decision));assert result['status']=='clarification' and result['tool_calls']==1
    result=ask(sample,Scripted({'answer':'invented'}));assert result['status']=='error'
    assert ask(sample,None)['status']=='backend_unavailable'
    assert ask(sample,Scripted(),question='')['status']=='error'


def test_schema_failure_never_calls_backend(tmp_path):
    backend=Scripted({'sql':'SELECT 1'})
    result=ask(tmp_path/'absent.duckdb',backend)
    assert result['status']=='error' and backend.calls==[]


def approved(schema):
    return ProviderConfig(provider='groq',model='explicit-test-model',api_key='not-a-real-key',approved=True,
        free_allowance_verified=True,questions_approved=True,approved_schema_sha256=schema_identity(schema))


def test_provider_requires_approval_and_exact_schema(monkeypatch):
    for key in ['SQLPILOT_PROVIDER','SQLPILOT_MODEL','GROQ_API_KEY','SQLPILOT_PROVIDER_APPROVED',
                'SQLPILOT_FREE_ALLOWANCE_VERIFIED','SQLPILOT_APPROVED_SCHEMA_SHA256','SQLPILOT_QUESTIONS_APPROVED']:
        monkeypatch.delenv(key,raising=False)
    assert not ProviderConfig.environment().status()['configured']
    with pytest.raises(ValueError):HostedJSONBackend(ProviderConfig())
    def forbidden(request):pytest.fail('Unapproved transmission')
    b=HostedJSONBackend(approved({'allowed':1}),transport=httpx.MockTransport(forbidden))
    with pytest.raises(ValueError,match='Schema'):b.generate('question',{'different':2})
    assert b.calls==0 and b.successful_generations==0 and 'not-a-real-key' not in repr(b.config)


def test_provider_mocked_http_payload_not_live_generation():
    schema={'table':{'columns':{'count':'BIGINT'}}};seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"clarification":"Which period?"}'}}]})
    b=HostedJSONBackend(approved(schema),transport=httpx.MockTransport(handler))
    assert b.generate('Question',schema)=={'clarification':'Which period?'} and not b.is_live
    content=json.loads(seen[0]['messages'][1]['content'])
    assert set(content)=={'question','schema'} and 'rows' not in content


def test_streamlit_missing_backend_and_real_schema(sample,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('SQLPILOT_DATABASE',str(sample))
    monkeypatch.delenv('SQLPILOT_PROVIDER_APPROVED',raising=False)
    app=AppTest.from_file(str(Path('src/citypulse_ai/sqlpilot/ui.py').resolve()),default_timeout=20).run()
    assert not app.exception and 'disabled' in app.info[0].value
    app.button[0].click().run()
    assert 'sqlpilot_schema' in app.session_state
    app.text_area[0].input('How many starts?')
    app.button[1].click().run()
    assert 'No approved' in app.error[0].value and not app.exception


def test_streamlit_runs_real_graph_and_mcp_with_double(sample,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from citypulse_ai.sqlpilot import providers
    monkeypatch.setenv('SQLPILOT_DATABASE',str(sample))
    monkeypatch.setattr(providers,'configured_backend',lambda:Scripted({'sql':'SELECT sum(ride_count) AS starts FROM hourly_rides'}))
    app=AppTest.from_file(str(Path('src/citypulse_ai/sqlpilot/ui.py').resolve()),default_timeout=20).run()
    app.text_area[0].input('How many sampled starts?');app.button[1].click().run()
    assert not app.exception and app.dataframe[0].value.iloc[0,0]==6
    assert 'SELECT' in app.code[0].value
    assert app.session_state['sqlpilot_result']['status']=='executed'
    assert not app.session_state['sqlpilot_result']['live_llm_exercised']


def test_question_corpus_references_use_real_mcp(sample):
    data=json.loads(Path('tests/fixtures/sqlpilot_questions.json').read_text())
    assert len(data['cases'])==9
    async def run():
        for case in data['cases']:
            if 'reference_sql' not in case:continue
            async with connect_tools(sample) as tools:
                result=await tools.call('sqlpilot_query',{'sql':case['reference_sql']})
                assert result['rows']==case['fixture_expected_rows']
    asyncio.run(run())


def test_evaluation_pending_never_calls_a_model(sample):
    from citypulse_ai.sqlpilot.evaluate_graph import evaluate,same_rows
    result=asyncio.run(evaluate(sample,{'cases':[]},None))
    assert result['cases_evaluated']==0 and not result['live_llm_exercised']
    assert same_rows([[1.000000001]],[[1.]]) and not same_rows([[None]],[[0]])


def test_no_trace_network_with_global_tracing_enabled(sample,monkeypatch):
    import socket
    monkeypatch.setenv('LANGSMITH_TRACING','true')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2','true')
    monkeypatch.setenv('LANGSMITH_API_KEY','not-a-real-key')
    original=socket.socket.connect
    def no_network(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):pytest.fail('Unexpected external tracing/network')
        return original(sock,address)
    monkeypatch.setattr(socket.socket,'connect',no_network)
    result=ask(sample,Scripted({'sql':'SELECT sum(ride_count) FROM hourly_rides'}))
    assert result['status']=='executed'


def test_replaceable_endpoint_requires_exact_operator_approval():
    from dataclasses import replace
    base=approved({'sample':1})
    alternate=replace(base,provider='openai-compatible',endpoint='https://approved.example/chat')
    assert not alternate.status()['configured']
    alternate=replace(alternate,approved_endpoint='https://approved.example/chat')
    assert alternate.status()['configured']
    assert not replace(alternate,endpoint='http://approved.example/chat',approved_endpoint='http://approved.example/chat').status()['configured']
    assert not replace(alternate,endpoint='https://[invalid',approved_endpoint='https://[invalid').status()['configured']


def test_prior_provider_success_is_not_current_live_evidence(sample):
    class PreviouslySuccessful:
        is_live=True
        calls=10
        successful_generations=10
        def generate(self,q,s):raise ValueError('No response in this invocation')
    result=ask(sample,PreviouslySuccessful())
    assert result['status']=='error'
    assert not result['live_llm_exercised'] and not result['live_llm_attempted']
