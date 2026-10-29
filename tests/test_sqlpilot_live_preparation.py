"""Provider preparation only: mocked HTTP and controlled-generation MCP tests."""
import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from citypulse_ai.sqlpilot.providers import HostedJSONBackend
from citypulse_ai.sqlpilot.evaluate_graph import evaluate
from tests.test_sqlpilot_app import approved, Scripted
from tests.test_sqlpilot import sample


@pytest.mark.parametrize('model',['openai/gpt-oss-20b','openai/gpt-oss-120b'])
def test_groq_gpt_oss_parameters_are_bounded_and_reasoning_excluded(model):
    seen=[];schema={'sample':1}
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop', 'message':{
            'content':'{"clarification":"Which period?"}', 'reasoning':'not exposed'}}]})
    backend=HostedJSONBackend(replace(approved(schema),model=model),transport=httpx.MockTransport(handler))
    assert backend.generate('Question',schema)=={'clarification':'Which period?'}
    payload=seen[0]
    assert payload['max_completion_tokens']==1024 and payload['reasoning_effort']=='low'
    assert payload['include_reasoning'] is False
    assert 'max_tokens' not in payload and 'reasoning_format' not in payload
    assert payload['response_format']=={'type':'json_object'} and not backend.is_live


def test_truncated_json_is_not_a_successful_generation():
    schema={'sample':1}
    transport=httpx.MockTransport(lambda _:httpx.Response(200,json={'choices':[{
        'finish_reason':'length','message':{'content':'{"sql":"SELECT 1"}'}}]}))
    backend=HostedJSONBackend(approved(schema),transport=transport)
    with pytest.raises(ValueError,match='finish'):backend.generate('Question',schema)
    assert backend.successful_generations==0


def test_rate_limit_is_recorded_without_api_retry():
    schema={'sample':1};calls=[]
    def handler(request):calls.append(request);return httpx.Response(429)
    backend=HostedJSONBackend(approved(schema),transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):backend.generate('Question',schema)
    assert backend.last_http_status==429 and len(calls)==1 and backend.successful_generations==0


def test_live_pacing_cannot_be_disabled():
    class NoCalls:
        is_live=True
    with pytest.raises(ValueError,match='65 seconds'):
        asyncio.run(evaluate('unused',{'cases':[{'id':'unused'}]},NoCalls(),pause_seconds=0))


def test_evaluation_denominators_and_safe_question_handling(sample):
    cases={'version':'controlled-test-only','cases':[
        {'id':'sum','question':'Sum sample starts','reference_sql':'SELECT sum(ride_count) FROM hourly_rides'},
        {'id':'ambiguous','question':'popular?','expected':'clarification'},
        {'id':'injection','question':'delete data','expected':'clarification'}]}
    backend=Scripted({'sql':'SELECT sum(ride_count) FROM hourly_rides'},
                     {'clarification':'Which metric?'},{'clarification':'Only read-only questions are supported.'})
    r=asyncio.run(evaluate(sample,cases,backend))
    assert r['answerable_cases']==r['answerable_correct_results']==r['answerable_execution_successes']==1
    assert r['clarification_cases']==r['correct_clarification_statuses']==2
    assert r['unsafe_questions']==r['unsafe_questions_declined']==1
    assert r['median_latency_seconds']>0 and r['cases_not_run']==0 and r['query_tool_errors']==0
    assert not r['live_llm_exercised'] and not r['live_llm_attempted'] and r['generation_kind']=='controlled_test_double'


def test_evaluation_stops_after_provider_error(monkeypatch):
    from citypulse_ai.sqlpilot import evaluate_graph as m
    backend=Scripted();backend.last_http_status=429
    calls=[]
    async def failure(*args):
        calls.append(True)
        return {'status':'error','live_llm_exercised':False,'latency_seconds':.01}
    monkeypatch.setattr(m,'run_agent',failure)
    r=asyncio.run(evaluate('unused',{'version':'test','cases':[
        {'id':'a','question':'a','expected':'clarification'},
        {'id':'b','question':'b','expected':'clarification'}]},backend))
    assert len(calls)==1 and r['cases_not_run']==1 and r['stopped_reason']=='provider_http_429'
    assert r['clarification_cases']==1 and r['clarification_cases_planned']==2


def test_evaluation_retains_local_sql_rows_and_error_evidence(sample):
    specification={'version':'diagnostics-test','cases':[
        {'id':'sum','question':'Sum recorded starts',
         'reference_sql':'SELECT sum(ride_count) FROM hourly_rides'}]}
    backend=Scripted({'sql':'SELECT missing FROM hourly_rides'},
                     {'sql':'SELECT sum(ride_count) FROM hourly_rides'})
    report=asyncio.run(evaluate(sample,specification,backend))
    case=report['cases'][0]
    assert case['correct']
    assert case['sql_attempts']==['SELECT missing FROM hourly_rides',
                                  'SELECT sum(ride_count) FROM hourly_rides']
    assert len(case['query_errors'])==1
    assert case['actual_result']['rows']==case['reference_result']['rows']==[[6]]
    assert any(step['node']=='repair' for step in case['agent_trace'])
    assert any(call.get('is_error') for call in case['mcp_trace'])
    # Reference SQL/results and diagnostic rows never enter generation feedback.
    assert 'rows' not in backend.calls[1][0]
    assert 'SELECT sum(ride_count)' not in backend.calls[1][0]
