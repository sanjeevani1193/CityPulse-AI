"""Bounded LangGraph state machine; SQL execution exclusively through MCP tools."""
import asyncio
import hashlib
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langsmith import tracing_context
from citypulse_ai.sqlpilot.client import connect_tools, ToolError


class State(TypedDict, total=False):
    question: str
    schema: dict
    sql: str
    status: str
    clarification: str
    error: str
    attempts: int
    tool_calls: int
    result: dict
    explanation: str
    trace: Annotated[list[dict], operator.add]


def build_graph(tools, backend):
    async def discover(state):
        try:
            value = await tools.call('sqlpilot_schema', {})
            return {'schema': value['schema'], 'status': 'generate', 'tool_calls': 1,
                    'trace': [{'node': 'discover_schema', 'outcome': 'success'}]}
        except Exception:
            return {'status': 'error', 'error': 'Approved sample schema could not be discovered', 'tool_calls': 1,
                    'trace': [{'node': 'discover_schema', 'outcome': 'error'}]}

    async def generate(state):
        attempt = state.get('attempts', 0)+1
        question = state['question']
        if attempt > 1:
            # Only the bounded error category and previous SQL, not query rows, are sent.
            question += '\nPrevious SQL: '+state['sql']+'\nExecution feedback: '+state['error']
        try:
            decision = await asyncio.to_thread(backend.generate, question, state['schema'])
            if not isinstance(decision, dict): raise ValueError('Invalid decision')
            if set(decision) == {'clarification'} and isinstance(decision['clarification'], str) and decision['clarification'].strip():
                return {'status': 'clarification', 'clarification': decision['clarification'][:1000],
                        'attempts': attempt, 'trace': [{'node': 'generate_sql', 'outcome': 'clarification', 'attempt': attempt}]}
            if set(decision) != {'sql'} or not isinstance(decision['sql'], str) or not 0 < len(decision['sql'].encode()) <= 4096:
                raise ValueError('Invalid SQL decision')
            return {'sql': decision['sql'], 'attempts': attempt, 'status': 'execute',
                    'trace': [{'node': 'generate_sql', 'outcome': 'sql', 'attempt': attempt,
                               'sql_sha256': hashlib.sha256(decision['sql'].encode()).hexdigest()}]}
        except Exception:
            return {'status': 'error', 'error': 'LLM generation failed or returned an invalid decision',
                    'attempts': attempt, 'trace': [{'node': 'generate_sql', 'outcome': 'error', 'attempt': attempt}]}

    async def execute(state):
        count = state['tool_calls']+1
        try:
            result = await tools.call('sqlpilot_query', {'sql': state['sql']})
            return {'result': result, 'status': 'explain', 'tool_calls': count,
                    'trace': [{'node': 'execute_sql', 'outcome': 'success', 'rows': len(result['rows'])}]}
        except ToolError:
            return {'error': 'SQL was rejected or could not bind/execute against the permitted schema',
                    'status': 'repair' if state['attempts'] < 2 and count < 3 else 'error', 'tool_calls': count,
                    'trace': [{'node': 'execute_sql', 'outcome': 'rejected', 'attempt': state['attempts']}]}
        except Exception:
            return {'status': 'error', 'error': 'MCP query execution failed', 'tool_calls': count,
                    'trace': [{'node': 'execute_sql', 'outcome': 'transport_error'}]}

    async def repair(state):
        return {'trace': [{'node': 'repair', 'outcome': 'one_correction_allowed'}]}

    async def explain(state):
        result = state['result']; explanation = result['explanation']
        if len(result['rows']) == 1 and len(result['columns']) == 1:
            explanation = f"The query returned {result['columns'][0]} = {result['rows'][0][0]}. "+explanation
        return {'status': 'executed', 'explanation': explanation,
                'trace': [{'node': 'explain', 'outcome': 'grounded_in_returned_rows'}]}

    graph = StateGraph(State)
    for name, function in [('discover_schema', discover), ('generate_sql', generate), ('execute_sql', execute), ('repair', repair), ('explain', explain)]:
        graph.add_node(name, function)
    graph.add_edge(START, 'discover_schema')
    graph.add_conditional_edges('discover_schema', lambda s: 'generate_sql' if s['status'] == 'generate' else END)
    graph.add_conditional_edges('generate_sql', lambda s: 'execute_sql' if s['status'] == 'execute' else END)
    graph.add_conditional_edges('execute_sql', lambda s: s['status'] if s['status'] in ('repair', 'explain') else END)
    graph.add_edge('repair', 'generate_sql'); graph.add_edge('explain', END)
    return graph.compile()


async def run_agent(question, database, backend=None):
    started = time.monotonic()
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        return {'status': 'error', 'error': 'Enter an analytical question of 1–2000 characters', 'trace': []}
    if backend is None:
        return {'status': 'backend_unavailable', 'error': 'No approved configured LLM. No external call or guessed SQL was made.', 'trace': [], 'live_llm_exercised': False}
    success_before = getattr(backend, 'successful_generations', 0)
    calls_before = getattr(backend, 'calls', 0)
    try:
        async with asyncio.timeout(60):
            async with connect_tools(database) as tools:
                with tracing_context(enabled=False):
                    result = await build_graph(tools, backend).ainvoke(
                        {'question': question, 'attempts': 0, 'tool_calls': 0, 'trace': []}, config={'recursion_limit': 10})
                result['mcp_trace'] = tools.audit
        result['latency_seconds'] = time.monotonic()-started
        result['live_llm_exercised'] = bool(getattr(backend, 'is_live', False) and getattr(backend, 'successful_generations', 0) > success_before)
        result['live_llm_attempted'] = bool(getattr(backend, 'is_live', False) and getattr(backend, 'calls', 0) > calls_before)
        return result
    except Exception:
        return {'status': 'error', 'error': 'Agent or MCP startup exceeded its budget or failed safely',
                'trace': [], 'live_llm_exercised': bool(getattr(backend, 'is_live', False) and getattr(backend, 'successful_generations', 0) > success_before),
                'live_llm_attempted': bool(getattr(backend, 'is_live', False) and getattr(backend, 'calls', 0) > calls_before),
                'latency_seconds': time.monotonic()-started}
