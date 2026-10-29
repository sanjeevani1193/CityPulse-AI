"""Future live NL evaluation. Disabled without operator-approved provider access."""
import argparse
import asyncio
import json
import math
import hashlib
import statistics
from pathlib import Path
import time

from citypulse_ai.sqlpilot.client import connect_tools
from citypulse_ai.sqlpilot.graph import run_agent
from citypulse_ai.sqlpilot.providers import configured_backend


def same_rows(actual, expected):
    if len(actual) != len(expected): return False
    for a, e in zip(actual, expected):
        if len(a) != len(e): return False
        for x, y in zip(a, e):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if not math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-8): return False
            elif x != y: return False
    return True


async def evaluate(database, specification, backend, *, pause_seconds=65):
    if backend is None:
        return {'status': 'live_backend_pending', 'live_llm_exercised': False, 'live_llm_attempted': False, 'cases_evaluated': 0}
    cases = specification['cases']
    if not 0 < len(cases) <= 10: raise ValueError('Evaluation is bounded to ten questions')
    live = bool(getattr(backend, 'is_live', False))
    if live and pause_seconds < 65: raise ValueError('Live evaluation requires at least 65 seconds between questions')
    records = []; started = time.monotonic(); previous_start = None; stopped_reason = None
    for case in cases:
        if live and previous_start is not None:
            await asyncio.sleep(max(0, pause_seconds-(time.monotonic()-previous_start)))
        previous_start = time.monotonic()
        expected = None
        if 'reference_sql' in case:
            async with connect_tools(database) as tools:
                reference = await tools.call('sqlpilot_query', {'sql': case['reference_sql']})
                expected = reference['rows']
        result = await run_agent(case['question'], database, backend)
        correct = (result['status'] == 'executed' and same_rows(result['result']['rows'], expected)) if expected is not None else result['status'] == case['expected']
        category = 'answerable' if expected is not None else 'unsafe_question' if case['id'] == 'injection' else 'clarification'
        records.append({'id': case['id'], 'category': category, 'status': result['status'], 'correct': correct,
            'live_llm_exercised': result.get('live_llm_exercised', False),
            'live_llm_attempted': result.get('live_llm_attempted', False),
            'latency_seconds': result.get('latency_seconds'), 'attempts': result.get('attempts', 0),
            # Local diagnostic evidence only: never sent to the model or telemetry.
            'sql_attempts': result.get('sql_attempts', []),
            'query_errors': result.get('query_errors', []),
            'actual_result': result.get('result'),
            'reference_result': reference if expected is not None else None,
            'clarification': result.get('clarification'), 'error': result.get('error'),
            'agent_trace': result.get('trace', []), 'mcp_trace': result.get('mcp_trace', []),
            'query_tool_rejections': sum(x.get('is_error', False) for x in result.get('mcp_trace', []) if x.get('tool') == 'sqlpilot_query')})
        http_status = getattr(backend, 'last_http_status', None)
        if http_status is not None and http_status >= 400:
            stopped_reason = 'provider_http_'+str(http_status)
            break
    answerable = [r for r in records if r['category'] == 'answerable']
    clarifications = [r for r in records if r['category'] != 'answerable']
    unsafe = [r for r in records if r['category'] == 'unsafe_question']
    latencies = [r['latency_seconds'] for r in records if r['latency_seconds'] is not None]
    return {'evaluation_version': specification['version'], 'cases': records,
            'provider': getattr(getattr(backend, 'config', None), 'provider', 'controlled_test_double'),
            'requested_model': getattr(getattr(backend, 'config', None), 'model', None), 'cases_evaluated': len(records),
            'cases_planned': len(cases), 'cases_not_run': len(cases)-len(records), 'stopped_reason': stopped_reason,
            'correct': sum(r['correct'] for r in records), 'executed': sum(r['status'] == 'executed' for r in records),
            'answerable_cases': len(answerable),
            'answerable_cases_planned': sum('reference_sql' in c for c in cases),
            'clarification_cases_planned': sum('reference_sql' not in c for c in cases),
            'unsafe_questions_planned': sum(c['id'] == 'injection' for c in cases),
            'answerable_execution_successes': sum(r['status'] == 'executed' for r in answerable),
            'answerable_correct_results': sum(r['correct'] for r in answerable),
            'clarification_cases': len(clarifications),
            'correct_clarification_statuses': sum(r['correct'] for r in clarifications),
            'unsafe_questions': len(unsafe), 'unsafe_questions_declined': sum(r['correct'] for r in unsafe),
            'query_tool_errors': sum(r['query_tool_rejections'] for r in records),
            'query_tool_error_note': 'Includes binding/execution errors as well as policy rejections; not an unsafe-SQL rejection rate',
            'median_latency_seconds': statistics.median(latencies) if latencies else None,
            'max_latency_seconds': max(latencies) if latencies else None,
            'question_spec_sha256': hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest(),
            'live_llm_exercised': any(r['live_llm_exercised'] for r in records),
            'live_llm_attempted': any(r['live_llm_attempted'] for r in records),
            'generation_kind': 'live' if live else 'controlled_test_double',
            'pause_between_questions_seconds': pause_seconds if live else 0,
            'elapsed_seconds': time.monotonic()-started}



if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', required=True)
    p.add_argument('--questions', type=Path, default=Path('tests/fixtures/sqlpilot_questions.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.questions.stat().st_size > 65536: raise SystemExit('Question corpus too large')
    result = asyncio.run(evaluate(args.database, json.loads(args.questions.read_text()), configured_backend()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({'live_llm_exercised': result['live_llm_exercised'], 'cases_evaluated': result['cases_evaluated']}))
