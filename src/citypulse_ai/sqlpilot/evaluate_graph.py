"""Future live NL evaluation. Disabled without operator-approved provider access."""
import argparse
import asyncio
import json
import math
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


async def evaluate(database, specification, backend):
    if backend is None:
        return {'status': 'live_backend_pending', 'live_llm_exercised': False, 'cases_evaluated': 0}
    cases = specification['cases']
    if not 0 < len(cases) <= 10: raise ValueError('Evaluation is bounded to ten questions')
    records = []; started = time.monotonic()
    for case in cases:
        expected = None
        if 'reference_sql' in case:
            async with connect_tools(database) as tools:
                reference = await tools.call('sqlpilot_query', {'sql': case['reference_sql']})
                expected = reference['rows']
        result = await run_agent(case['question'], database, backend)
        correct = (result['status'] == 'executed' and same_rows(result['result']['rows'], expected)) if expected is not None else result['status'] == case['expected']
        records.append({'id': case['id'], 'status': result['status'], 'correct': correct,
            'live_llm_exercised': result.get('live_llm_exercised', False),
            'latency_seconds': result.get('latency_seconds'), 'attempts': result.get('attempts', 0),
            'query_tool_rejections': sum(x.get('is_error', False) for x in result.get('mcp_trace', []) if x.get('tool') == 'sqlpilot_query')})
    return {'evaluation_version': specification['version'], 'cases': records, 'cases_evaluated': len(records),
            'correct': sum(r['correct'] for r in records), 'executed': sum(r['status'] == 'executed' for r in records),
            'live_llm_exercised': any(r['live_llm_exercised'] for r in records),
            'generation_kind': 'live' if getattr(backend, 'is_live', False) else 'controlled_test_double',
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
