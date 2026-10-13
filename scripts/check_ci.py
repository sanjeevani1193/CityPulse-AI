"""Small repository syntax/configuration checks; no Docker builds or ML imports.

Docker checks cover this project's Dockerfile subset and serving safety contract,
not every feature of Docker's parser. Compose's own parser is run separately.
"""
import ast
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check():
    python_files = []
    for directory in ('src', 'tests', 'docker', 'scripts'):
        python_files.extend((ROOT / directory).rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))

    workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    assert workflow['on'] == {'push': {'branches': ['main']}, 'pull_request': {'branches': ['main']}}
    assert workflow['permissions'] == {'contents': 'read'}
    assert set(workflow['jobs']) == {'repository-checks', 'gateway-tests', 'worker-tests'}
    for job in workflow['jobs'].values():
        assert 0 < job['timeout-minutes'] <= 15
        for step in job['steps']:
            assert not step.get('continue-on-error', False)
            if 'uses' in step:
                assert re.fullmatch(r'actions/(checkout|setup-python)@[a-f0-9]{40}', step['uses'])

    for name, module in [('gateway', 'app'), ('worker', 'lstm_worker')]:
        path = ROOT / f'docker/{name}.Dockerfile'
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
        instructions = [line.split(' ', 1) for line in lines]
        assert all(len(parts) == 2 and parts[0] in {'FROM', 'ENV', 'RUN', 'WORKDIR', 'COPY', 'USER', 'HEALTHCHECK', 'CMD'} for parts in instructions)
        assert re.fullmatch(r'FROM python:[\w.\-]+@sha256:[a-f0-9]{64}', lines[0])
        assert 'USER 10001:10001' in lines
        assert any(line.startswith('HEALTHCHECK ') for line in lines)
        assert json.loads(next(value for key,value in instructions if key == 'CMD')) == [
            'python', '-m', 'uvicorn', f'citypulse_ai.api.{module}:app', '--host', '0.0.0.0',
            '--port', '8000' if name == 'gateway' else '8001', '--workers', '1']
        for key,value in instructions:
            if key == 'COPY':
                source = value.split()[0]
                assert source in {f'docker/{name}.requirements.txt', 'src/citypulse_ai'}
                assert (ROOT / source).exists()

    compose = yaml.safe_load((ROOT / 'compose.yaml').read_text())
    services = compose['services']
    assert set(services) == {'forecast-gateway', 'lstm-worker'}
    assert services['forecast-gateway']['ports'] == ['127.0.0.1:8000:8000']
    assert 'ports' not in services['lstm-worker']
    assert services['lstm-worker']['networks'] == ['inference']
    assert compose['networks']['inference']['internal'] is True
    assert services['forecast-gateway']['environment']['CITYPULSE_LSTM_WORKER_URL'] == 'http://lstm-worker:8001'
    for service in services.values():
        assert service['read_only'] is True
        assert service['mem_limit'] == service['memswap_limit'] == '1536m'
        for mount in service['volumes']:
            assert mount['type'] == 'bind' and mount['read_only'] is True
            assert mount['bind']['create_host_path'] is False

    files = [ROOT / '.github/workflows/ci.yml', ROOT / 'compose.yaml', ROOT / '.dockerignore']
    files += list((ROOT / 'ci').glob('*.txt'))
    files += list((ROOT / 'tests/ci').glob('*.py')) + [Path(__file__)]
    files += list((ROOT / 'docker').glob('*.Dockerfile')) + list((ROOT / 'docker').glob('*.requirements.txt'))
    for path in files:
        for number,line in enumerate(path.read_text().splitlines(), 1):
            assert line == line.rstrip(), f'{path}:{number}: trailing whitespace'
    print(f'Python syntax: {len(python_files)} files; workflow YAML, Dockerfile contracts, Compose safety and whitespace passed')


if __name__ == '__main__':
    check()
