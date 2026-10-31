"""Deployment contracts; controlled inference doubles, no training/cloud calls."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from citypulse_ai.api.cloudrun_xgboost import create_app
from citypulse_ai.api.station_app import create_app as six_model_app
from tests.ci.test_station_contracts import ContractRuntime, UnavailableWorker, payload


@pytest.fixture
def client():
    with TestClient(create_app(ContractRuntime)) as value:
        yield value


def test_contract_parity_default_and_model_listing(client):
    with TestClient(six_model_app(ContractRuntime, UnavailableWorker)) as original:
        for endpoint in ('/health', '/stations'):
            assert client.get(endpoint).json() == original.get(endpoint).json()
        data = payload()
        actual = client.post('/predict', json=data)
        assert actual.status_code == 200
        assert actual.json() == original.post('/predict', json=data).json()
        assert len(actual.json()['predicted_ride_counts']) == 24
        assert client.post('/predict', json=data).json() == actual.json()
        assert len(original.get('/models').json()['models']) == 6
    assert [m['id'] for m in client.get('/models').json()['models']] == ['xgboost']


@pytest.mark.parametrize('model', ['pytorch-mlp', 'pytorch-tft', 'tensorflow-lstm', 'lag-24', 'lag-168', 'unknown'])
def test_only_xgboost_is_exposed(client, model):
    assert client.post('/predict', json={**payload(), 'model': model}).status_code == 422


@pytest.mark.parametrize('failure', ['future', 'duplicate', 'missing', 'negative', 'midday', 'station', 'artifact_path'])
def test_input_safety_is_preserved(client, failure):
    data = payload()
    if failure == 'future': data['history'][-1]['timestamp'] = data['origin']
    if failure == 'duplicate': data['history'][1]['timestamp'] = data['history'][0]['timestamp']
    if failure == 'missing': data['history'].pop()
    if failure == 'negative': data['history'][0]['ride_count'] = -1
    if failure == 'midday': data['origin'] = '2025-08-01T12:00:00-04:00'
    if failure == 'station': data['station_id'] = 'unknown'
    if failure == 'artifact_path': data['artifact_path'] = '/untrusted/model.ubj'
    assert client.post('/predict', json=data).status_code == 422


@pytest.mark.parametrize('origin', ['2026-03-08T00:00:00-05:00', '2025-11-02T00:00:00-04:00'])
def test_elapsed_dst_contract(client, origin):
    from datetime import datetime
    result = client.post('/predict', json=payload(origin)).json()
    stamps = [datetime.fromisoformat(t) for t in result['target_timestamps_utc']]
    assert len(stamps) == 24
    assert all((b-a).total_seconds() == 3600 for a,b in zip(stamps, stamps[1:]))
    assert len(set(result['utc_offsets'])) == 2


def test_missing_or_corrupt_artifacts_fail_closed():
    class Missing:
        def __init__(self): raise ValueError('Missing artifact')
    class Corrupt(ContractRuntime):
        def load(self, model): raise ValueError('Checksum mismatch')
    for runtime in (Missing, Corrupt):
        with TestClient(create_app(runtime)) as c:
            assert c.get('/health').status_code == 503
            assert c.get('/health').json()['model_ready'] is False
            assert c.get('/stations').status_code == 503
            assert c.post('/predict', json=payload()).status_code == 503


def test_load_once_and_no_worker_or_other_model_loads():
    calls=[]
    class Counted(ContractRuntime):
        def load(self, model): calls.append(model); super().load(model)
    with TestClient(create_app(Counted)) as c:
        for _ in range(2): assert c.post('/predict', json=payload()).status_code == 200
    assert calls == ['xgboost']


def test_cold_import_never_requires_heavy_frameworks():
    code = '''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','sklearn','google'}:
   raise ImportError('Deliberately absent optional framework')
sys.meta_path.insert(0,Deny())
from citypulse_ai.api.cloudrun_xgboost import app
assert app.title.startswith('CityPulse')
assert not any(name in sys.modules for name in ('torch','tensorflow','sklearn'))
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=25)
    assert result.returncode == 0, result.stderr


def test_platform_and_minimal_build_configuration():
    import yaml
    docker = Path('docker/cloudrun-xgboost.Dockerfile').read_text()
    assert "platform.machine() == 'x86_64'" in docker
    assert 'USER 10001:10001' in docker
    assert 'cloudrun_xgboost' in docker
    requirements = Path('docker/cloudrun-xgboost.requirements.txt').read_text().splitlines()
    assert not any(line.startswith(('torch', 'tensorflow', 'scikit-learn')) for line in requirements)
    config = yaml.safe_load(Path('infra/cloudbuild-xgboost.yaml').read_text())
    assert '--platform=linux/amd64' in config['steps'][0]['args']
    assert config['timeout'] == '900s'
    assert config['options']['logging'] == 'CLOUD_LOGGING_ONLY'


def test_bundle_is_create_only_and_contains_exact_verified_artifacts(tmp_path, monkeypatch):
    from citypulse_ai.api import cloudrun_bundle as bundle
    import hashlib
    inputs=tmp_path/'source';inputs.mkdir()
    expected={}
    for name in bundle.ARTIFACTS:
        p=inputs/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode())
        expected[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
    class Store:
        def __init__(self, root):self.root=Path(root);self.specification={'artifacts':expected}
        def file(self, name):return self.root/name
    monkeypatch.setattr(bundle,'ArtifactStore',Store)
    target=tmp_path/'context'
    bundle.prepare(inputs,target,Path.cwd())
    actual={str(p.relative_to(target/'artifacts')) for p in (target/'artifacts').rglob('*') if p.is_file()}
    assert actual == set(bundle.ARTIFACTS)
    assert not (target/'src/citypulse_ai/sqlpilot').exists()
    assert not (target/'src/citypulse_ai/experiments').exists()
    assert not list(target.rglob('*.pyc'))
    with pytest.raises(FileExistsError):bundle.prepare(inputs,target,Path.cwd())
    assert json.loads((target/'bundle-receipt.json').read_text())['artifacts']==expected


def test_bundle_omits_arbitrary_json_secrets_and_weights(tmp_path, monkeypatch):
    from citypulse_ai.api import cloudrun_bundle as bundle
    import hashlib
    import shutil
    source=tmp_path/'source';source.mkdir()
    expected={}
    for name in bundle.ARTIFACTS:
        path=source/name;path.parent.mkdir(exist_ok=True);path.write_bytes(name.encode())
        expected[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}
    class Store:
        def __init__(self, root):self.root=Path(root);self.specification={'artifacts':expected}
        def file(self, name):return self.root/name
    monkeypatch.setattr(bundle,'ArtifactStore',Store)
    repo=tmp_path/'repo';package=repo/'src/citypulse_ai';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    for name in ('api','features','forecasting','etl'):
        folder=package/name;folder.mkdir()
        (folder/'__init__.py').write_text('')
        (folder/'credentials.json').write_text('{"fake":"not-a-real-secret"}')
        (folder/'weights.pt').write_bytes(b'not-a-real-checkpoint')
    (package/'api/station_artifacts.json').write_text('{}')
    (repo/'docker').mkdir();(repo/'infra').mkdir()
    for name in ('cloudrun-xgboost.Dockerfile','cloudrun-xgboost.Dockerfile.dockerignore','cloudrun-xgboost.requirements.txt'):
        shutil.copyfile(Path('docker')/name,repo/'docker'/name)
    shutil.copyfile('infra/cloudbuild-xgboost.yaml',repo/'infra/cloudbuild-xgboost.yaml')
    output=tmp_path/'context';bundle.prepare(source,output,repo)
    assert not list((output/'src').rglob('credentials.json'))
    assert not list(output.rglob('*.pt'))
    assert (output/'src/citypulse_ai/api/station_artifacts.json').is_file()
