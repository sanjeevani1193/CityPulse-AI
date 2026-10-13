"""Host-only saved-reference HTTP checks; run from the repository root.

Starts/stops only this Compose project, preserving containers, images and caches.
Uses existing compact experiment evidence; never trains or downloads data.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'outputs/docker-verification' / uuid.uuid4().hex
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'pytorch-mlp': ROOT / 'outputs/phase3-pytorch-mlp-jan-feb-2024/run-20261012T125502822756Z',
    'xgboost': ROOT / 'outputs/phase3-xgboost-jan-feb-2024/recovered-83a9eaa90d964c889a03c7b6ec7de09d',
    'tensorflow-lstm': ROOT / 'outputs/phase3-tensorflow-lstm-jan-feb-2024/run-20261012T141848308777Z',
}
FILES = [SOURCES['pytorch-mlp'] / n for n in ('best_model.pt', 'experiment.json')]
FILES += [SOURCES['xgboost'] / n for n in ('model.ubj', 'model_manifest.json')]
FILES += [SOURCES['tensorflow-lstm'] / n for n in ('best.weights.h5', 'experiment.json')]
HASHES = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}
report = {'free_before_bytes': shutil.disk_usage(ROOT).free, 'checks': {}, 'memory_samples': []}
stop = threading.Event()

def docker(*args):
    return subprocess.check_output(['docker', *args], text=True)

def sample():
    while not stop.is_set():
        try:
            lines = docker('stats', '--no-stream', '--format', '{{json .}}').splitlines()
            rows = [json.loads(x) for x in lines]
            rows = [r for r in rows if 'forecast-gateway' in r['Name'] or 'lstm-worker' in r['Name']]
            report['memory_samples'].append({'time': time.time(), 'containers': rows})
        except subprocess.CalledProcessError:
            pass
        stop.wait(0.5)

def wait_ready(client):
    for _ in range(240):
        try:
            response = client.get('/models')
            if response.status_code == 200 and all(x['ready'] for x in response.json()['models']):
                return response.json()
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise AssertionError('Models did not become ready within 240 seconds')

history = json.loads((SOURCES['xgboost'] / 'recovery_history.json').read_text())['history']
origin = pd.Timestamp('2024-03-01T00:00:00-05:00')
request = {'origin': origin.isoformat(), 'history': [r for r in history if origin-pd.Timedelta(hours=168) <= pd.Timestamp(r['timestamp']) < origin]}
refs = {n: json.loads((p / 'experiment.json').read_text())['example_forecast'] for n,p in SOURCES.items() if n != 'xgboost'}
refs['xgboost'] = json.loads((SOURCES['xgboost'].parent / 'experiment.json').read_text())['example_next_day_forecast']
thread = threading.Thread(target=sample, daemon=True)
thread.start()
try:
    docker('compose', 'up', '-d')
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=15) as client:
        report['models'] = wait_ready(client)
        assert client.get('/health').status_code == 200
        results = {}
        for name,reference in refs.items():
            payload = {**request, 'model': name}
            a = client.post('/predict', json=payload)
            b = client.post('/predict', json=payload)
            assert a.status_code == b.status_code == 200 and a.json() == b.json()
            values = a.json()['predicted_ride_counts']
            expected = np.array([r['prediction'] for r in reference])
            assert len(values) == 24
            # Cross-platform float32 kernels may differ in last bits. Record actual error.
            np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-3)
            bad = {**payload, 'history': payload['history'] + [{'timestamp': origin.isoformat(), 'ride_count': 1}]}
            assert client.post('/predict', json=bad).status_code == 422
            invalid = {**payload, 'origin': '2024-03-01T01:00:00-05:00'}
            assert client.post('/predict', json=invalid).status_code == 422
            results[name] = a.json()
            report['checks'][name] = {'http': 200, 'count': 24, 'maximum_reference_error': float(np.max(np.abs(np.array(values)-expected))), 'deterministic': True, 'future_rejected': True, 'non_midnight_rejected': True}
            for day,last in [('2024-03-10T00:00:00-05:00','2024-03-11T00:00:00-04:00'), ('2024-11-03T00:00:00-04:00','2024-11-03T22:00:00-05:00')]:
                end = pd.Timestamp(day).tz_convert('UTC')
                context = [{'timestamp':t.isoformat(), 'ride_count':100+i%24} for i,t in enumerate(pd.date_range(end-pd.Timedelta(hours=168), periods=168, freq='h'))]
                response = client.post('/predict', json={'origin':day, 'model':name, 'history':context})
                assert response.status_code == 200
                assert response.json()['target_timestamps_local'][-1] == last
                assert len(response.json()['predicted_ride_counts']) == 24
        assert client.post('/predict', json=request).json() == results['pytorch-mlp']
        def concurrent(name):
            response = client.post('/predict', json={**request,'model':name})
            assert response.status_code == 200 and response.json() == results[name]
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(concurrent, list(refs)*10))
        report['running_containers'] = docker('compose', 'ps', '--format', 'json')
        report['cgroup_memory_high_water_bytes'] = {}
        for service in ['forecast-gateway', 'lstm-worker']:
            report['cgroup_memory_high_water_bytes'][service] = docker(
                'compose', 'exec', '-T', service, 'python', '-c',
                "from pathlib import Path; paths=[Path('/sys/fs/cgroup/memory.peak'), Path('/sys/fs/cgroup/memory/memory.max_usage_in_bytes')]; print(next((p.read_text().strip() for p in paths if p.exists()), 'unsupported by this cgroup/kernel'))").strip()
        docker('compose', 'stop', 'lstm-worker')
        assert client.get('/models').json()['models'][-1]['ready'] is False
        assert client.post('/predict', json={**request,'model':'tensorflow-lstm'}).status_code == 503
        for name in ['pytorch-mlp','xgboost']:
            assert client.post('/predict',json={**request,'model':name}).json() == results[name]
        docker('compose', 'start', 'lstm-worker')
        wait_ready(client)
        assert client.post('/predict',json={**request,'model':'tensorflow-lstm'}).json() == results['tensorflow-lstm']
        report['failure_isolation_and_restart'] = True
        (OUT/'responses.json').write_text(json.dumps(results,indent=2)+'\n')
    report['artifact_hashes_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest()==HASHES[str(p)] for p in FILES)
    assert report['artifact_hashes_unchanged']
    report['success'] = True
except Exception as error:
    report['error'] = repr(error)
    raise
finally:
    try:
        docker('compose','stop')
    finally:
        stop.set(); thread.join(timeout=10)
    report['free_after_bytes'] = shutil.disk_usage(ROOT).free
    report['docker_storage'] = docker('system','df')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'memory_samples'},indent=2))
    print('Evidence:', OUT)
