"""Runner-only frozen validation-demo checks; no training or held-out reader.

Private HTTP payloads remain in the ephemeral workspace. Only the allowlisted
summary is suitable for publishing from a public repository.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path('outputs/forecasting-v1')


def tft_reference():
    import numpy as np
    import torch
    from citypulse_ai.forecasting.tft_training import restore
    from citypulse_ai.forecasting.tft import inverse_quantiles
    from citypulse_ai.api.station_catalog import catalog
    torch.set_num_threads(1)
    _, model, _ = restore(ROOT/'artifacts/tft/epoch-006.pt', catalog()['dataset_manifest_sha256'])
    a = np.load(ROOT/'demo-original-window.npz'); model.eval()
    inputs = (torch.tensor(a['station'], dtype=torch.long), *[
        torch.tensor(a[k], dtype=torch.float32) for k in ['encoder_known', 'encoder_observed', 'decoder_known']])
    with torch.inference_mode():
        q = inverse_quantiles(model(*inputs), torch.tensor([float(a['mean'])]), torch.tensor([float(a['scale'])])).numpy()[0]
    body = json.loads((ROOT/'native-local-private.json').read_text())['models']['pytorch-tft']['response']
    predicted = np.array([body['quantiles'][k] for k in ['q10', 'q50', 'q90']]).T
    np.testing.assert_allclose(predicted, np.maximum(q, 0), rtol=1e-5, atol=1e-4)
    return {'original_forward_max_abs_error': float(np.max(np.abs(predicted-np.maximum(q, 0)))),
            'saved_per_target_TFT_validation_reference': False}


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            values[key] = int(value.strip().split()[0])*1024
    return values


class Verification:
    def __init__(self, worker_python, mode):
        self.worker_python = worker_python; self.mode = mode
        self.children = []; self.handles = []; self.stop = threading.Event()
        self.failure = None
        self.summary = {'mode': mode, 'checks': {}, 'held_out_inference': False,
                        'peak_combined_process_RSS_bytes': 0, 'peak_swap_used_bytes': 0,
                        'minimum_host_available_bytes': host_memory()['MemAvailable'],
                        'minimum_disk_available_bytes': shutil.disk_usage('.').free}

    def monitor(self):
        try:
            self._monitor()
        except Exception:
            self.failure = 'Resource monitor failed closed'
            for child in self.children.copy(): self.terminate(child)

    def _monitor(self):
        while not self.stop.is_set():
            mem = host_memory(); swap = mem['SwapTotal']-mem['SwapFree']
            free = shutil.disk_usage('.').free
            rss = 0
            for child in self.children.copy():
                try:
                    for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                        if line.startswith('VmRSS:'): rss += int(line.split()[1])*1024
                except FileNotFoundError: pass
            if self.mode == 'docker':
                stats = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}'],
                                       capture_output=True, text=True, timeout=10, check=True)
                sizes = []
                for line in stats.stdout.splitlines():
                    value = json.loads(line)['MemUsage'].split('/')[0].strip()
                    import re
                    match = re.fullmatch(r'([0-9.]+)([KMGT]?i?B)', value)
                    units = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3,
                             'kB': 1000, 'MB': 1000**2, 'GB': 1000**3}
                    if not match or match[2] not in units: raise RuntimeError('Cannot parse container memory')
                    sizes.append(float(match[1])*units[match[2]])
                self.summary['peak_container_memory_bytes'] = max(sum(sizes), self.summary.get('peak_container_memory_bytes', 0))
            self.summary['peak_combined_process_RSS_bytes'] = max(rss, self.summary['peak_combined_process_RSS_bytes'])
            self.summary['peak_swap_used_bytes'] = max(swap, self.summary['peak_swap_used_bytes'])
            self.summary['minimum_host_available_bytes'] = min(mem['MemAvailable'], self.summary['minimum_host_available_bytes'])
            self.summary['minimum_disk_available_bytes'] = min(free, self.summary['minimum_disk_available_bytes'])
            if swap or mem['MemAvailable'] < 2*1024**3 or free < 3*1024**3 or rss > 2*1024**3:
                self.failure = 'Runner memory/swap/disk/process-RSS guard failed'
                for child in self.children.copy(): self.terminate(child)
                return
            self.stop.wait(2)

    @staticmethod
    def terminate(child):
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)

    def launch(self, command, name):
        if self.failure: raise RuntimeError(self.failure)
        log = (ROOT/(name+'-private.log')).open('w'); self.handles.append(log)
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        self.children.append(child); return child

    def run(self, command, name):
        child = self.launch(command, name)
        deadline = time.monotonic()+120
        while child.poll() is None:
            if time.monotonic() > deadline: self.terminate(child); raise RuntimeError(name+' timeout')
            time.sleep(.25)
        if self.failure or child.returncode: raise RuntimeError(name+' failed; private runner log retained only in workspace')

    def server(self, python, module, port, name):
        import httpx
        child = self.launch([python, '-m', 'uvicorn', module+':app', '--host', '127.0.0.1', '--port', str(port), '--workers', '1'], name)
        with httpx.Client(trust_env=False, timeout=2) as client:
            deadline = time.monotonic()+90
            while time.monotonic() < deadline:
                if child.poll() is not None or self.failure: raise RuntimeError(name+' startup failed')
                try:
                    if client.get(f'http://127.0.0.1:{port}/health').status_code == 200: return child
                except httpx.HTTPError: pass
                time.sleep(1)
        raise RuntimeError(name+' readiness timeout')

    def smoke(self, models, name):
        from scripts.verify_station_api import verify
        result = verify('http://127.0.0.1:8000', ROOT, ROOT/(name+'-private.json'), models)
        self.summary['checks'][name] = result
        if self.failure: raise RuntimeError(self.failure)

    def invalid_requests(self):
        import copy
        import httpx
        payload = json.loads((ROOT/'demo-request.json').read_text())
        invalid = []
        for key, value in [('station_id', 'not-frozen'), ('model', 'unknown'), ('origin', '2025-08-01T12:00:00-04:00')]:
            invalid.append({**payload, key: value})
        for fault in ['missing', 'duplicate', 'negative', 'future']:
            p = copy.deepcopy(payload)
            if fault == 'missing': p['history'].pop()
            if fault == 'duplicate': p['history'][1]['timestamp'] = p['history'][0]['timestamp']
            if fault == 'negative': p['history'][0]['ride_count'] = -1
            if fault == 'future': p['history'][-1]['timestamp'] = p['origin']
            invalid.append(p)
        with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=15) as client:
            assert all(client.post('/predict', json=p).status_code == 422 for p in invalid)
        self.summary['checks']['invalid_request_cases_rejected'] = len(invalid)

    def execute(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        watcher = threading.Thread(target=self.monitor, daemon=True); watcher.start()
        try:
            if self.mode == 'native':
                gateway = self.server(sys.executable, 'citypulse_ai.api.station_app', 8000, 'native-gateway')
                self.smoke(['xgboost', 'pytorch-mlp', 'pytorch-tft', 'lag-24', 'lag-168'], 'native-local')
                self.invalid_requests(); self.terminate(gateway)
                self.run([sys.executable, '-m', 'citypulse_ai.evaluation.serving_verification', '--mode', 'tft-reference'], 'tft-reference')
                self.summary['checks']['TFT_original_forward'] = json.loads((ROOT/'tft-reference-summary.json').read_text())
                worker = self.server(self.worker_python, 'citypulse_ai.api.station_worker', 8001, 'native-lstm')
                gateway = self.server(sys.executable, 'citypulse_ai.api.station_app', 8000, 'worker-gateway')
                self.smoke(['tensorflow-lstm'], 'native-lstm')
                self.terminate(worker)
                self.isolation()
            else:
                self.smoke(['xgboost', 'pytorch-mlp', 'pytorch-tft', 'lag-24', 'lag-168', 'tensorflow-lstm'], 'docker-all-six')
                self.invalid_requests()
                self.run(['docker', 'compose', '-f', 'compose.station.yaml', '--profile', 'lstm', 'stop', 'lstm-worker'], 'docker-worker-stop')
                self.isolation()
                self.run(['docker', 'compose', '-f', 'compose.station.yaml', '--profile', 'lstm', 'start', '--wait', 'lstm-worker'], 'docker-worker-restart')
                self.smoke(['tensorflow-lstm'], 'docker-worker-recovered')
            if self.failure: raise RuntimeError(self.failure)
            self.summary['success'] = True
        except Exception as error:
            self.summary['success'] = False
            self.summary['failure'] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            raise
        finally:
            self.stop.set(); watcher.join(timeout=5)
            for child in self.children: self.terminate(child)
            for handle in self.handles: handle.close()
            Path('verification-reports').mkdir(exist_ok=True)
            Path(f'verification-reports/{self.mode}.json').write_text(json.dumps(self.summary, indent=2)+'\n')

    def isolation(self):
        import httpx
        payload = json.loads((ROOT/'demo-request.json').read_text())
        with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=20) as client:
            assert client.post('/predict', json={**payload, 'model': 'tensorflow-lstm'}).status_code == 503
            for model in ['xgboost', 'pytorch-mlp']:
                assert client.post('/predict', json={**payload, 'model': model}).status_code == 200
            assert client.get('/health').status_code == 200
        self.summary['checks']['worker_failure_isolated'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['native', 'docker', 'tft-reference'], required=True)
    parser.add_argument('--worker-python', default='.worker/bin/python')
    args = parser.parse_args()
    if args.mode == 'tft-reference':
        (ROOT/'tft-reference-summary.json').write_text(json.dumps(tft_reference()))
    else:
        try: Verification(args.worker_python, args.mode).execute()
        except Exception:
            print('Serving verification failed; see sanitized summary. Private inputs/logs are not uploaded.', file=sys.stderr)
            sys.exit(1)
