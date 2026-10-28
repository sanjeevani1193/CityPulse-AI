"""Real API checks on the approved validation demo, never held-out test inference."""
import argparse
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from citypulse_ai.api.station_contract import StationResponse, MODELS


def verify(url, demo, output, models=MODELS):
    demo = Path(demo); payload = json.loads((demo/'demo-request.json').read_text())
    refs = json.loads((demo/'demo-references.json').read_text())
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != refs['request_sha256']:
        raise ValueError('Approved development request differs')
    report = {'fixture': {'station_id': payload['station_id'], 'origin': payload['origin'],
                         'scope': 'validation only'}, 'models': {}, 'rtol': 1e-5, 'atol': 1e-4}
    with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
        health = client.get('/health'); health.raise_for_status()
        report['health'] = health.json(); report['model_listing'] = client.get('/models').json()
        for model in models:
            result = client.post('/predict', json={**payload, 'model': model}); result.raise_for_status()
            body = StationResponse.model_validate(result.json())
            expected_hours = pd.date_range(pd.Timestamp(payload['origin']).tz_convert('UTC'), periods=24, freq='h')
            if not pd.DatetimeIndex(pd.to_datetime(body.target_timestamps_utc, utc=True)).equals(expected_hours):
                raise ValueError('Returned elapsed-hour grid differs')
            if body.model != model or body.station_id != payload['station_id'] or body.timezone != 'America/New_York':
                raise ValueError('Returned forecast identity differs')
            row = {'status': result.status_code, 'response': body.model_dump()}
            if model in refs['expected']:
                expected = refs['expected'][model]
                np.testing.assert_allclose(body.predicted_ride_counts, expected, rtol=1e-5, atol=1e-4)
                row['saved_validation_or_lag_reference_max_abs_error'] = float(np.max(np.abs(np.asarray(body.predicted_ride_counts)-expected)))
            else:
                row['saved_per_target_validation_reference_available'] = False
            again = client.post('/predict', json={**payload, 'model': model}); again.raise_for_status()
            if again.json() != result.json(): raise ValueError('Repeated inference changed')
            report['models'][model] = row
        bad = json.loads(json.dumps(payload)); bad['history'][-1]['timestamp'] = payload['origin']
        if client.post('/predict', json=bad).status_code != 422:
            raise ValueError('Future input was not rejected')
        report['future_input_status'] = 422
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return {m: {'status': r['status'], 'forecasts': len(r['response']['predicted_ride_counts']),
                'reference_error': r.get('saved_validation_or_lag_reference_max_abs_error')} for m, r in report['models'].items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--demo', type=Path, default=Path('outputs/forecasting-v1'))
    p.add_argument('--output', type=Path, default=Path('outputs/forecasting-v1/api-smoke.json'))
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS)); a = p.parse_args()
    print(json.dumps(verify(a.url, a.demo, a.output, a.models), indent=2))
