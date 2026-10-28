"""Artifact-free station HTTP/feature contracts. Doubles are not real forecasts."""
from types import SimpleNamespace
import hashlib
import json

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from citypulse_ai.api.station_app import create_app, StationLSTMClient
from citypulse_ai.api.station_contract import StationRequest, context, response, tabular
from citypulse_ai.api.station_runtime import ArtifactStore
from citypulse_ai.features.station_dataset import CountScaler, KNOWN_FEATURES, OBSERVED_FEATURES


def payload(origin='2025-08-01T00:00:00-04:00'):
    origin = pd.Timestamp(origin).tz_convert('UTC')
    return {'station_id': '001', 'origin': origin.isoformat(),
            'history': [{'timestamp': t.isoformat(), 'ride_count': i%9}
                        for i, t in enumerate(pd.date_range(origin-pd.Timedelta(hours=336), periods=336, freq='h'))]}


def store():
    scaler = CountScaler(4., 2., 1000, pd.Timestamp('2024-01-01T05:00:00Z'), pd.Timestamp('2025-07-01T04:00:00Z'))
    return SimpleNamespace(codes={'001': 1}, scalers={'001': scaler})


class ContractRuntime:
    def __init__(self): self.store = store(); self.models = {}; self.calls = 0
    def load(self, name): self.models[name] = object()
    def listing(self): return [{'id': m, 'available': True, 'loaded': True, 'ready': True}
                               for m in ['xgboost','pytorch-mlp','pytorch-tft','tensorflow-lstm','lag-24','lag-168']]
    def predict(self, request):
        self.calls += 1
        frame, w = context(request, self.store.codes, self.store.scalers)
        assert not w.target_mask.any() and w.encoder_known.shape == (336, 11)
        values = frame.ride_count.iloc[312:336].to_numpy(dtype=float, na_value=np.nan)
        values = np.nan_to_num(values)  # Declared protocol fixture only, not an imputation adapter.
        q = None if request.model != 'pytorch-tft' else {'q10': np.maximum(values-1,0), 'q50': values, 'q90': values+1}
        return response(request, values, 'synthetic-contract', q)


class UnavailableWorker:
    def ready(self): return False
    def close(self): pass
    def predict(self, request):
        from citypulse_ai.api.lstm_client import WorkerError
        raise WorkerError(503, 'Worker unavailable')


@pytest.fixture
def client():
    with TestClient(create_app(ContractRuntime, UnavailableWorker)) as c: yield c


def test_default_models_stations_quantiles_and_failure_isolation(client):
    assert client.get('/health').status_code == 200
    assert client.get('/models').json()['default_model'] == 'xgboost'
    assert len(client.get('/models').json()['models']) == 6
    assert client.get('/stations').json()['station_ids'] == ['001']
    a = client.post('/predict', json=payload()); b = client.post('/predict', json=payload())
    assert a.status_code == 200 and a.json() == b.json() and a.json()['model'] == 'xgboost'
    assert len(a.json()['predicted_ride_counts']) == 24
    q = client.post('/predict', json={**payload(), 'model': 'pytorch-tft'}).json()
    assert set(q['quantiles']) == {'q10','q50','q90'} and q['prediction_interval']['nominal_coverage'] == .8
    assert client.post('/predict', json={**payload(),'model':'tensorflow-lstm'}).status_code == 503
    for m in ['xgboost','pytorch-mlp','pytorch-tft','lag-24','lag-168']:
        assert client.post('/predict', json={**payload(),'model':m}).status_code == 200


@pytest.mark.parametrize('fault', ['station','model','future','duplicate','missing','negative','bool','naive','midday','publication','status','all_unknown','pretraining'])
def test_invalid_inputs_do_not_reach_inference(client, fault):
    p = payload()
    if fault == 'station': p['station_id']='unknown'
    elif fault == 'model': p['model']='arbitrary'
    elif fault == 'future': p['history'][-1]['timestamp']=p['origin']
    elif fault == 'duplicate': p['history'][1]['timestamp']=p['history'][0]['timestamp']
    elif fault == 'missing': p['history'].pop(10)
    elif fault == 'negative': p['history'][0]['ride_count']=-1
    elif fault == 'bool': p['history'][0]['ride_count']=True
    elif fault == 'naive': p['origin']='2025-08-01T00:00:00'
    elif fault == 'midday': p['origin']='2025-08-01T12:00:00-04:00'
    elif fault == 'publication': p['history'][0]['available_at']='2025-08-02T00:00:00Z'
    elif fault == 'status': p['history'][0]['station_available']=True
    elif fault == 'all_unknown':
        for r in p['history']: r['ride_count']=None
    elif fault == 'pretraining': p=payload('2025-06-01T00:00:00-04:00')
    assert client.post('/predict', json=p).status_code == 422


@pytest.mark.parametrize('origin', ['2025-11-02T00:00:00-04:00','2026-03-08T00:00:00-05:00'])
def test_dst_elapsed_grid_and_unknown_masks(client, origin):
    p=payload(origin);p['history'][0]['ride_count']=None
    result=client.post('/predict',json=p);assert result.status_code==200
    body=result.json();hours=pd.to_datetime(body['target_timestamps_utc'],utc=True)
    assert len(hours)==24 and (hours[1:]-hours[:-1]==pd.Timedelta(hours=1)).all()
    assert len(set(body['utc_offsets']))==2 and body['observed_history_hours']==335
    r=StationRequest.model_validate(p);data=store();frame,w=context(r,data.codes,data.scalers)
    assert not w.encoder_observed[0,1] and w.encoder_observed[0,0]==0
    assert frame.ride_count.iloc[0] is pd.NA and not frame.label_observed_mask.iloc[0]
    assert not frame.availability_known_mask.any() and not w.target_mask.any()
    features=tabular(frame,1);assert features.shape==(24,48)
    assert np.isnan(features[0,23])  # unavailable lag336; observed zeros elsewhere remain zero


def test_artifact_integrity_missing_corrupt_and_symlink(tmp_path):
    codes={str(i):i+1 for i in range(100)}
    d={'format_version':1,'station_codes':codes,'known_features':list(KNOWN_FEATURES),'observed_features':list(OBSERVED_FEATURES),
       'scalers':{s:{'mean':4.,'scale':2.,'sample_count':100,'train_start_utc':'2024-01-01T05:00:00Z',
                     'train_end_utc':'2025-07-01T04:00:00Z'} for s in codes}}
    p=tmp_path/'preprocessing.json';p.write_text(json.dumps(d))
    spec={'artifacts':{'preprocessing.json':{'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}}
    assert len(ArtifactStore(tmp_path,spec).codes)==100
    p.write_text('{}')
    with pytest.raises(ValueError): ArtifactStore(tmp_path,spec)
    with pytest.raises(ValueError): ArtifactStore(tmp_path/'absent',spec)
    p.write_text(json.dumps(d)); other=tmp_path/'symlink';other.mkdir()
    (other/'preprocessing.json').symlink_to(p)
    with pytest.raises(ValueError): ArtifactStore(other,spec)


def test_worker_identity_checked_and_external_url_rejected(monkeypatch):
    monkeypatch.setenv('CITYPULSE_LSTM_WORKER_URL','http://example.com')
    with pytest.raises(ValueError): StationLSTMClient()
    monkeypatch.setenv('CITYPULSE_LSTM_WORKER_URL','http://127.0.0.1:8001')
    def handler(req):
        p=StationRequest.model_validate(json.loads(req.content))
        return httpx.Response(200,json=response(p,np.ones(24),'wrong-version').model_dump())
    c=StationLSTMClient(transport=httpx.MockTransport(handler))
    from citypulse_ai.api.lstm_client import WorkerError
    with pytest.raises(WorkerError) as e: c.predict(StationRequest.model_validate({**payload(),'model':'tensorflow-lstm'}))
    assert e.value.status==502;c.close()


def test_trusted_backend_loads_once_and_failed_model_stays_unready(tmp_path, monkeypatch):
    import sys
    from threading import RLock
    from citypulse_ai.api.station_runtime import StationRuntime, ModelUnavailable
    from citypulse_ai.forecasting.station_xgboost import FEATURES, FEATURE_TYPES
    calls = []
    class Booster:
        feature_names = list(FEATURES)
        feature_types = list(FEATURE_TYPES)
        def load_model(self, path): calls.append(path)
        def num_boosted_rounds(self): return 300
        def set_param(self, settings): assert settings == {'device': 'cpu', 'nthread': 1}
    prep = tmp_path/'preprocessing.json'
    prep.write_text(json.dumps({'station_codes': {'001': 1}, 'feature_names': list(FEATURES)}))
    runtime = StationRuntime.__new__(StationRuntime)
    runtime.store = store()
    runtime.store.file = lambda name: prep
    runtime.store.version = lambda name: 'synthetic'
    runtime.models = {}; runtime.errors = {}; runtime.lock = RLock()
    monkeypatch.setitem(sys.modules, 'xgboost', SimpleNamespace(Booster=Booster))
    assert runtime.load('xgboost') is runtime.load('xgboost') and len(calls) == 1
    # A different invalid frozen schema must fail, not become an alternate model.
    runtime.models.clear(); Booster.feature_names = ['wrong']
    with pytest.raises(ModelUnavailable): runtime.load('xgboost')
    assert 'xgboost' not in runtime.models
    listed = {v['id']: v for v in runtime.listing()}
    assert not listed['xgboost']['ready'] and listed['lag-24']['ready']
