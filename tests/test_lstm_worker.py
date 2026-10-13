"""Frozen real LSTM restoration and HTTP tests in the isolated TF environment."""

import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('fastapi')
tf=pytest.importorskip('tensorflow')
from tensorflow import keras
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from citypulse_ai.api import lstm_worker as worker
from citypulse_ai.api.lstm_contract import ARTIFACT_DIRECTORY,WEIGHTS_SHA256,RECORD_SHA256


def request(origin='2024-03-01T00:00:00-05:00'):
    end=pd.Timestamp(origin).tz_convert('UTC')
    return {'origin':origin,'history':[{'timestamp':t.isoformat(),'ride_count':100+i%24}
        for i,t in enumerate(pd.date_range(end-pd.Timedelta(hours=48),periods=48,freq='h'))]}


@pytest.fixture(scope='module')
def client():
    with TestClient(worker.create_worker()) as session:
        assert session.get('/health').status_code==200
        yield session


def test_restoration_parity_scaler_freeze_and_repeatability(client,monkeypatch):
    history_path=Path('outputs/phase3-xgboost-jan-feb-2024/recovered-83a9eaa90d964c889a03c7b6ec7de09d/recovery_history.json')
    history=json.loads(history_path.read_text())['history'];origin=pd.Timestamp('2024-03-01T00:00:00-05:00')
    last48=[r for r in history if origin-pd.Timedelta(hours=48)<=pd.Timestamp(r['timestamp'])<origin]
    record=json.loads((ARTIFACT_DIRECTORY/'experiment.json').read_text());forecaster=client.app.state.forecaster
    before=[w.copy() for w in forecaster.model.get_weights()]
    assert forecaster.model.input_shape==(None,48,1) and forecaster.model.output_shape==(None,24)
    for attr,key in [('mean_','scaler_mean'),('scale_','scaler_scale'),('var_','scaler_variance')]:
        np.testing.assert_array_equal(getattr(forecaster.scaler,attr),record[key])
    def forbidden(*a,**k):pytest.fail('Training/scaler fitting is forbidden')
    monkeypatch.setattr(keras.Model,'fit',forbidden);monkeypatch.setattr(StandardScaler,'fit',forbidden)
    data={'origin':origin.isoformat(),'history':last48}
    first=client.post('/predict',json=data);second=client.post('/predict',json=data)
    assert first.status_code==second.status_code==200 and first.json()==second.json()
    np.testing.assert_array_equal(first.json()['predicted_ride_counts'],[r['prediction'] for r in record['example_forecast']])
    assert all(np.array_equal(a,b) for a,b in zip(before,forecaster.model.get_weights()))
    assert hashlib.sha256((ARTIFACT_DIRECTORY/'best.weights.h5').read_bytes()).hexdigest()==WEIGHTS_SHA256
    assert hashlib.sha256((ARTIFACT_DIRECTORY/'experiment.json').read_bytes()).hexdigest()==RECORD_SHA256


@pytest.mark.parametrize('origin,final',[('2024-03-10T00:00:00-05:00','2024-03-11T00:00:00-04:00'),('2024-11-03T00:00:00-04:00','2024-11-03T22:00:00-05:00')])
def test_worker_dst(client,origin,final):
    result=client.post('/predict',json=request(origin));assert result.status_code==200
    data=result.json();assert len(data['predicted_ride_counts'])==24
    assert data['target_timestamps_local'][-1]==final
    assert set(data['utc_offsets'])=={'-04:00','-05:00'}


@pytest.mark.parametrize('case',['short','extra','future','gap','duplicate','negative','fraction','naive','midnight','arbitrary_url'])
def test_invalid_worker_history(client,case):
    data=request()
    if case=='short':data['history']=data['history'][:-1]
    elif case=='extra':data['history'].append(data['history'][-1])
    elif case=='future':data['history'][-1]['timestamp']=data['origin']
    elif case=='gap':data['history'][20]['timestamp']=data['history'][19]['timestamp']
    elif case=='duplicate':data['history'][1]=data['history'][0]
    elif case=='negative':data['history'][0]['ride_count']=-1
    elif case=='fraction':data['history'][0]['ride_count']=1.5
    elif case=='naive':data['origin']='2024-03-01T00:00:00'
    elif case=='midnight':data['origin']='2024-03-01T01:00:00-05:00'
    else:data['worker_url']='http://example.com'
    assert client.post('/predict',json=data).status_code==422


def test_model_loaded_once(monkeypatch):
    calls=[];original=worker.load_lstm
    def loader():calls.append(True);return original()
    monkeypatch.setattr(worker,'load_lstm',loader)
    with TestClient(worker.create_worker()) as session:
        assert session.post('/predict',json=request()).status_code==200
        assert session.post('/predict',json=request()).status_code==200
    assert len(calls)==1


def test_missing_artifact_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setenv('CITYPULSE_LSTM_ARTIFACT_DIR',str(tmp_path))
    with TestClient(worker.create_worker()) as session:
        assert session.get('/health').status_code==503
        assert session.post('/predict',json=request()).status_code==503
