"""Artifact-free station worker transforms; never substitute these for real parity."""
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi.testclient import TestClient

from citypulse_ai.api.station_worker import create_worker, StationLSTM
from tests.ci.test_station_contracts import payload, store


class TensorDouble:
    def __init__(self): self.calls=[]
    def __call__(self, inputs, training):
        assert training is False and inputs['history'].shape==(1,336,16) and inputs['calendar'].shape==(1,24,11)
        self.calls.append(inputs)
        return np.zeros((1,24),dtype='float32')


def test_transform_inverse_and_load_once():
    model=StationLSTM.__new__(StationLSTM);model.store=store();model.store.version=lambda m:'synthetic'
    model.model=TensorDouble();loads=[]
    def loader():loads.append(True);return model
    with TestClient(create_worker(loader)) as c:
        p={**payload(),'model':'tensorflow-lstm'}
        a=c.post('/predict',json=p);b=c.post('/predict',json=p)
        assert a.status_code==200 and a.json()==b.json()
        assert a.json()['predicted_ride_counts']==[4.]*24 and loads==[True]
        assert model.model.calls[0]['history'][0,0,11]==-2.
        assert c.get('/health').json()['model_ready']
        assert c.post('/predict',json=payload()).status_code==422


def test_missing_artifacts_unready(tmp_path,monkeypatch):
    monkeypatch.setenv('CITYPULSE_STATION_ARTIFACT_DIR',str(tmp_path))
    with TestClient(create_worker()) as c:
        assert c.get('/health').status_code==503
        assert c.post('/predict',json={**payload(),'model':'tensorflow-lstm'}).status_code==503
