"""Gateway protocol fixtures only; TensorFlow belongs to the separate worker."""

import json
import sys
import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient
from citypulse_ai.api import app as api
from citypulse_ai.api.lstm_client import LSTMWorkerClient
from citypulse_ai.api.lstm_contract import MODEL_NAME, MODEL_VERSION
from citypulse_ai.api.schemas import prediction_response
from tests.test_inference_api import payload


def test_gateway_forwards_only_48_validated_hours_and_reports_readiness(monkeypatch):
    seen = []
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"model_ready":True,"model_name":MODEL_NAME,"model_version":MODEL_VERSION})
        data = json.loads(request.content);seen.append(data)
        return httpx.Response(200, json=prediction_response(data["origin"], [1.0]*24, MODEL_NAME, MODEL_VERSION).model_dump())
    monkeypatch.setattr(api, "LSTMWorkerClient", lambda:LSTMWorkerClient(transport=httpx.MockTransport(handler)))
    with TestClient(api.create_app()) as client:
        assert client.get("/models").json()["models"][-1]["ready"] is True
        for hours in [48,168]:
            request = {**payload(hours=hours),"model":"tensorflow-lstm"}
            result = client.post("/predict",json=request)
            assert result.status_code == 200 and len(result.json()["predicted_ride_counts"])==24
            assert seen[-1] == {"origin":request["origin"],"history":request["history"][-48:]}
        for model in ["pytorch-mlp","xgboost"]:
            assert client.post("/predict",json={**payload(hours=48),"model":model}).status_code==422
    assert "tensorflow" not in sys.modules


@pytest.mark.parametrize("failure,status", [("connection",503),("timeout",504),("not_ready",503),
    ("http_error",502),("bad_json",502),("wrong_version",502),("wrong_grid",502),("short",502),("negative",502),("nan",502)])
def test_gateway_failure_isolation(monkeypatch,failure,status):
    def handler(request):
        if failure=="connection":raise httpx.ConnectError("offline")
        if failure=="timeout":raise httpx.ReadTimeout("slow")
        if failure=="not_ready":return httpx.Response(503,json={"model_ready":False})
        if failure=="http_error":return httpx.Response(500)
        if failure=="bad_json":return httpx.Response(200,text="invalid")
        data=json.loads(request.content)
        result=prediction_response(data["origin"],[1.0]*24,MODEL_NAME,MODEL_VERSION).model_dump()
        if failure=="wrong_version":result['model_version']='untrusted'
        if failure=="wrong_grid":result['utc_offsets']=['+00:00']*24
        if failure=="short":result['predicted_ride_counts']=[1.0]
        if failure=="negative":result['predicted_ride_counts'][0]=-1
        if failure=="nan":return httpx.Response(200,text=json.dumps({**result,'predicted_ride_counts':[float('nan')]*24}))
        return httpx.Response(200,json=result)
    monkeypatch.setattr(api,"LSTMWorkerClient",lambda:LSTMWorkerClient(transport=httpx.MockTransport(handler)))
    with TestClient(api.create_app()) as client:
        assert client.post('/predict',json={**payload(hours=48),'model':'tensorflow-lstm'}).status_code==status
        assert client.get('/health').status_code==200
        assert client.post('/predict',json=payload()).status_code==200


def test_worker_readiness_changes_without_gateway_restart(monkeypatch):
    state={'ready':False}
    def handler(request):
        return httpx.Response(200,json={'model_ready':state['ready'],'model_name':MODEL_NAME,'model_version':MODEL_VERSION})
    monkeypatch.setattr(api,'LSTMWorkerClient',lambda:LSTMWorkerClient(transport=httpx.MockTransport(handler)))
    with TestClient(api.create_app()) as client:
        assert client.get('/models').json()['models'][-1]['ready'] is False
        state['ready']=True
        assert client.get('/models').json()['models'][-1]['ready'] is True


@pytest.mark.parametrize('url',['https://127.0.0.1:8001','http://example.com','http://user:password@localhost:8001','http://localhost:8001/path','http://localhost:bad'])
def test_worker_url_restricted_to_operator_configured_loopback(monkeypatch,url):
    monkeypatch.setenv('CITYPULSE_LSTM_WORKER_URL',url)
    with pytest.raises(ValueError):LSTMWorkerClient()


@pytest.mark.parametrize('case',['future','gap','duplicate','midnight','negative'])
def test_invalid_lstm_inputs_never_reach_worker(monkeypatch,case):
    calls=[]
    def handler(request):calls.append(request);raise AssertionError('Invalid data reached worker')
    monkeypatch.setattr(api,'LSTMWorkerClient',lambda:LSTMWorkerClient(transport=httpx.MockTransport(handler)))
    request={**payload(hours=49),'model':'tensorflow-lstm'}
    if case=='future':request['history'][-1]['timestamp']=request['origin']
    elif case=='gap':del request['history'][20]
    elif case=='duplicate':request['history'][1]=request['history'][0]
    elif case=='midnight':request['origin']='2024-03-01T01:00:00-05:00'
    else:request['history'][0]['ride_count']=-1
    with TestClient(api.create_app()) as client:
        assert client.post('/predict',json=request).status_code==422
    assert calls==[]
