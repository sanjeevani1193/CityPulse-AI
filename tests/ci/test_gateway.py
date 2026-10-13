"""Artifact-free gateway contracts. Test doubles are not trained model predictions."""
import httpx
import numpy as np
import pandas as pd
import pytest
import torch  # Required runtime: missing imports must fail collection.
import xgboost
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from citypulse_ai.api import app as api
from citypulse_ai.api.lstm_client import LSTMWorkerClient
from citypulse_ai.api.model import load_serving_model, MODEL_NAME, MODEL_VERSION
from citypulse_ai.api.xgboost import load_xgboost_model
from citypulse_ai.forecasting.xgboost_model import FEATURES
from tests.test_inference_api import (
    payload, test_dst_elapsed_hours_and_calendar_features,
    test_invalid_inputs_fail_closed,
)
from tests.test_multimodel_api import (
    test_model_listing_and_backward_compatible_default,
    test_shared_contract_dst_and_repeated_requests,
    test_identical_issuance_validation, test_unknown_models_rejected,
)
from tests.test_lstm_gateway import (
    test_gateway_forwards_only_48_validated_hours_and_reports_readiness,
    test_gateway_failure_isolation, test_worker_readiness_changes_without_gateway_restart,
    test_worker_url_restricted_to_operator_configured_loopback,
    test_compose_hostname_requires_explicit_operator_opt_in,
    test_invalid_lstm_inputs_never_reach_worker,
)


class ContractModel:
    def __init__(self, name, version):
        self.model_name, self.model_version = name, version
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        assert len(frame) == 24 and list(frame[FEATURES].columns) == FEATURES
        # Explicit protocol fixture, not a benchmark or a substitute model.
        return frame.lag_24.to_numpy(dtype=float)


@pytest.fixture(autouse=True)
def isolated_models(monkeypatch):
    calls = {'pytorch-mlp': 0, 'xgboost': 0}
    models = {'pytorch-mlp': ContractModel(MODEL_NAME, MODEL_VERSION),
              'xgboost': ContractModel(api.XGBOOST_NAME, api.XGBOOST_VERSION)}
    def loader(name):
        calls[name] += 1
        return models[name]
    monkeypatch.setattr(api, 'load_serving_model', lambda: loader('pytorch-mlp'))
    monkeypatch.setattr(api, 'load_xgboost_model', lambda: loader('xgboost'))
    # No actual worker or external HTTP; protocol tests supply their own transport.
    monkeypatch.setattr(api, 'LSTMWorkerClient', lambda: LSTMWorkerClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))))
    def forbidden(*args, **kwargs):
        pytest.fail('Training or preprocessing fitting is forbidden in CI')
    monkeypatch.setattr(StandardScaler, 'fit', forbidden)
    monkeypatch.setattr(xgboost.XGBRegressor, 'fit', forbidden)
    return models, calls


@pytest.fixture
def client(isolated_models):
    with TestClient(api.create_app()) as session:
        yield session


def test_features_causal_and_models_loaded_once(client, isolated_models):
    assert FEATURES == ['hour', 'weekday', 'is_weekend', 'hour_sin', 'hour_cos',
                        'weekday_sin', 'weekday_cos', 'lag_24', 'lag_168']
    models, calls = isolated_models
    request = payload()
    history = pd.Series([r['ride_count'] for r in request['history']],
                        index=pd.DatetimeIndex([r['timestamp'] for r in request['history']]))
    for name in models:
        for _ in range(2):
            response = client.post('/predict', json={**request, 'model': name})
            assert response.status_code == 200
        frame = models[name].frames[-1]
        for lag in (24, 168):
            source = pd.DatetimeIndex(frame.hour_start) - pd.Timedelta(hours=lag)
            assert (source < pd.Timestamp(request['origin'])).all()
            np.testing.assert_array_equal(frame[f'lag_{lag}'], history.reindex(source))
    assert calls == {'pytorch-mlp': 1, 'xgboost': 1}


def test_missing_artifacts_are_unready_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'load_serving_model', lambda: load_serving_model(tmp_path))
    monkeypatch.setattr(api, 'load_xgboost_model', lambda: load_xgboost_model(tmp_path))
    with TestClient(api.create_app()) as session:
        assert session.get('/health').status_code == 503
        assert not any(row['ready'] for row in session.get('/models').json()['models'])
        for name in ('pytorch-mlp', 'xgboost', 'tensorflow-lstm'):
            assert session.post('/predict', json={**payload(), 'model': name}).status_code == 503


@pytest.mark.parametrize('loader,names', [
    (load_serving_model, ('best_model.pt', 'experiment.json')),
    (load_xgboost_model, ('model.ubj', 'model_manifest.json')),
])
def test_corrupt_artifact_rejected_before_deserialization(tmp_path, loader, names):
    for name in names:
        (tmp_path / name).write_bytes(b'not trusted model data')
    with pytest.raises(ValueError, match='checksum mismatch'):
        loader(tmp_path)
