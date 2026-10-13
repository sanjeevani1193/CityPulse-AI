"""Real TF/scaler operations with an explicit non-learned tensor test double."""
import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from citypulse_ai.api import lstm_worker as worker
from citypulse_ai.forecasting.lstm import LSTMForecaster
from tests.test_lstm_worker import (
    request, test_worker_dst, test_invalid_worker_history, test_missing_artifact_fails_closed,
)


class TensorContractDouble:
    def __init__(self):
        self.inputs = []

    def __call__(self, inputs, *, training):
        assert training is False and inputs.shape == (1, 48, 1)
        assert inputs.dtype == np.float32
        self.inputs.append(inputs.copy())
        return tf.reduce_mean(inputs, axis=(1, 2))[:, None] + tf.linspace(0., 1., 24)[None, :]


@pytest.fixture
def frozen_forecaster(monkeypatch):
    scaler = StandardScaler()
    # Declared synthetic statistics, never presented as the trained scaler.
    scaler.mean_, scaler.scale_, scaler.var_ = np.array([100.]), np.array([10.]), np.array([100.])
    scaler.n_features_in_, scaler.n_samples_seen_ = 1, 48
    model = TensorContractDouble()
    forecaster = LSTMForecaster(model, scaler)
    calls = []
    def load():
        calls.append(True)
        return forecaster
    monkeypatch.setattr(worker, 'load_lstm', load)
    def forbidden(*args, **kwargs):
        pytest.fail('Training or preprocessing fitting is forbidden in CI')
    monkeypatch.setattr(StandardScaler, 'fit', forbidden)
    monkeypatch.setattr(tf.keras.Model, 'fit', forbidden)
    return forecaster, calls


@pytest.fixture
def client(frozen_forecaster):
    with TestClient(worker.create_worker()) as session:
        assert session.get('/health').status_code == 200
        yield session


def test_tensor_preprocessing_inverse_units_repeatability_and_load_once(client, frozen_forecaster):
    forecaster, calls = frozen_forecaster
    stats = [x.copy() for x in (forecaster.scaler.mean_, forecaster.scaler.scale_, forecaster.scaler.var_)]
    data = request()
    first = client.post('/predict', json=data)
    second = client.post('/predict', json=data)
    assert first.status_code == second.status_code == 200 and first.json() == second.json()
    raw = np.array([r['ride_count'] for r in data['history']], dtype=float).reshape(48, 1)
    np.testing.assert_array_equal(forecaster.model.inputs[-1], ((raw-100.)/10.).astype('float32')[None])
    np.testing.assert_allclose(first.json()['predicted_ride_counts'], raw.mean()+np.linspace(0, 10, 24), atol=1e-5)
    for previous,current in zip(stats, (forecaster.scaler.mean_, forecaster.scaler.scale_, forecaster.scaler.var_)):
        np.testing.assert_array_equal(previous, current)
    assert calls == [True] and len(first.json()['predicted_ride_counts']) == 24


def test_corrupt_worker_artifact_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('CITYPULSE_LSTM_ARTIFACT_DIR', str(tmp_path))
    (tmp_path/'best.weights.h5').write_bytes(b'not trusted weights')
    with pytest.raises(ValueError, match='invalid trusted LSTM artifact'):
        worker.load_lstm()
