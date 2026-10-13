"""Bounded loopback HTTP adapter; this module never imports TensorFlow."""

import os
from urllib.parse import urlsplit

import httpx
from citypulse_ai.api.lstm_contract import MODEL_NAME, MODEL_VERSION
from citypulse_ai.api.schemas import PredictionResponse, prediction_response


class WorkerError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


class LSTMWorkerClient:
    def __init__(self, *, transport=None):
        url = os.environ.get("CITYPULSE_LSTM_WORKER_URL", "http://127.0.0.1:8001")
        parsed = urlsplit(url)
        _ = parsed.port  # Reject malformed ports before constructing the client.
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("LSTM worker URL must be a fixed loopback HTTP origin")
        self.client = httpx.Client(base_url=url, timeout=httpx.Timeout(5.0, connect=1.0),
                                   limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                                   follow_redirects=False, trust_env=False, transport=transport)

    def close(self):
        self.client.close()

    def ready(self):
        try:
            response = self.client.get("/health")
            body = response.json()
            return (response.status_code == 200 and body.get("model_ready") is True
                    and body.get("model_name") == MODEL_NAME and body.get("model_version") == MODEL_VERSION)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False

    def predict(self, request):
        payload = {"origin": request.origin,
                   "history": [row.model_dump() for row in request.history[-48:]]}
        try:
            response = self.client.post("/predict", json=payload)
        except httpx.TimeoutException as error:
            raise WorkerError(504, "LSTM worker timed out") from error
        except httpx.HTTPError as error:
            raise WorkerError(503, "LSTM worker is unavailable") from error
        if response.status_code == 503:
            raise WorkerError(503, "LSTM worker is not model-ready")
        if response.status_code != 200:
            raise WorkerError(502, "LSTM worker returned an unexpected status")
        try:
            result = PredictionResponse.model_validate(response.json())
            if any(value < 0 for value in result.predicted_ride_counts):
                raise ValueError("Negative worker predictions")
            expected = prediction_response(request.origin, result.predicted_ride_counts, MODEL_NAME, MODEL_VERSION)
            if result != expected:
                raise ValueError("Worker metadata or forecast grid differs from the request contract")
            return result
        except (ValueError, TypeError) as error:
            raise WorkerError(502, "Invalid LSTM worker response") from error
