"""Run in .venv-tensorflow-311, bound to 127.0.0.1; no gateway imports."""

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from sklearn.preprocessing import StandardScaler

from citypulse_ai.api.lstm_contract import (
    ARTIFACT_DIRECTORY, MODEL_NAME, MODEL_VERSION, WEIGHTS_SHA256, RECORD_SHA256,
)
from citypulse_ai.api.schemas import LSTMRequest, PredictionResponse, prediction_response


def load_lstm():
    directory = Path(os.environ.get("CITYPULSE_LSTM_ARTIFACT_DIR", ARTIFACT_DIRECTORY))
    for name, digest in (("best.weights.h5", WEIGHTS_SHA256), ("experiment.json", RECORD_SHA256)):
        path = directory / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or invalid trusted LSTM artifact: {name}")
    record = json.loads((directory / "experiment.json").read_text())
    import tensorflow as tf
    from tensorflow import keras
    from citypulse_ai.forecasting.lstm import configure_cpu, LSTMForecaster
    if tf.__version__ != record["versions"]["tensorflow"] or keras.__version__ != record["versions"]["keras"]:
        raise ValueError("LSTM runtime differs from the verified TensorFlow/Keras versions")
    configure_cpu(record["settings"]["seed"])
    # Same architecture, without compile/optimizer state: inference needs weights only.
    with tf.device("/CPU:0"):
        model = keras.Sequential([keras.Input(shape=(48, 1)), keras.layers.LSTM(32),
                                  keras.layers.Dense(24)], name="citywide_lstm")
        model.load_weights(directory / "best.weights.h5")
    scaler = StandardScaler()
    scaler.mean_ = np.array(record["scaler_mean"])
    scaler.scale_ = np.array(record["scaler_scale"])
    scaler.var_ = np.array(record["scaler_variance"])
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = record["scaler_training_hours"]
    if (model.input_shape != (None, 48, 1) or model.output_shape != (None, 24)
            or model.count_params() != 5144 or scaler.n_samples_seen_ != 744
            or any(v.shape != (1,) or not np.isfinite(v).all()
                   for v in (scaler.mean_, scaler.scale_, scaler.var_))
            or not (scaler.scale_ > 0).all()):
        raise ValueError("LSTM architecture/scaler contract does not match")
    return LSTMForecaster(model, scaler)


def create_worker() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        try:
            application.state.forecaster = load_lstm()
        except Exception:
            logging.getLogger(__name__).exception("Trusted LSTM could not be loaded")
        yield
        application.state.forecaster = None

    application = FastAPI(title="CityPulse local LSTM worker", lifespan=lifespan)
    application.state.forecaster = None
    inference_lock = Lock()  # Bound simultaneous TensorFlow execution in one worker.

    @application.get("/health")
    def health():
        ready = application.state.forecaster is not None
        return JSONResponse(status_code=200 if ready else 503, content={
            "application": "citypulse-lstm-worker", "model_ready": ready,
            "model_name": MODEL_NAME, "model_version": MODEL_VERSION})

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: LSTMRequest):
        forecaster = application.state.forecaster
        if forecaster is None:
            raise HTTPException(503, "Trusted LSTM is unavailable")
        with inference_lock:
            predictions = forecaster.predict_context(request.series().to_numpy(dtype=float).reshape(48, 1))
        return prediction_response(request.origin, predictions, MODEL_NAME, MODEL_VERSION)

    return application


app = create_worker()
