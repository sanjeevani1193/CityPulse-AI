"""Isolated station LSTM process; never imports PyTorch or XGBoost runtimes."""
from contextlib import asynccontextmanager
import logging
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from citypulse_ai.api.station_contract import StationRequest, StationResponse, context, response
from citypulse_ai.api.station_runtime import ArtifactStore


class StationLSTM:
    def __init__(self):
        self.store = ArtifactStore()
        from citypulse_ai.forecasting.station_lstm import configure_cpu
        import tensorflow as tf
        from tensorflow import keras
        if tf.__version__ != '2.21.0' or keras.__version__ != '3.15.1':
            raise ValueError('Use the verified TensorFlow 2.21.0 / Keras 3.15.1 runtime')
        configure_cpu()
        self.model = keras.models.load_model(self.store.file('lstm/epoch-018.keras'), compile=False, safe_mode=True)
        if self.model.count_params() != 8953 or self.model.output_shape != (None, 24):
            raise ValueError('Frozen LSTM shape differs')

    def predict(self, request):
        _, window = context(request, self.store.codes, self.store.scalers)
        inputs = {'history': np.concatenate([window.encoder_known, window.encoder_observed], axis=-1)[None].astype('float32'),
                  'station': window.static_station_code.reshape(1, 1).astype('int32'),
                  'calendar': window.decoder_known[None].astype('float32')}
        prediction = np.asarray(self.model(inputs, training=False), dtype=float)[0]
        scaler = self.store.scalers[request.station_id]
        return response(request, prediction*scaler.scale+scaler.mean, self.store.version('tensorflow-lstm'))


def create_worker(loader=StationLSTM):
    @asynccontextmanager
    async def lifespan(app):
        try: app.state.model = loader()
        except Exception: logging.getLogger(__name__).exception('Trusted station LSTM unavailable')
        yield
        app.state.model = None

    app = FastAPI(title='CityPulse station LSTM worker', version='1.0.0', lifespan=lifespan)
    app.state.model = None; lock = Lock()

    @app.get('/health')
    def health():
        ready = app.state.model is not None
        return JSONResponse(status_code=200 if ready else 503, content={
            'application': 'citypulse-station-lstm-worker', 'model_ready': ready,
            'model_version': app.state.model.store.version('tensorflow-lstm') if ready else None})

    @app.post('/predict', response_model=StationResponse)
    def predict(request: StationRequest):
        if request.model != 'tensorflow-lstm': raise HTTPException(422, 'This worker serves only tensorflow-lstm')
        if app.state.model is None: raise HTTPException(503, 'Trusted station LSTM unavailable')
        try:
            with lock: return app.state.model.predict(request)
        except ValueError as error: raise HTTPException(422, str(error)) from error

    return app


app = create_worker()
