"""Final station100 gateway. Citywide api.app remains a separate legacy entry."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from citypulse_ai.api.lstm_client import LSTMWorkerClient, WorkerError
from citypulse_ai.api.station_contract import MODELS, StationRequest, StationResponse, response
from citypulse_ai.api.station_runtime import StationRuntime, ModelUnavailable, catalog


class StationLSTMClient(LSTMWorkerClient):
    """Reuse the existing bounded localhost/Compose URL policy and HTTP pool."""
    version = 'epoch-018.keras:'+catalog()['artifacts']['lstm/epoch-018.keras']['sha256'][:12]

    def ready(self):
        try:
            result = self.client.get('/health')
            body = result.json()
            return (result.status_code == 200 and body.get('model_ready') is True
                    and body.get('model_version') == self.version
                    and body.get('application') == 'citypulse-station-lstm-worker')
        except Exception:
            return False

    def predict(self, request):
        import httpx
        try:
            result = self.client.post('/predict', json=request.model_dump())
        except httpx.TimeoutException as error:
            raise WorkerError(504, 'Station LSTM worker timed out') from error
        except httpx.HTTPError as error:
            raise WorkerError(503, 'Station LSTM worker unavailable') from error
        if result.status_code == 503:
            raise WorkerError(503, 'Station LSTM worker not ready')
        if result.status_code != 200:
            raise WorkerError(502, 'Unexpected station LSTM worker status')
        try:
            value = StationResponse.model_validate(result.json())
            expected = response(request, value.predicted_ride_counts, self.version)
            if value != expected:
                raise ValueError('Worker forecast identity/grid differs')
            return value
        except (ValueError, TypeError) as error:
            raise WorkerError(502, 'Invalid station LSTM worker response') from error


def create_app(runtime_factory=StationRuntime, client_factory=StationLSTMClient):
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.runtime = runtime_factory()
            app.state.runtime.load('xgboost')
        except Exception:
            logging.getLogger(__name__).exception('Default station XGBoost unavailable; provision trusted artifacts')
        try:
            app.state.worker = client_factory()
        except ValueError:
            logging.getLogger(__name__).exception('Invalid local worker configuration')
        yield
        if app.state.worker is not None: app.state.worker.close()
        app.state.worker = None; app.state.runtime = None

    app = FastAPI(title='CityPulse AI Forecasting v1 — station100', version='1.0.0', lifespan=lifespan)
    app.state.runtime = None; app.state.worker = None

    @app.get('/health')
    def health():
        runtime = app.state.runtime
        ready = runtime is not None and 'xgboost' in runtime.models
        return JSONResponse(status_code=200 if ready else 503, content={
            'application': 'citypulse-station100', 'application_ready': True, 'model_ready': ready,
            'default_model': 'xgboost', 'version': '1.0.0'})

    @app.get('/models')
    def models():
        values = app.state.runtime.listing() if app.state.runtime else [
            {'id': m, 'available': False, 'loaded': False, 'ready': False} for m in MODELS]
        for value in values:
            if value['id'] == 'tensorflow-lstm':
                ready = app.state.worker.ready() if app.state.worker else False
                value.update(available=ready, loaded=ready, ready=ready, state='ready' if ready else 'unavailable')
        return {'scope': 'station100', 'default_model': 'xgboost', 'encoder_hours': 336,
                'horizon_hours': 24, 'timezone': 'America/New_York', 'models': values}

    @app.get('/stations')
    def stations():
        if app.state.runtime is None: raise HTTPException(503, 'Provision trusted station artifacts')
        return {'station_ids': sorted(app.state.runtime.store.codes), 'selection': 'Jan 2024–Jun 2025 training only',
                'station_operation': 'Unknown without independent evidence'}

    @app.post('/predict', response_model=StationResponse)
    def predict(request: StationRequest):
        runtime = app.state.runtime
        if runtime is None: raise HTTPException(503, 'Provision trusted station artifacts')
        if request.station_id not in runtime.store.codes:
            raise HTTPException(422, 'Unknown frozen station ID; consult GET /stations')
        if request.model == 'tensorflow-lstm':
            if app.state.worker is None: raise HTTPException(503, 'Station LSTM worker unavailable')
            try: return app.state.worker.predict(request)
            except WorkerError as error: raise HTTPException(error.status, error.detail) from error
        try:
            return runtime.predict(request)
        except ModelUnavailable as error:
            logging.getLogger(__name__).exception('Trusted station backend unavailable')
            raise HTTPException(503, 'Trusted selected model unavailable') from error
        except ValueError as error:
            # Input-context failures are distinct from missing/corrupt artifacts.
            if 'artifact' in str(error).lower() or 'checksum' in str(error).lower():
                raise HTTPException(503, 'Trusted selected model unavailable') from error
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logging.getLogger(__name__).exception('Station model inference failed')
            raise HTTPException(503, 'Selected model runtime unavailable; inspect server logs') from error

    return app


app = create_app()
