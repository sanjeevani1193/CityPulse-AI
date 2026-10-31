"""XGBoost-only deployment; original six-model station_app is unchanged."""
# Deployment smoke changes must preserve the frozen station100 inference contract.
from contextlib import asynccontextmanager
import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from citypulse_ai.api.station_contract import StationRequest, StationResponse
from citypulse_ai.api.station_runtime import StationRuntime, ModelUnavailable


class XGBoostRequest(StationRequest):
    model: Literal['xgboost'] = 'xgboost'


def create_app(runtime_factory=StationRuntime):
    @asynccontextmanager
    async def lifespan(app):
        try:
            runtime = runtime_factory()
            runtime.load('xgboost')
            app.state.runtime = runtime
        except Exception:
            logging.getLogger(__name__).exception('Verified XGBoost artifacts unavailable')
            app.state.runtime = None
        yield
        app.state.runtime = None

    app = FastAPI(title='CityPulse station100 — XGBoost deployment', version='1.0.0', lifespan=lifespan)
    app.state.runtime = None

    @app.get('/health')
    def health():
        ready = app.state.runtime is not None and 'xgboost' in app.state.runtime.models
        return JSONResponse(status_code=200 if ready else 503, content={
            'application': 'citypulse-station100', 'application_ready': True,
            'model_ready': ready, 'default_model': 'xgboost', 'version': '1.0.0'})

    def runtime():
        if app.state.runtime is None:
            raise HTTPException(503, 'Provision trusted station artifacts')
        return app.state.runtime

    @app.get('/stations')
    def stations():
        return {'station_ids': sorted(runtime().store.codes),
                'selection': 'Jan 2024–Jun 2025 training only',
                'station_operation': 'Unknown without independent evidence'}

    @app.get('/models')
    def models():
        values = [value for value in runtime().listing() if value['id'] == 'xgboost'] if app.state.runtime else [
            {'id': 'xgboost', 'available': False, 'loaded': False, 'ready': False}]
        return {'scope': 'station100', 'default_model': 'xgboost', 'encoder_hours': 336,
                'horizon_hours': 24, 'timezone': 'America/New_York', 'models': values}

    @app.post('/predict', response_model=StationResponse)
    def predict(request: XGBoostRequest):
        backend = runtime()
        if request.station_id not in backend.store.codes:
            raise HTTPException(422, 'Unknown frozen station ID; consult GET /stations')
        try:
            return backend.predict(request)
        except ModelUnavailable as error:
            raise HTTPException(503, 'Trusted selected model unavailable') from error
        except ValueError as error:
            if 'artifact' in str(error).lower() or 'checksum' in str(error).lower():
                raise HTTPException(503, 'Trusted selected model unavailable') from error
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logging.getLogger(__name__).exception('Station XGBoost inference failed')
            raise HTTPException(503, 'Selected model runtime unavailable; inspect server logs') from error

    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), workers=1)
