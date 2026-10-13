"""Run locally: python -m uvicorn citypulse_ai.api.app:app --host 127.0.0.1."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from citypulse_ai.api.model import load_serving_model, MODEL_NAME, MODEL_VERSION
from citypulse_ai.api.xgboost import load_xgboost_model
from citypulse_ai.api.xgboost import MODEL_NAME as XGBOOST_NAME, MODEL_VERSION as XGBOOST_VERSION
from citypulse_ai.api.schemas import PredictionRequest, PredictionResponse, instant, prediction_response
from citypulse_ai.api.lstm_client import LSTMWorkerClient, WorkerError
from citypulse_ai.api.lstm_contract import MODEL_NAME as LSTM_NAME, MODEL_VERSION as LSTM_VERSION
from citypulse_ai.forecasting.xgboost_model import forecast_features


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.models = {}
        for identifier, loader in (("pytorch-mlp", load_serving_model), ("xgboost", load_xgboost_model)):
            try:
                application.state.models[identifier] = loader()
            except Exception:
                logging.getLogger(__name__).exception("Trusted model %s could not be loaded", identifier)
        # Retain the old default-model handle and health semantics.
        application.state.model = application.state.models.get("pytorch-mlp")
        try:
            application.state.lstm_worker = LSTMWorkerClient()
        except ValueError:
            logging.getLogger(__name__).exception("Invalid internal LSTM worker configuration")
        yield
        if application.state.lstm_worker is not None:
            application.state.lstm_worker.close()
        application.state.lstm_worker = None
        application.state.model = None
        application.state.models = {}

    application = FastAPI(title="CityPulse AI citywide model inference", version="0.2.0", lifespan=lifespan)
    application.state.model = None
    application.state.models = {}
    application.state.lstm_worker = None

    @application.get("/models")
    def models():
        listing = [
            {"id": identifier, "model_name": name, "model_version": version,
             "ready": identifier in application.state.models}
            for identifier, name, version in (("pytorch-mlp", MODEL_NAME, MODEL_VERSION),
                                             ("xgboost", XGBOOST_NAME, XGBOOST_VERSION))]
        worker = application.state.lstm_worker
        listing.append({"id": "tensorflow-lstm", "model_name": LSTM_NAME,
                        "model_version": LSTM_VERSION, "ready": worker.ready() if worker else False})
        return {"default_model": "pytorch-mlp", "models": listing}

    @application.get("/health")
    def health():
        ready = application.state.model is not None
        return JSONResponse(status_code=200 if ready else 503, content={
            "application": "citypulse-ai", "application_ready": True,
            "model_ready": ready, "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION if ready else None})

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        if request.model == "tensorflow-lstm":
            if application.state.lstm_worker is None:
                raise HTTPException(503, "LSTM worker is unavailable")
            try:
                return application.state.lstm_worker.predict(request)
            except WorkerError as error:
                raise HTTPException(error.status, error.detail) from error
        model = application.state.models.get(request.model)
        if model is None:
            raise HTTPException(status_code=503, detail=f"Selected model {request.model} is unavailable; check startup logs")
        try:
            frame = forecast_features(request.series(), instant(request.origin))
            predictions = model.predict(frame)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return prediction_response(request.origin, predictions, model.model_name, model.model_version)

    return application


app = create_app()
