"""Run locally: python -m uvicorn citypulse_ai.api.app:app --host 127.0.0.1."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from citypulse_ai.api.model import load_serving_model, MODEL_NAME, MODEL_VERSION
from citypulse_ai.api.xgboost import load_xgboost_model
from citypulse_ai.api.xgboost import MODEL_NAME as XGBOOST_NAME, MODEL_VERSION as XGBOOST_VERSION
from citypulse_ai.api.schemas import PredictionRequest, PredictionResponse, instant
from citypulse_ai.forecasting.baselines import LOCAL_ZONE
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
        yield
        application.state.model = None
        application.state.models = {}

    application = FastAPI(title="CityPulse AI citywide model inference", version="0.2.0", lifespan=lifespan)
    application.state.model = None
    application.state.models = {}

    @application.get("/models")
    def models():
        return {"default_model": "pytorch-mlp", "models": [
            {"id": identifier, "model_name": name, "model_version": version,
             "ready": identifier in application.state.models}
            for identifier, name, version in (("pytorch-mlp", MODEL_NAME, MODEL_VERSION),
                                             ("xgboost", XGBOOST_NAME, XGBOOST_VERSION))]}

    @application.get("/health")
    def health():
        ready = application.state.model is not None
        return JSONResponse(status_code=200 if ready else 503, content={
            "application": "citypulse-ai", "application_ready": True,
            "model_ready": ready, "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION if ready else None})

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        model = application.state.models.get(request.model)
        if model is None:
            raise HTTPException(status_code=503, detail=f"Selected model {request.model} is unavailable; check startup logs")
        try:
            frame = forecast_features(request.series(), instant(request.origin))
            predictions = model.predict(frame)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        targets = frame.hour_start.dt.tz_convert("UTC")
        local = targets.dt.tz_convert(LOCAL_ZONE)
        return PredictionResponse(
            origin=instant(request.origin).tz_convert(LOCAL_ZONE).isoformat(), timezone=LOCAL_ZONE,
            model_name=model.model_name, model_version=model.model_version,
            target_timestamps_utc=[value.isoformat() for value in targets],
            target_timestamps_local=[value.isoformat() for value in local],
            utc_offsets=[value.strftime("%z")[:3] + ":" + value.strftime("%z")[3:] for value in local],
            predicted_ride_counts=predictions.tolist())

    return application


app = create_app()
