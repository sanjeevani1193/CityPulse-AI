"""Run locally: python -m uvicorn citypulse_ai.api.app:app --host 127.0.0.1."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from citypulse_ai.api.model import load_serving_model, MODEL_NAME, MODEL_VERSION
from citypulse_ai.api.schemas import PredictionRequest, PredictionResponse, instant
from citypulse_ai.forecasting.baselines import LOCAL_ZONE
from citypulse_ai.forecasting.xgboost_model import forecast_features


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.model = None
        try:
            application.state.model = load_serving_model()
        except Exception:
            logging.getLogger(__name__).exception("Trusted MLP artifacts could not be loaded")
        yield
        application.state.model = None

    application = FastAPI(title="CityPulse AI citywide MLP inference", version="0.1.0", lifespan=lifespan)
    application.state.model = None

    @application.get("/health")
    def health():
        ready = application.state.model is not None
        return JSONResponse(status_code=200 if ready else 503, content={
            "application": "citypulse-ai", "application_ready": True,
            "model_ready": ready, "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION if ready else None})

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        model = application.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Trusted model is unavailable; check startup logs")
        try:
            frame = forecast_features(request.series(), instant(request.origin))
            predictions = model.predict(frame)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        targets = frame.hour_start.dt.tz_convert("UTC")
        local = targets.dt.tz_convert(LOCAL_ZONE)
        return PredictionResponse(
            origin=instant(request.origin).tz_convert(LOCAL_ZONE).isoformat(), timezone=LOCAL_ZONE,
            model_name=MODEL_NAME, model_version=MODEL_VERSION,
            target_timestamps_utc=[value.isoformat() for value in targets],
            target_timestamps_local=[value.isoformat() for value in local],
            utc_offsets=[value.strftime("%z")[:3] + ":" + value.strftime("%z")[3:] for value in local],
            predicted_ride_counts=predictions.tolist())

    return application


app = create_app()
