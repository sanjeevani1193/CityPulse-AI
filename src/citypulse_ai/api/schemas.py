"""Strict hourly observation contract with explicit timestamp offsets."""

import pandas as pd
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from citypulse_ai.forecasting.baselines import LOCAL_ZONE


def instant(value: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is None:
            raise ValueError("Timestamp must include an explicit UTC offset or Z")
        result = result.tz_convert("UTC")
        if result != result.floor("h"):
            raise ValueError("Timestamp must be an exact elapsed-hour boundary")
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Use a valid timezone-aware ISO timestamp on an hour boundary") from error


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: StrictStr
    ride_count: int = Field(strict=True, ge=0, le=2**53 - 1)


class HistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: StrictStr
    history: list[Observation] = Field(min_length=48, max_length=8760)

    @model_validator(mode="after")
    def validate_history(self):
        origin = instant(self.origin)
        local = origin.tz_convert(LOCAL_ZONE)
        if local != local.normalize():
            raise ValueError("Forecast origin must be America/New_York local midnight")
        hours = pd.DatetimeIndex([instant(item.timestamp) for item in self.history])
        if hours.has_duplicates:
            raise ValueError("Duplicate input hours are not allowed, including equivalent offsets")
        if not hours.is_monotonic_increasing:
            raise ValueError("History must be in increasing chronological order")
        if not (hours < origin).all():
            raise ValueError("All historical hours must precede forecast issuance")
        if hours[-1] != origin - pd.Timedelta(hours=1):
            raise ValueError("History must end at the hour immediately preceding issuance")
        if not ((hours[1:] - hours[:-1]) == pd.Timedelta(hours=1)).all():
            raise ValueError("Missing hourly observations are not allowed; never infer zeros")
        return self

    def series(self) -> pd.Series:
        return pd.Series([item.ride_count for item in self.history],
                         index=pd.DatetimeIndex([instant(item.timestamp) for item in self.history]))


class PredictionRequest(HistoryRequest):
    model: Literal["pytorch-mlp", "xgboost", "tensorflow-lstm"] = "pytorch-mlp"

    @model_validator(mode="after")
    def validate_model_history(self):
        if self.model != "tensorflow-lstm" and len(self.history) < 168:
            raise ValueError("MLP and XGBoost require at least 168 historical hours")
        return self


class LSTMRequest(HistoryRequest):
    history: list[Observation] = Field(min_length=48, max_length=48)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    origin: str
    timezone: str
    model_name: str
    model_version: str
    target_timestamps_utc: list[str] = Field(min_length=24, max_length=24)
    target_timestamps_local: list[str] = Field(min_length=24, max_length=24)
    utc_offsets: list[str] = Field(min_length=24, max_length=24)
    predicted_ride_counts: list[float] = Field(min_length=24, max_length=24)


def prediction_response(origin, predictions, model_name, model_version) -> PredictionResponse:
    origin = instant(str(origin))
    targets = pd.date_range(origin, periods=24, freq="h")
    local = targets.tz_convert(LOCAL_ZONE)
    return PredictionResponse(
        origin=origin.tz_convert(LOCAL_ZONE).isoformat(), timezone=LOCAL_ZONE,
        model_name=model_name, model_version=model_version,
        target_timestamps_utc=[value.isoformat() for value in targets],
        target_timestamps_local=[value.isoformat() for value in local],
        utc_offsets=[value.strftime("%z")[:3] + ":" + value.strftime("%z")[3:] for value in local],
        predicted_ride_counts=list(predictions))
