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


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: StrictStr
    model: Literal["pytorch-mlp", "xgboost"] = "pytorch-mlp"
    history: list[Observation] = Field(min_length=168, max_length=8760)

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


class PredictionResponse(BaseModel):
    origin: str
    timezone: str
    model_name: str
    model_version: str
    target_timestamps_utc: list[str]
    target_timestamps_local: list[str]
    utc_offsets: list[str]
    predicted_ride_counts: list[float]
