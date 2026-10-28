"""Final station100 HTTP contract; no model framework imports."""
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from citypulse_ai.api.schemas import instant, prediction_response
from citypulse_ai.features.station_dataset import build_station_panel, make_window
from citypulse_ai.forecasting.station_xgboost import feature_block

MODELS = ('xgboost', 'pytorch-mlp', 'tensorflow-lstm', 'pytorch-tft', 'lag-24', 'lag-168')


def publication(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Availability timestamps require an explicit offset')
    return result.tz_convert('UTC')


class StationObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    timestamp: StrictStr
    ride_count: int | None = Field(default=None, strict=True, ge=0, le=2**53-1)
    available_at: StrictStr | None = None
    station_available: bool | None = Field(default=None, strict=True)
    status_available_at: StrictStr | None = None


class StationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    station_id: StrictStr
    origin: StrictStr
    model: Literal['xgboost', 'pytorch-mlp', 'tensorflow-lstm', 'pytorch-tft', 'lag-24', 'lag-168'] = 'xgboost'
    history: list[StationObservation] = Field(min_length=336, max_length=336)

    @model_validator(mode='after')
    def causal_history(self):
        origin = instant(self.origin); local = origin.tz_convert('America/New_York')
        if local != local.normalize():
            raise ValueError('Forecast issuance must be America/New_York local midnight')
        if origin < pd.Timestamp('2025-07-01', tz='America/New_York').tz_convert('UTC'):
            raise ValueError('Frozen final models/cohort require issuance after the training period')
        hours = pd.DatetimeIndex([instant(r.timestamp) for r in self.history])
        expected = pd.date_range(origin-pd.Timedelta(hours=336), periods=336, freq='h')
        if not hours.equals(expected):
            raise ValueError('Require 336 ordered, unique consecutive UTC hours ending before issuance; use null counts for unknown observations')
        if not any(r.ride_count is not None for r in self.history):
            raise ValueError('At least one observed historical count is required')
        for hour, row in zip(hours, self.history):
            if row.available_at is not None:
                available = publication(row.available_at)
                if row.ride_count is None or not hour+pd.Timedelta(hours=1) <= available <= origin:
                    raise ValueError('A recorded hourly count must be available after bucket end and by issuance')
            if (row.station_available is None) != (row.status_available_at is None):
                raise ValueError('Known operating status needs its publication timestamp; otherwise both remain unknown')
            if row.status_available_at is not None and publication(row.status_available_at) > origin:
                raise ValueError('Future operating-status publications are unavailable at issuance')
        return self


class StationResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    station_id: str
    model: str
    model_name: str
    model_version: str
    origin: str
    timezone: str
    target_timestamps_utc: list[str] = Field(min_length=24, max_length=24)
    target_timestamps_local: list[str] = Field(min_length=24, max_length=24)
    utc_offsets: list[str] = Field(min_length=24, max_length=24)
    predicted_ride_counts: list[float] = Field(min_length=24, max_length=24)
    quantiles: dict[str, list[float]] | None = None
    prediction_interval: dict | None = None
    observed_history_hours: int
    target_policy: str = 'accepted_recorded_starts_v1'
    postprocessing: str = 'max(0, prediction) after inverse scaling'
    observation_assumption: str = 'Caller-supplied counts available by issuance; retrospective bucket-end assumption when available_at is omitted'
    input_provenance: str = 'Caller assertions; not independently certified archive or station-operation evidence'

    @model_validator(mode='after')
    def finite_forecasts(self):
        values = np.asarray(self.predicted_ride_counts)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Require 24 finite nonnegative forecasts')
        if self.model == 'pytorch-tft':
            if self.quantiles is None or set(self.quantiles) != {'q10', 'q50', 'q90'}:
                raise ValueError('TFT requires all three quantiles')
            q = np.array([self.quantiles[k] for k in ('q10', 'q50', 'q90')])
            if q.shape != (3, 24) or not np.isfinite(q).all() or (q < 0).any() or (np.diff(q, axis=0) < 0).any():
                raise ValueError('Invalid ordered quantiles')
            if not np.array_equal(values, q[1]):
                raise ValueError('TFT point forecasts must equal q50')
            if self.prediction_interval != {'lower': q[0].tolist(), 'upper': q[2].tolist(), 'nominal_coverage': .8}:
                raise ValueError('TFT interval must use q10 and q90')
        elif self.quantiles is not None or self.prediction_interval is not None:
            raise ValueError('Point models do not supply prediction intervals')
        return self


def response(request, values, version, quantiles=None):
    counts = np.maximum(np.asarray(values, dtype=float), 0)
    base = prediction_response(request.origin, counts, 'station100-'+request.model, version).model_dump()
    q = None if quantiles is None else {k: np.maximum(np.asarray(v, dtype=float), 0).tolist() for k, v in quantiles.items()}
    interval = None if q is None else {'lower': q['q10'], 'upper': q['q90'], 'nominal_coverage': .8}
    return StationResponse(**base, station_id=request.station_id, model=request.model, quantiles=q,
        prediction_interval=interval, observed_history_hours=sum(r.ride_count is not None for r in request.history))


def context(request, codes, scalers):
    """Original panel/calendar/window code; future labels stay masked and unused."""
    if request.station_id not in codes:
        raise ValueError('Unknown frozen station ID; consult GET /stations')
    origin = instant(request.origin); start = origin-pd.Timedelta(hours=336)
    rows = [{'station_id': request.station_id, 'hour_start_utc': instant(r.timestamp),
             'ride_count_accepted': r.ride_count} for r in request.history if r.ride_count is not None]
    observed = pd.DataFrame(rows)
    quality = observed[['station_id', 'hour_start_utc']].assign(count_complete=True, evidence_id='caller_asserted_accepted_count')
    statuses = [{'station_id': request.station_id, 'hour_start_utc': instant(r.timestamp),
                 'station_available': r.station_available, 'status_available_at_utc': publication(r.status_available_at)}
                for r in request.history if r.station_available is not None]
    frame = build_station_panel(observed, {request.station_id: codes[request.station_id]}, start=start,
        end=origin+pd.Timedelta(hours=24), quality=quality, status=pd.DataFrame(statuses) if statuses else None)
    window = make_window(frame, request.station_id, origin, split_start=origin,
                         split_end=origin+pd.Timedelta(hours=24), scalers=scalers, split_role='evaluation')
    if window.target_mask.any():
        raise ValueError('Inference must not contain future target labels')
    return frame, window


def tabular(frame, code):
    from citypulse_ai.features.station_dataset import KNOWN_FEATURES
    return feature_block(frame.ride_count.to_numpy(dtype=float, na_value=np.nan),
        frame.label_observed_mask.to_numpy(dtype=bool), frame[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=np.float32),
        frame.station_available.to_numpy(dtype=float, na_value=np.nan),
        frame.availability_known_mask.to_numpy(dtype=bool), 336, code)
