"""Small, causal seasonal-naive benchmarks for hourly ride-start counts."""

import math

import pandas as pd

LOCAL_ZONE = "America/New_York"


def utc_boundary(value) -> pd.Timestamp:
    """Require an explicit timezone rather than guessing a local instant."""
    instant = pd.Timestamp(value)
    if pd.isna(instant) or instant.tzinfo is None:
        raise ValueError("An explicit timezone-aware boundary is required")
    return instant.tz_convert("UTC")


def prepare_citywide(observed: pd.Series, start, end, *, assume_missing_zero=False) -> pd.DataFrame:
    """Complete the elapsed-hour grid; unknown hours stay unknown by default.

    Explicit zero filling means assuming missing records denote zero validated
    starts, not zero underlying demand. It does not establish archive coverage.
    """
    left, right = utc_boundary(start), utc_boundary(end)
    if left >= right or left != left.floor("h") or right != right.floor("h"):
        raise ValueError("Use increasing hour-aligned boundaries")
    if not isinstance(observed.index, pd.DatetimeIndex) or observed.index.tz is None:
        raise ValueError("Observed hours must have a timezone-aware DatetimeIndex")
    counts = observed.copy()
    counts.index = counts.index.tz_convert("UTC")
    if counts.index.hasnans or counts.index.has_duplicates or not counts.index.equals(counts.index.floor("h")):
        raise ValueError("Observed hours must be unique, nonmissing hour boundaries")
    if not ((counts.index >= left) & (counts.index < right)).all():
        raise ValueError("Observed hours fall outside the reporting window")
    counts = pd.to_numeric(counts, errors="raise")
    known = counts.dropna()
    if not known.map(math.isfinite).all() or not known.ge(0).all() or not known.mod(1).eq(0).all():
        raise ValueError("Ride counts must be finite nonnegative integers")
    grid = pd.date_range(left, right, freq="h", inclusive="left")
    frame = counts.reindex(grid).astype("Int64").rename("ride_count").rename_axis("hour_start").reset_index()
    frame["is_observed"] = frame.ride_count.notna()
    frame["is_zero_filled"] = ~frame.is_observed & assume_missing_zero
    if assume_missing_zero:
        frame["ride_count"] = frame.ride_count.fillna(0)
    if frame.ride_count.sum() != known.sum():
        raise ValueError("Hourly grid failed count conservation")
    return frame


def seasonal_naive(series: pd.Series, origin, *, lag_hours: int, horizon=24) -> pd.DataFrame:
    """Forecast a fixed block using only observations strictly before its origin."""
    if not isinstance(lag_hours, int) or not isinstance(horizon, int) or horizon <= 0 or lag_hours < horizon:
        raise ValueError("Integer lag must cover the entire positive forecast horizon")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None or series.index.has_duplicates:
        raise ValueError("Series must have a unique timezone-aware hourly index")
    origin = utc_boundary(origin)
    if origin != origin.floor("h"):
        raise ValueError("Forecast origin must be an hour boundary")
    targets = pd.date_range(origin, periods=horizon, freq="h")
    source_hours = targets - pd.Timedelta(hours=lag_hours)
    if not (source_hours < origin).all():
        raise ValueError("Lag would use information unavailable at forecast origin")
    prediction = series.reindex(source_hours)
    if prediction.isna().any():
        raise ValueError("Required lag observations are missing or unavailable")
    if not prediction.map(math.isfinite).all() or not prediction.ge(0).all():
        raise ValueError("Lag counts must be finite and nonnegative")
    return pd.DataFrame({"origin": origin, "hour_start": targets,
                         "source_hour": source_hours, "prediction": prediction.to_numpy(dtype=float)})


def error_metrics(actual, predicted) -> dict:
    """Pooled hourly MAE/RMSE; WAPE is a percentage, undefined for all-zero truth."""
    actual, predicted = pd.Series(actual, dtype=float).reset_index(drop=True), pd.Series(predicted, dtype=float).reset_index(drop=True)
    if actual.empty or len(actual) != len(predicted):
        raise ValueError("Metrics require equal nonempty arrays")
    if any(not values.map(math.isfinite).all() or not values.ge(0).all() for values in (actual, predicted)):
        raise ValueError("Metrics require finite nonnegative values, without missing observations")
    absolute = (actual - predicted).abs()
    denominator = float(actual.sum())
    return {"MAE": float(absolute.mean()), "RMSE": math.sqrt(float(((actual - predicted) ** 2).mean())),
            "WAPE_percent": 100 * float(absolute.sum()) / denominator if denominator else math.nan}


def evaluate_midnights(series: pd.Series, start, end, *, lags=(24, 168)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin evaluation; completed earlier evaluation days become history.

    No within-day updates. End is exclusive; every issued forecast must have all
    24 elapsed-hour targets inside the split and all lag observations available.
    """
    left, right = utc_boundary(start), utc_boundary(end)
    local_left, local_right = left.tz_convert(LOCAL_ZONE), right.tz_convert(LOCAL_ZONE)
    if left >= right or local_left != local_left.normalize() or local_right != local_right.normalize():
        raise ValueError("Evaluation needs increasing NYC local-midnight boundaries")
    if not lags or len(set(lags)) != len(lags):
        raise ValueError("Provide distinct seasonal lags")
    forecasts = []
    for origin in pd.date_range(local_left, local_right, freq="D", inclusive="left"):
        for lag in lags:
            frame = seasonal_naive(series, origin, lag_hours=lag)
            if not frame.hour_start.lt(right).all():
                raise ValueError("A 24-elapsed-hour forecast extends outside the split")
            frame["actual"] = series.reindex(pd.DatetimeIndex(frame.hour_start)).to_numpy(dtype=float, na_value=math.nan)
            if frame.actual.isna().any():
                raise ValueError("Evaluation target observations are missing")
            frame["lag_hours"] = lag
            forecasts.append(frame)
    predictions = pd.concat(forecasts, ignore_index=True)
    scores = [{"lag_hours": lag, "forecast_origins": frame.origin.nunique(), "target_hours": len(frame),
               **error_metrics(frame.actual, frame.prediction)} for lag, frame in predictions.groupby("lag_hours")]
    return predictions, pd.DataFrame(scores)
