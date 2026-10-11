"""January-only XGBoost development experiment with midnight forecast issuance."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from citypulse_ai.forecasting.baselines import LOCAL_ZONE, error_metrics, utc_boundary
from citypulse_ai.forecasting.experiment import START, END

VALIDATION_START = pd.Timestamp("2024-02-01", tz=LOCAL_ZONE)
TEST_START = pd.Timestamp("2024-02-15", tz=LOCAL_ZONE)
FEATURES = ["hour", "weekday", "is_weekend", "hour_sin", "hour_cos",
            "weekday_sin", "weekday_cos", "lag_24", "lag_168"]
CONFIGURATIONS = (
    {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "subsample": 1.0},
    {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0},
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
)


def forecast_features(series: pd.Series, origin, *, allow_missing=False) -> pd.DataFrame:
    """Use target-time calendar values and lag observations known at issuance.

    Target demand is not read here. The optional missing mode is for dropping
    unavailable training rows, never imputing future demand or prediction inputs.
    """
    origin = utc_boundary(origin)
    local_origin = origin.tz_convert(LOCAL_ZONE)
    if local_origin != local_origin.normalize():
        raise ValueError("Forecast must be issued at NYC local midnight")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None or series.index.has_duplicates:
        raise ValueError("Demand history needs a unique timezone-aware hourly index")
    if series.index.hasnans or not series.index.equals(series.index.floor("h")):
        raise ValueError("History timestamps must be nonmissing hour boundaries")
    targets = pd.date_range(origin, periods=24, freq="h")
    local = targets.tz_convert(LOCAL_ZONE)
    frame = pd.DataFrame({"origin": origin, "hour_start": targets, "hour": local.hour,
                          "weekday": local.dayofweek, "is_weekend": (local.dayofweek >= 5).astype(int)})
    for name, values, period in [("hour", local.hour, 24), ("weekday", local.dayofweek, 7)]:
        frame[name + "_sin"] = np.sin(2 * np.pi * values / period)
        frame[name + "_cos"] = np.cos(2 * np.pi * values / period)
    for lag in (24, 168):
        source_hours = targets - pd.Timedelta(hours=lag)
        if not (source_hours < origin).all():
            raise ValueError("Lag source unavailable at forecast issuance")
        values = series.reindex(source_hours).to_numpy(dtype=float, na_value=np.nan)
        known = values[~np.isnan(values)]
        if not np.isfinite(known).all() or (known < 0).any():
            raise ValueError("Lag demand must be finite and nonnegative")
        frame[f"lag_{lag}"] = values
        frame[f"lag_{lag}_source_hour"] = source_hours
    if not allow_missing and frame[FEATURES].isna().any().any():
        raise ValueError("Required lag observations are missing; do not fill from future data")
    return frame


def training_rows(series: pd.Series) -> tuple[pd.DataFrame, int]:
    """January targets only; discard warm-up rows lacking either historical lag."""
    january = series.loc[(series.index >= START) & (series.index < VALIDATION_START)]
    blocks = []
    for origin in pd.date_range(START, VALIDATION_START, freq="D", inclusive="left"):
        block = forecast_features(january, origin, allow_missing=True)
        block["target"] = january.reindex(pd.DatetimeIndex(block.hour_start)).to_numpy(dtype=float, na_value=np.nan)
        blocks.append(block)
    rows = pd.concat(blocks, ignore_index=True)
    available = rows[FEATURES + ["target"]].notna().all(axis=1)
    kept = rows.loc[available].reset_index(drop=True)
    if kept.empty or not np.isfinite(kept.target).all() or not kept.target.ge(0).all():
        raise ValueError("No valid January training rows or invalid training targets")
    return kept, int((~available).sum())


def forecast_next_day(model, series: pd.Series, origin) -> pd.DataFrame:
    """One frozen model predicts all 24 targets without reading their actuals."""
    frame = forecast_features(series, origin)
    predictions = np.asarray(model.predict(frame[FEATURES]), dtype=float)
    if predictions.shape != (24,) or not np.isfinite(predictions).all():
        raise ValueError("Model must return 24 finite predictions")
    # A predeclared physical constraint, applied identically in validation/test.
    frame["prediction"] = np.maximum(predictions, 0)
    return frame


def evaluate_model(model, series: pd.Series, start, end) -> tuple[pd.DataFrame, dict]:
    left, right = utc_boundary(start), utc_boundary(end)
    local_left, local_right = left.tz_convert(LOCAL_ZONE), right.tz_convert(LOCAL_ZONE)
    if left >= right or local_left != local_left.normalize() or local_right != local_right.normalize():
        raise ValueError("Evaluation requires increasing local midnight boundaries")
    blocks = []
    for origin in pd.date_range(local_left, local_right, freq="D", inclusive="left"):
        block = forecast_next_day(model, series, origin)
        if not block.hour_start.lt(right).all():
            raise ValueError("24 elapsed-hour targets extend outside the evaluation split")
        block["actual"] = series.reindex(pd.DatetimeIndex(block.hour_start)).to_numpy(dtype=float, na_value=np.nan)
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    return frame, {"forecast_origins": frame.origin.nunique(), "target_hours": len(frame),
                   **error_metrics(frame.actual, frame.prediction)}


@dataclass
class SelectedModel:
    model: object
    parameters: dict
    validation_predictions: pd.DataFrame
    validation_metrics: dict
    search_results: pd.DataFrame
    training_rows: int
    dropped_training_rows: int


def select_model(validation_history: pd.Series) -> SelectedModel:
    """Accept reference/validation history only. Never fit on validation or test."""
    if validation_history.empty or not (validation_history.index < TEST_START).all():
        raise ValueError("Selection history must exclude all February 15–29 test observations")
    from xgboost import XGBRegressor

    train, dropped = training_rows(validation_history)
    results, best = [], None
    for number, configuration in enumerate(CONFIGURATIONS):
        parameters = {**configuration, "random_state": 42, "n_jobs": 1, "tree_method": "hist",
                      "objective": "reg:squarederror", "eval_metric": "mae", "importance_type": "gain"}
        model = XGBRegressor(**parameters)
        model.fit(train[FEATURES], train.target)  # No eval_set or early stopping.
        predictions, scores = evaluate_model(model, validation_history, VALIDATION_START, TEST_START)
        results.append({"configuration": number, **configuration, **scores})
        if best is None or scores["MAE"] < best.validation_metrics["MAE"]:
            # Exact ties keep the earlier predefined configuration.
            best = SelectedModel(model, parameters, predictions, scores, pd.DataFrame(), len(train), dropped)
    best.search_results = pd.DataFrame(results)
    return best


def run_xgboost_experiment(series: pd.Series) -> tuple[SelectedModel, pd.DataFrame, pd.DataFrame]:
    """Select on validation, then evaluate the frozen January-trained model once."""
    selection_history = series.loc[series.index < TEST_START].copy()
    selected = select_model(selection_history)
    test_predictions, test_scores = evaluate_model(selected.model, series, TEST_START, END)
    predictions = pd.concat([selected.validation_predictions.assign(split="validation"),
                             test_predictions.assign(split="test")], ignore_index=True)
    metrics = pd.DataFrame([{"split": "validation", **selected.validation_metrics},
                            {"split": "test", **test_scores}])
    return selected, predictions, metrics
