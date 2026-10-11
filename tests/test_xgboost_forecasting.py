"""January-only fitting and causal midnight feature availability."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from citypulse_ai.forecasting.baselines import LOCAL_ZONE
from citypulse_ai.forecasting.experiment import START, END
from citypulse_ai.forecasting.xgboost_model import (
    FEATURES, CONFIGURATIONS, VALIDATION_START, TEST_START, forecast_features,
    training_rows, forecast_next_day, select_model, run_xgboost_experiment,
)


def history():
    hours = pd.date_range(START, END, freq="h", inclusive="left").tz_convert("UTC")
    local = hours.tz_convert(LOCAL_ZONE)
    return pd.Series(100 + local.hour * 30 + local.dayofweek * 10, index=hours, dtype=float)


def test_feature_target_alignment_and_warmup():
    series = history()
    rows, dropped = training_rows(series)
    assert len(rows) == 576 and dropped == 168
    assert rows.hour_start.min() == pd.Timestamp("2024-01-08", tz=LOCAL_ZONE)
    assert rows.hour_start.max() < VALIDATION_START
    np.testing.assert_array_equal(rows.target, series.reindex(pd.DatetimeIndex(rows.hour_start)))
    for lag in (24, 168):
        assert rows[f"lag_{lag}_source_hour"].lt(rows.origin).all()
        np.testing.assert_array_equal(rows[f"lag_{lag}"], series.reindex(pd.DatetimeIndex(rows[f"lag_{lag}_source_hour"])))
    first = rows.iloc[0]
    assert first.hour == 0 and first.weekday == 0 and first.is_weekend == 0
    assert first.hour_sin == pytest.approx(0) and first.hour_cos == pytest.approx(1)
    assert first.weekday_cos == pytest.approx(1)


def test_future_actual_changes_cannot_change_features_or_training():
    series = history()
    origin = pd.Timestamp("2024-02-15", tz=LOCAL_ZONE)
    before = forecast_features(series, origin)
    changed = series.copy()
    changed.loc[changed.index >= origin] = 999999
    pd.testing.assert_frame_equal(before, forecast_features(changed, origin))
    changed.loc[changed.index >= VALIDATION_START] = 777777
    pd.testing.assert_frame_equal(training_rows(series)[0], training_rows(changed)[0])


def test_missing_training_lags_are_dropped_not_imputed():
    series = history().drop(pd.Timestamp("2024-01-10 12:00", tz=LOCAL_ZONE))
    rows, dropped = training_rows(series)
    assert dropped == 171 and len(rows) == 573
    assert not rows[FEATURES + ["target"]].isna().any().any()
    with pytest.raises(ValueError, match="missing"):
        forecast_features(series, pd.Timestamp("2024-01-11", tz=LOCAL_ZONE))


@pytest.mark.parametrize("origin", ["2024-02-15", "2024-02-29"])
def test_next_day_contract_and_no_target_access(origin):
    origin = pd.Timestamp(origin, tz=LOCAL_ZONE)
    series = history()
    class Model:
        def predict(self, features):
            assert list(features.columns) == FEATURES
            return features.lag_24.to_numpy()
    # Entire forecast day is absent: issuing predictions still works.
    frame = forecast_next_day(Model(), series.loc[series.index < origin], origin)
    assert len(frame) == 24 and frame.hour_start.min() == origin
    assert frame.hour_start.max() == origin.tz_convert("UTC") + pd.Timedelta(hours=23)
    assert frame.lag_24_source_hour.lt(frame.origin).all()
    assert frame.lag_168_source_hour.lt(frame.origin).all()
    assert "target" not in frame and "actual" not in frame


def test_non_midnight_and_unavailable_lag_are_rejected():
    with pytest.raises(ValueError, match="midnight"):
        forecast_features(history(), pd.Timestamp("2024-02-15 01:00", tz=LOCAL_ZONE))
    with pytest.raises(ValueError, match="missing"):
        forecast_features(history(), START)


def test_selection_rejects_test_data_before_fitting():
    with pytest.raises(ValueError, match="exclude"):
        select_model(history())


def test_reproducible_training_and_validation_only_selection():
    series = history().loc[history().index < TEST_START]
    first, second = select_model(series), select_model(series)
    assert first.parameters == second.parameters
    pd.testing.assert_frame_equal(first.search_results, second.search_results)
    np.testing.assert_array_equal(first.validation_predictions.prediction, second.validation_predictions.prediction)
    assert first.training_rows == 576 and first.dropped_training_rows == 168
    assert first.parameters["random_state"] == 42
    assert len(first.search_results) == len(CONFIGURATIONS) == 4
    assert first.validation_predictions.hour_start.ge(VALIDATION_START).all()
    assert first.validation_predictions.hour_start.lt(TEST_START).all()


def test_all_fits_are_january_only_and_selected_model_is_not_refitted(monkeypatch):
    import xgboost
    original = xgboost.XGBRegressor
    expected, _ = training_rows(history())
    fitted = []
    class RecordingRegressor(original):
        def fit(self, X, y, **kwargs):
            pd.testing.assert_frame_equal(X, expected[FEATURES])
            pd.testing.assert_series_equal(y, expected.target)
            assert not kwargs  # No validation/test eval_set or early stopping.
            fitted.append(self)
            return super().fit(X, y)
    monkeypatch.setattr(xgboost, "XGBRegressor", RecordingRegressor)
    selected, predictions, scores = run_xgboost_experiment(history())
    assert len(fitted) == 4 and selected.model in fitted
    assert scores.forecast_origins.tolist() == [14, 15]
    assert scores.target_hours.tolist() == [336, 360]
    assert predictions[predictions.split.eq("test")].hour_start.min() == TEST_START
    assert predictions[predictions.split.eq("test")].hour_start.max() < END


def test_negative_predictions_are_clipped_consistently():
    class Model:
        def predict(self, features):
            return np.full(len(features), -1)
    frame = forecast_next_day(Model(), history(), TEST_START)
    assert frame.prediction.eq(0).all()
