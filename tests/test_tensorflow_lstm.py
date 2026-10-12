"""Causal daily sequence alignment, scaling, CPU fitting and weight restoration."""

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip('tensorflow')
from citypulse_ai.forecasting.baselines import LOCAL_ZONE, error_metrics
from citypulse_ai.forecasting.experiment import START, END
from citypulse_ai.forecasting.xgboost_model import VALIDATION_START, TEST_START
from citypulse_ai.forecasting.lstm import (
    configure_cpu, prepare_sequences, observed_context, build_model,
    fit_lstm, forecast_next_day, run_lstm_experiment, LSTMForecaster,
)

configure_cpu()


def history():
    hours = pd.date_range(START, END, freq='h', inclusive='left').tz_convert('UTC')
    return pd.Series(np.arange(len(hours)) + 100, index=hours, dtype=float)


def selection_history():
    series = history()
    return series.loc[series.index < TEST_START]


def test_shapes_alignment_and_chronological_splits():
    series = history()
    january = series.loc[series.index < VALIDATION_START]
    train = prepare_sequences(january, START, VALIDATION_START, drop_initial_warmup=True)
    validation = prepare_sequences(series, VALIDATION_START, TEST_START)
    test = prepare_sequences(series, TEST_START, END)
    assert train.inputs.shape == (29, 48, 1) and train.targets.shape == (29, 24)
    assert validation.inputs.shape == (14, 48, 1) and test.inputs.shape == (15, 48, 1)
    assert train.dropped_warmup == 2
    assert train.origins.min() == START + pd.Timedelta(days=2)
    assert train.origins.max() < VALIDATION_START <= validation.origins.min()
    assert validation.origins.max() < TEST_START <= test.origins.min()
    for split in (train, validation, test):
        for origin, context, truth in zip(split.origins, split.inputs, split.targets):
            assert origin.tz_convert(LOCAL_ZONE).hour == 0
            np.testing.assert_array_equal(context.ravel(), series.loc[origin-pd.Timedelta(hours=48):origin-pd.Timedelta(hours=1)])
            np.testing.assert_array_equal(truth, series.loc[origin:origin+pd.Timedelta(hours=23)])
    assert test.origins[-1] == pd.Timestamp('2024-02-29', tz=LOCAL_ZONE)


def test_future_actuals_cannot_change_context_and_completed_days_are_available():
    original = history()
    changed = original.copy()
    changed.loc[changed.index >= TEST_START] = 999999
    np.testing.assert_array_equal(observed_context(original, TEST_START), observed_context(changed, TEST_START))
    next_origin = TEST_START + pd.Timedelta(days=1)
    next_context = observed_context(changed, next_origin)
    np.testing.assert_array_equal(next_context[-24:, 0], np.full(24, 999999))


def test_initial_warmup_is_explicit_and_internal_missing_hours_are_errors():
    series = history()
    with pytest.raises(ValueError, match='48 observed'):
        prepare_sequences(series, START, VALIDATION_START)
    damaged = series.drop(START + pd.Timedelta(days=20, hours=3))
    with pytest.raises(ValueError, match='observed'):
        prepare_sequences(damaged, START, VALIDATION_START, drop_initial_warmup=True)
    with pytest.raises(ValueError, match='midnight'):
        observed_context(series, TEST_START + pd.Timedelta(hours=1))


def test_model_dimensions_parameter_count_and_forward():
    model = build_model()
    assert model.input_shape == (None, 48, 1)
    assert model.output_shape == (None, 24)
    assert model.count_params() == 5144
    assert model(tf.zeros((3,48,1)), training=False).shape == (3,24)


def test_january_only_scaling_and_validation_does_not_change_statistics(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler
    original_fit = StandardScaler.fit
    arrays = []
    def recording_fit(self, values, *args, **kwargs):
        arrays.append(np.asarray(values).copy())
        return original_fit(self, values, *args, **kwargs)
    monkeypatch.setattr(StandardScaler, 'fit', recording_fit)
    first = fit_lstm(selection_history(), tmp_path/'first.weights.h5', max_epochs=2)
    changed = selection_history()
    changed.loc[changed.index >= VALIDATION_START] *= 10
    second = fit_lstm(changed, tmp_path/'second.weights.h5', max_epochs=2)
    january = history().loc[history().index < VALIDATION_START].to_numpy().reshape(-1,1)
    assert len(arrays) == 2
    for values in arrays:
        np.testing.assert_array_equal(values, january)
    np.testing.assert_array_equal(first.forecaster.scaler.mean_, second.forecaster.scaler.mean_)
    assert first.forecaster.scaler.n_samples_seen_ == 744


def test_best_weight_restoration_and_prediction_reproducibility(tmp_path):
    first = fit_lstm(selection_history(), tmp_path/'first.weights.h5', max_epochs=3)
    second = fit_lstm(selection_history(), tmp_path/'second.weights.h5', max_epochs=3)
    pd.testing.assert_frame_equal(first.history, second.history)
    context = observed_context(history(), TEST_START)
    np.testing.assert_array_equal(first.forecaster.predict_context(context), second.forecaster.predict_context(context))
    restored = build_model()
    restored.load_weights(first.checkpoint_path)
    forecaster = LSTMForecaster(restored, first.forecaster.scaler)
    np.testing.assert_array_equal(first.forecaster.predict_context(context), forecaster.predict_context(context))
    assert first.best_epoch == first.history.loc[first.history.validation_mse_scaled.idxmin(),'epoch']
    assert first.validation_mse_scaled == pytest.approx(first.history.validation_mse_scaled.min(), rel=1e-6)


def test_no_test_data_during_fit_and_single_frozen_evaluation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='exclude'):
        fit_lstm(history(), tmp_path/'invalid.weights.h5')
    import citypulse_ai.forecasting.lstm as module
    original_fit = module.fit_lstm
    calls = []
    def recording_fit(series, *args, **kwargs):
        assert series.index.max() < TEST_START
        calls.append(series.copy())
        return original_fit(series, *args, **kwargs)
    monkeypatch.setattr(module, 'fit_lstm', recording_fit)
    result, predictions, metrics = run_lstm_experiment(history(), tmp_path/'run.weights.h5', max_epochs=2)
    assert len(calls) == 1
    assert result.training_sequences == 29 and result.validation_sequences == 14
    assert metrics.forecast_origins.tolist() == [14,15]
    assert metrics.target_hours.tolist() == [336,360]
    assert predictions[predictions.split.eq('test')].hour_start.min() == TEST_START
    assert predictions.hour_start.max() < END
    test = predictions[predictions.split.eq('test')]
    for key,value in error_metrics(test.actual,test.prediction).items():
        assert metrics.loc[metrics.split.eq('test'),key].iloc[0] == value


def test_24_outputs_without_forecast_day_truth_and_no_checkpoint_overwrite(tmp_path):
    result = fit_lstm(selection_history(), tmp_path/'best.weights.h5', max_epochs=2)
    original = history()
    issued = forecast_next_day(result.forecaster, original.loc[original.index < TEST_START], TEST_START)
    assert len(issued) == 24 and issued.prediction.ge(0).all()
    assert issued.hour_start.min() == TEST_START
    assert issued.hour_start.max() == TEST_START + pd.Timedelta(hours=23)
    before = result.checkpoint_path.read_bytes()
    with pytest.raises(FileExistsError):
        fit_lstm(selection_history(), result.checkpoint_path, max_epochs=2)
    assert before == result.checkpoint_path.read_bytes()


def test_restoration_uses_an_earlier_best_epoch_when_later_epochs_worsen(tmp_path, monkeypatch):
    """Script epoch states/losses to exercise callbacks without another real fit."""
    from types import SimpleNamespace
    def scripted_fit(model, dataset, *, validation_data, epochs, callbacks, verbose):
        model.stop_training = False
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()
        for epoch, validation_loss in enumerate([1.0, 0.5, 1.2]):
            for weight in model.trainable_variables:
                weight.assign(tf.ones_like(weight) * (epoch + 0.1))
            for callback in callbacks:
                callback.on_epoch_end(epoch, {'loss': 3.0-epoch, 'val_loss': validation_loss})
        for callback in callbacks:
            callback.on_train_end()
        return SimpleNamespace(history={'loss': [3.0,2.0,1.0], 'val_loss': [1.0,0.5,1.2]})
    monkeypatch.setattr(tf.keras.Model, 'fit', scripted_fit)
    result = fit_lstm(selection_history(), tmp_path/'scripted.weights.h5', max_epochs=3, patience=1)
    assert result.best_epoch == 2
    for weight in result.forecaster.model.trainable_variables:
        np.testing.assert_allclose(weight.numpy(), 1.1, atol=1e-6)
