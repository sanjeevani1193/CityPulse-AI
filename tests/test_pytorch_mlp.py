"""Shape, learning, train-only scaling and midnight forecast contract checks."""

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
from torch import nn
from citypulse_ai.forecasting.baselines import LOCAL_ZONE, error_metrics
from citypulse_ai.forecasting.experiment import START, END
from citypulse_ai.forecasting.xgboost_model import (
    FEATURES, VALIDATION_START, TEST_START, training_rows,
    forecast_features, forecast_next_day,
)
from citypulse_ai.forecasting.mlp import DemandMLP, fit_mlp, load_mlp, run_mlp_experiment


def history():
    hours = pd.date_range(START, END, freq="h", inclusive="left").tz_convert("UTC")
    local = hours.tz_convert(LOCAL_ZONE)
    return pd.Series(100 + local.hour * 30 + local.dayofweek * 10, index=hours, dtype=float)


def selection_history():
    series = history()
    return series.loc[series.index < TEST_START]


@pytest.mark.parametrize("batch", [1, 24, 64])
def test_tensor_shape_forward_and_architecture(batch):
    model = DemandMLP()
    output = model(torch.zeros(batch, len(FEATURES), dtype=torch.float32))
    assert output.shape == (batch, 1) and output.dtype == torch.float32
    layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(9, 32), (32, 16), (16, 1)]
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(batch, 8))


def test_backward_gradients_and_adam_update():
    torch.manual_seed(42)
    model = DemandMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs, targets = torch.ones(4, 9), torch.full((4, 1), 5.0)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(inputs), targets)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())


def test_scalers_fit_january_only_and_warmup_is_identical(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler
    fitted = []
    original = StandardScaler.fit
    def record_fit(self, values, *args, **kwargs):
        fitted.append(np.asarray(values).copy())
        return original(self, values, *args, **kwargs)
    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    result = fit_mlp(selection_history(), tmp_path / 'best.pt', max_epochs=3)
    train, dropped = training_rows(history())
    assert len(fitted) == 2
    np.testing.assert_array_equal(fitted[0], train[FEATURES])
    np.testing.assert_array_equal(fitted[1], train[['target']])
    assert result.training_rows == 576 and result.dropped_training_rows == dropped == 168
    np.testing.assert_allclose(result.forecaster.feature_scaler.mean_, train[FEATURES].mean())
    assert result.forecaster.target_scaler.mean_[0] == pytest.approx(train.target.mean())


def test_validation_distribution_does_not_change_scaling(tmp_path):
    series = selection_history()
    first = fit_mlp(series, tmp_path / 'first.pt', max_epochs=2)
    changed = series.copy()
    changed.loc[changed.index >= VALIDATION_START] *= 20
    second = fit_mlp(changed, tmp_path / 'second.pt', max_epochs=2)
    np.testing.assert_array_equal(first.forecaster.feature_scaler.mean_, second.forecaster.feature_scaler.mean_)
    np.testing.assert_array_equal(first.forecaster.target_scaler.mean_, second.forecaster.target_scaler.mean_)


def test_cpu_training_reproducibility_and_checkpoint_restore(tmp_path):
    first = fit_mlp(selection_history(), tmp_path / 'a.pt', max_epochs=6)
    second = fit_mlp(selection_history(), tmp_path / 'b.pt', max_epochs=6)
    pd.testing.assert_frame_equal(first.history, second.history)
    for key, value in first.forecaster.network.state_dict().items():
        assert torch.equal(value, second.forecaster.network.state_dict()[key])
    restored = load_mlp(first.checkpoint_path)
    features = forecast_features(history(), TEST_START)[FEATURES]
    np.testing.assert_array_equal(first.forecaster.predict(features), restored.predict(features))
    assert not restored.network.training
    assert all(parameter.device.type == 'cpu' for parameter in restored.network.parameters())
    assert first.best_epoch == first.history.loc[first.history.validation_mse_scaled.idxmin(), 'epoch']


def test_early_stopping_is_bounded_and_preserves_lowest_loss(tmp_path):
    result = fit_mlp(selection_history(), tmp_path / 'best.pt', max_epochs=20,
                     patience=2, learning_rate=1e-12, min_delta=1e-4)
    assert len(result.history) == 3
    saved = torch.load(result.checkpoint_path, weights_only=True)
    assert saved['best_validation_mse_scaled'] == result.history.validation_mse_scaled.min()
    assert saved['best_epoch'] == result.best_epoch


def test_no_fit_or_selection_on_test_data(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exclude"):
        fit_mlp(history(), tmp_path / 'invalid.pt')
    assert not (tmp_path / 'invalid.pt').exists()
    import citypulse_ai.forecasting.mlp as module
    original = module.fit_mlp
    calls = []
    def record_fit(series, *args, **kwargs):
        assert series.index.max() < TEST_START
        calls.append(series.copy())
        return original(series, *args, **kwargs)
    monkeypatch.setattr(module, 'fit_mlp', record_fit)
    result, predictions, metrics = run_mlp_experiment(history(), tmp_path / 'run.pt', max_epochs=2)
    assert len(calls) == 1
    assert metrics.target_hours.tolist() == [336, 360]
    assert metrics.forecast_origins.tolist() == [14, 15]
    assert predictions[predictions.split.eq('test')].hour_start.min() == TEST_START
    assert predictions.hour_start.max() < END


def test_forecast_has_24_hours_without_future_actuals_and_allows_completed_history(tmp_path):
    result = fit_mlp(selection_history(), tmp_path / 'best.pt', max_epochs=2)
    series = history()
    issued = forecast_next_day(result.forecaster, series.loc[series.index < TEST_START], TEST_START)
    assert len(issued) == 24 and issued.hour_start.min() == TEST_START
    assert issued.lag_24_source_hour.lt(issued.origin).all()
    assert issued.lag_168_source_hour.lt(issued.origin).all()
    changed = series.copy()
    changed.loc[changed.index >= TEST_START] *= 10
    pd.testing.assert_frame_equal(issued, forecast_next_day(result.forecaster, changed, TEST_START))
    next_origin = TEST_START + pd.Timedelta(days=1)
    next_features = forecast_features(changed, next_origin)
    np.testing.assert_array_equal(next_features.lag_24, changed.reindex(pd.DatetimeIndex(issued.hour_start)))
    leap = pd.Timestamp('2024-02-29', tz=LOCAL_ZONE)
    assert len(forecast_next_day(result.forecaster, series.loc[series.index < leap], leap)) == 24


def test_missing_lags_rejected_and_existing_checkpoint_preserved(tmp_path):
    result = fit_mlp(selection_history(), tmp_path / 'best.pt', max_epochs=2)
    saved = result.checkpoint_path.read_bytes()
    with pytest.raises(FileExistsError):
        fit_mlp(selection_history(), result.checkpoint_path, max_epochs=2)
    assert result.checkpoint_path.read_bytes() == saved
    series = history().drop(TEST_START - pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="missing"):
        forecast_next_day(result.forecaster, series, TEST_START)


def test_metric_calculations_agree_with_tensor_mse():
    actual, predicted = [2, 4], [1, 6]
    scores = error_metrics(actual, predicted)
    mse = nn.MSELoss()(torch.tensor(predicted, dtype=torch.float32), torch.tensor(actual, dtype=torch.float32)).item()
    assert mse == 2.5
    assert scores['MAE'] == 1.5
    assert scores['RMSE'] == pytest.approx(np.sqrt(mse))
    assert scores['WAPE_percent'] == 50
