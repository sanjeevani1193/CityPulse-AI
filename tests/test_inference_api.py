"""Frozen real checkpoint + small requests; no training, ingestion or cloud use."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient
from citypulse_ai.api import app as api
from citypulse_ai.api.model import ARTIFACT_DIRECTORY, load_serving_model
from citypulse_ai.api.schemas import PredictionRequest
from citypulse_ai.forecasting.mlp import load_mlp
from citypulse_ai.forecasting.xgboost_model import FEATURES, forecast_features


def payload(origin="2024-03-01T00:00:00-05:00", hours=168):
    start = pd.Timestamp(origin).tz_convert("UTC") - pd.Timedelta(hours=hours)
    return {"origin": origin, "history": [{"timestamp": t.isoformat(), "ride_count": 100 + n % 24}
        for n, t in enumerate(pd.date_range(start, periods=hours, freq="h"))]}


@pytest.fixture(scope="module")
def client():
    with TestClient(api.create_app()) as session:
        assert session.get("/health").status_code == 200
        yield session


def test_health_frozen_model_and_repeatability(client, monkeypatch):
    # Any fit or training call during inference is a regression.
    monkeypatch.setattr(StandardScaler, "fit", lambda *a, **k: pytest.fail("Scaler fitted during inference"))
    model = client.app.state.model.forecaster.network
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    calls = []
    hook = model.register_forward_pre_hook(lambda *a: calls.append(torch.is_inference_mode_enabled()))
    try:
        first = client.post("/predict", json=payload())
        second = client.post("/predict", json=payload())
    finally:
        hook.remove()
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert calls == [True, True] and not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.device.type == "cpu" for p in model.parameters())
    response = first.json()
    for key in ["target_timestamps_utc", "target_timestamps_local", "utc_offsets", "predicted_ride_counts"]:
        assert len(response[key]) == 24
    assert min(response["predicted_ride_counts"]) >= 0
    assert response["timezone"] == "America/New_York"
    assert client.get("/health").json()["model_ready"] is True


def test_single_startup_load_and_offset_aware_history(monkeypatch):
    calls = []
    original = api.load_serving_model
    def loader():
        calls.append(True)
        return original()
    monkeypatch.setattr(api, "load_serving_model", loader)
    with TestClient(api.create_app()) as session:
        request = payload("2024-11-10T00:00:00-05:00")
        for row in request["history"]:
            row["timestamp"] = pd.Timestamp(row["timestamp"]).tz_convert("America/New_York").isoformat()
        assert session.post("/predict", json=request).status_code == 200
        request["origin"] = pd.Timestamp(request["origin"]).tz_convert("UTC").isoformat()
        assert session.post("/predict", json=request).status_code == 200
        assert session.get("/health").status_code == 200
    assert calls == [True]


def test_exact_feature_scaler_and_adapter_parity(client):
    request = PredictionRequest(**payload())
    frame = forecast_features(request.series(), request.origin)
    model = client.app.state.model.forecaster
    checkpoint = torch.load(ARTIFACT_DIRECTORY / "best_model.pt", weights_only=True, map_location="cpu")
    assert FEATURES == checkpoint["features"]
    assert FEATURES == ["hour", "weekday", "is_weekend", "hour_sin", "hour_cos",
                        "weekday_sin", "weekday_cos", "lag_24", "lag_168"]
    np.testing.assert_array_equal(model.feature_scaler.mean_, checkpoint["feature_mean"].numpy())
    np.testing.assert_array_equal(model.feature_scaler.scale_, checkpoint["feature_scale"].numpy())
    for lag in [24, 168]:
        np.testing.assert_array_equal(frame[f"lag_{lag}"], request.series().reindex(
            pd.DatetimeIndex(frame.hour_start) - pd.Timedelta(hours=lag)))
        assert frame[f"lag_{lag}_source_hour"].lt(pd.Timestamp(request.origin)).all()
    expected = np.maximum(load_mlp(ARTIFACT_DIRECTORY / "best_model.pt").predict(frame[FEATURES]), 0)
    np.testing.assert_array_equal(client.post("/predict", json=payload()).json()["predicted_ride_counts"], expected)


@pytest.mark.parametrize("origin,offsets,final", [
    ("2024-03-10T00:00:00-05:00", {"-05:00", "-04:00"}, "2024-03-11T00:00:00-04:00"),
    ("2024-11-03T00:00:00-04:00", {"-04:00", "-05:00"}, "2024-11-03T22:00:00-05:00"),
])
def test_dst_elapsed_hours_and_calendar_features(client, origin, offsets, final):
    response = client.post("/predict", json=payload(origin)).json()
    targets = pd.DatetimeIndex(response["target_timestamps_utc"])
    assert len(targets) == targets.nunique() == 24
    assert (targets[1:] - targets[:-1] == pd.Timedelta(hours=1)).all()
    assert response["target_timestamps_local"][-1] == final
    assert set(response["utc_offsets"]) == offsets
    frame = forecast_features(PredictionRequest(**payload(origin)).series(), origin)
    local = targets.tz_convert("America/New_York")
    np.testing.assert_array_equal(frame.hour, local.hour)
    np.testing.assert_array_equal(frame.weekday, local.dayofweek)
    if "03-10" in origin:assert 2 not in local.hour
    else:assert list(local.hour).count(1) == 2


@pytest.mark.parametrize("case", ["duplicate", "same_instant", "order", "missing", "future", "short",
    "negative", "fractional", "boolean", "null", "naive", "bad_timestamp", "minute",
    "not_midnight", "wrong_dst_offset", "missing_last", "extra_path", "missing_count", "distant_gap"])
def test_invalid_inputs_fail_closed(client, case):
    request = payload(hours=169)
    history = request["history"]
    if case == "duplicate":history[1] = deepcopy(history[0])
    elif case == "same_instant":history[1]["timestamp"] = pd.Timestamp(history[0]["timestamp"]).tz_convert("America/New_York").isoformat()
    elif case == "order":history[0], history[1] = history[1], history[0]
    elif case == "missing":del history[30]
    elif case == "future":history[-1]["timestamp"] = pd.Timestamp(request["origin"]).tz_convert("UTC").isoformat()
    elif case == "short":request["history"] = history[-167:]
    elif case in ["negative", "fractional", "boolean", "null"]:
        history[0]["ride_count"] = {"negative": -1, "fractional": 1.5, "boolean": True, "null": None}[case]
    elif case == "naive":request["origin"] = "2024-03-01T00:00:00"
    elif case == "bad_timestamp":history[0]["timestamp"] = "not a date"
    elif case == "minute":history[0]["timestamp"] = "2024-02-22T04:01:00Z"
    elif case == "not_midnight":request["origin"] = "2024-03-01T01:00:00-05:00"
    elif case == "wrong_dst_offset":request["origin"] = "2024-03-10T00:00:00-04:00"
    elif case == "missing_last":del history[-1]
    elif case == "extra_path":request["model_path"] = "/untrusted/model.pt"
    elif case == "missing_count":del history[0]["ride_count"]
    elif case == "distant_gap":history[0]["timestamp"] = "1900-01-01T00:00:00Z"
    assert client.post("/predict", json=request).status_code == 422


@pytest.mark.parametrize("case", ["missing", "corrupt", "record_corrupt"])
def test_bad_artifacts_readiness_and_no_fallback(tmp_path, monkeypatch, case):
    # Copy only tiny artifacts; never modify the original experiment.
    for name in ["best_model.pt", "experiment.json"]:
        if case == "missing" and name == "best_model.pt":continue
        contents = (ARTIFACT_DIRECTORY / name).read_bytes()
        if (case == "corrupt" and name.endswith(".pt")) or (case == "record_corrupt" and name.endswith(".json")):
            contents = b"corrupt"
        (tmp_path / name).write_bytes(contents)
    monkeypatch.setattr(api, "load_serving_model", lambda: load_serving_model(tmp_path))
    with TestClient(api.create_app()) as session:
        assert session.get("/health").status_code == 503
        assert session.get("/health").json()["model_ready"] is False
        assert session.post("/predict", json=payload()).status_code == 503


def test_saved_experiment_example_parity(client):
    # Read the surviving small hourly artifacts; no archive ETL or training.
    from citypulse_ai.forecasting.experiment import load_development_citywide
    root = Path(__file__).resolve().parents[1]
    frame, _ = load_development_citywide(root / "outputs/historical")
    origin = pd.Timestamp("2024-03-01T00:00:00-05:00")
    history = frame.loc[frame.hour_start.ge(origin - pd.Timedelta(hours=168)) & frame.hour_start.lt(origin)]
    request = {"origin": origin.isoformat(), "history": [
        {"timestamp": row.hour_start.isoformat(), "ride_count": int(row.ride_count)}
        for row in history.itertuples()]}
    record = json.loads((ARTIFACT_DIRECTORY / "experiment.json").read_text())
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    np.testing.assert_array_equal(response.json()["predicted_ride_counts"],
                                  [row["prediction"] for row in record["example_forecast"]])
