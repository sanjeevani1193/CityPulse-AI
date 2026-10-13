"""Two frozen trusted models; no fits, source ingestion or artifact mutation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
xgboost = pytest.importorskip("xgboost")
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from citypulse_ai.api import app as api
from citypulse_ai.api.model import ARTIFACT_DIRECTORY as MLP_DIRECTORY
from citypulse_ai.api.xgboost import ARTIFACT_DIRECTORY, load_xgboost_model
from tests.test_inference_api import payload


@pytest.fixture(scope="module")
def client():
    with TestClient(api.create_app()) as session:
        assert all(row["ready"] for row in session.get("/models").json()["models"][:2])
        yield session


def test_model_listing_and_backward_compatible_default(client):
    listing = client.get("/models")
    assert listing.status_code == 200
    assert listing.json()["default_model"] == "pytorch-mlp"
    assert [r["id"] for r in listing.json()["models"]] == ["pytorch-mlp", "xgboost", "tensorflow-lstm"]
    assert all(r["ready"] and r["model_version"] for r in listing.json()["models"][:2])
    default = client.post("/predict", json=payload())
    explicit = client.post("/predict", json={**payload(), "model": "pytorch-mlp"})
    assert default.status_code == explicit.status_code == 200
    assert default.json() == explicit.json()
    assert default.json()["model_name"] == "citywide-pytorch-mlp"


@pytest.mark.parametrize("model", ["pytorch-mlp", "xgboost"])
def test_real_saved_example_predictions(client, model):
    history = json.loads((ARTIFACT_DIRECTORY / "recovery_history.json").read_text())["history"]
    origin = pd.Timestamp("2024-03-01T00:00:00-05:00")
    last_week = [r for r in history if origin - pd.Timedelta(hours=168) <= pd.Timestamp(r["timestamp"]) < origin]
    response = client.post("/predict", json={"origin": origin.isoformat(), "history": last_week, "model": model})
    assert response.status_code == 200
    if model == "xgboost":
        record = json.loads((ARTIFACT_DIRECTORY.parent / "experiment.json").read_text())
        expected = record["example_next_day_forecast"]
    else:
        record = json.loads((MLP_DIRECTORY / "experiment.json").read_text())
        expected = record["example_forecast"]
    np.testing.assert_array_equal(response.json()["predicted_ride_counts"], [r["prediction"] for r in expected])


@pytest.mark.parametrize("model", ["pytorch-mlp", "xgboost"])
@pytest.mark.parametrize("origin", ["2024-03-10T00:00:00-05:00", "2024-11-03T00:00:00-04:00"])
def test_shared_contract_dst_and_repeated_requests(client, model, origin):
    request = {**payload(origin), "model": model}
    first = client.post("/predict", json=request);second = client.post("/predict", json=request)
    assert first.status_code == second.status_code == 200 and first.json() == second.json()
    body = first.json();hours = pd.DatetimeIndex(body["target_timestamps_utc"])
    assert len(body["predicted_ride_counts"]) == len(hours) == hours.nunique() == 24
    assert (hours[1:] - hours[:-1] == pd.Timedelta(hours=1)).all()
    assert set(body["utc_offsets"]) == {"-04:00", "-05:00"}
    assert body["model_name"] == ("xgboost" if model == "xgboost" else "citywide-pytorch-mlp")


@pytest.mark.parametrize("model", ["pytorch-mlp", "xgboost"])
@pytest.mark.parametrize("invalid", ["future", "duplicate", "missing", "non_midnight"])
def test_identical_issuance_validation(client, model, invalid):
    request = {**payload(hours=169), "model": model}
    if invalid == "future":request["history"][-1]["timestamp"] = request["origin"]
    elif invalid == "duplicate":request["history"][1] = request["history"][0]
    elif invalid == "missing":del request["history"][30]
    else:request["origin"] = "2024-03-01T01:00:00-05:00"
    assert client.post("/predict", json=request).status_code == 422


@pytest.mark.parametrize("name", ["unknown", "tensorflow", "tft", "", "/tmp/model.ubj", None])
def test_unknown_models_rejected(client, name):
    assert client.post("/predict", json={**payload(), "model": name}).status_code == 422


def test_xgboost_raw_features_no_scaling_or_fit(client, monkeypatch):
    def forbidden(*args, **kwargs):pytest.fail("Preprocessing fit/transform or model training during XGBoost inference")
    monkeypatch.setattr(StandardScaler, "transform", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(xgboost.XGBRegressor, "fit", forbidden)
    model = client.app.state.models["xgboost"]
    before = bytes(model.regressor.get_booster().save_raw())
    response = client.post("/predict", json={**payload(), "model": "xgboost"})
    assert response.status_code == 200
    assert before == bytes(model.regressor.get_booster().save_raw())


def test_both_loaders_once_and_trusted_environment_paths(monkeypatch):
    calls = {"mlp": 0, "xgb": 0}
    mlp_loader, xgb_loader = api.load_serving_model, api.load_xgboost_model
    def mlp():calls["mlp"] += 1;return mlp_loader()
    def xgb():calls["xgb"] += 1;return xgb_loader()
    monkeypatch.setattr(api, "load_serving_model", mlp)
    monkeypatch.setattr(api, "load_xgboost_model", xgb)
    monkeypatch.setenv("CITYPULSE_MLP_ARTIFACT_DIR", str(MLP_DIRECTORY))
    monkeypatch.setenv("CITYPULSE_XGBOOST_ARTIFACT_DIR", str(ARTIFACT_DIRECTORY))
    monkeypatch.setattr(xgboost.XGBRegressor, "fit", lambda *a, **k: pytest.fail("Startup trained XGBoost"))
    with TestClient(api.create_app()) as session:
        for name in ["pytorch-mlp", "xgboost", "xgboost"]:
            assert session.post("/predict", json={**payload(), "model": name}).status_code == 200
    assert calls == {"mlp": 1, "xgb": 1}


@pytest.mark.parametrize("failure", ["missing", "corrupt_model", "corrupt_manifest", "native_runtime", "version"])
def test_xgboost_failure_isolated_from_mlp(tmp_path, monkeypatch, failure):
    for name in ["model.ubj", "model_manifest.json"]:
        if failure == "missing":continue
        contents = (ARTIFACT_DIRECTORY / name).read_bytes()
        if failure == "corrupt_model" and name.endswith("ubj"):contents = b"bad"
        if failure == "corrupt_manifest" and name.endswith("json"):contents = b"bad"
        (tmp_path / name).write_bytes(contents)
    if failure == "native_runtime":
        monkeypatch.setattr(api, "load_xgboost_model", lambda: (_ for _ in ()).throw(ImportError("OpenMP unavailable")))
    else:
        monkeypatch.setenv("CITYPULSE_XGBOOST_ARTIFACT_DIR", str(tmp_path))
        if failure == "version":monkeypatch.setattr(xgboost, "__version__", "wrong")
    with TestClient(api.create_app()) as session:
        listing = {r["id"]: r for r in session.get("/models").json()["models"]}
        assert listing["pytorch-mlp"]["ready"] is True and listing["xgboost"]["ready"] is False
        assert session.get("/health").status_code == 200
        assert session.post("/predict", json=payload()).status_code == 200
        assert session.post("/predict", json={**payload(), "model": "xgboost"}).status_code == 503


def test_loader_checks_booster_feature_order(monkeypatch):
    actual = xgboost.XGBRegressor
    class WrongFeatures(actual):
        def load_model(self, path):
            super().load_model(path)
            self.get_booster().feature_names = list(reversed(self.get_booster().feature_names))
    monkeypatch.setattr(xgboost, "XGBRegressor", WrongFeatures)
    with pytest.raises(ValueError, match="unexpected features"):
        load_xgboost_model()
