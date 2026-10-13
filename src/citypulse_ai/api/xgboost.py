"""Inference-only adapter for the verified January XGBoost reproduction."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from citypulse_ai.forecasting.xgboost_model import FEATURES

ARTIFACT_DIRECTORY = (Path(__file__).resolve().parents[3] / "outputs" /
    "phase3-xgboost-jan-feb-2024" / "recovered-83a9eaa90d964c889a03c7b6ec7de09d")
MODEL_SHA256 = "98687a1aefb4d669eed40547720e1e5f4a5502dce7987f4007756b23d9edb57f"
MANIFEST_SHA256 = "102e0240783a337e651202874e9e93ba11978b5f7d9bb88d8ae4b4d36a97eef9"
MODEL_NAME = "xgboost"
MODEL_VERSION = "january-2024-selected-150-trees:98687a1aefb4"


@dataclass(frozen=True)
class XGBoostServingModel:
    regressor: object
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION

    def predict(self, features):
        raw = features[FEATURES]
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError("Prediction requires finite raw features")
        predictions = np.maximum(np.asarray(self.regressor.predict(raw), dtype=float), 0)
        if predictions.shape != (24,) or not np.isfinite(predictions).all():
            raise ValueError("Model did not produce 24 finite predictions")
        return predictions


def load_xgboost_model(directory: Path | None = None) -> XGBoostServingModel:
    directory = Path(directory if directory is not None else
                     os.environ.get("CITYPULSE_XGBOOST_ARTIFACT_DIR", ARTIFACT_DIRECTORY))
    for name, digest in (("model.ubj", MODEL_SHA256), ("model_manifest.json", MANIFEST_SHA256)):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required trusted XGBoost artifact unavailable: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Trusted XGBoost artifact checksum mismatch: {name}")
    manifest = json.loads((directory / "model_manifest.json").read_text())
    if (manifest["status"] != "verified_ready_for_inference"
            or manifest["feature_order"] != FEATURES or manifest["model_version"] != MODEL_VERSION
            or manifest["artifact_filename"] != "model.ubj"
            or manifest["artifact_sha256"] != MODEL_SHA256
            or manifest["preprocessing"] != "raw engineered features; no fitted scalers"):
        raise ValueError("XGBoost manifest does not match the trusted forecast contract")
    # Import native XGBoost only when loading this optional model. Missing OpenMP
    # or XGBoost must not prevent the MLP application from starting.
    import xgboost
    if xgboost.__version__ != manifest["versions"]["xgboost"]:
        raise ValueError("XGBoost version differs from the verified reproduction")
    model = xgboost.XGBRegressor(**manifest["hyperparameters"])
    model.load_model(directory / "model.ubj")
    if (model.get_booster().feature_names != FEATURES or model.n_features_in_ != 9
            or model.get_booster().num_boosted_rounds() != 150):
        raise ValueError("Loaded XGBoost model has unexpected features or tree count")
    return XGBoostServingModel(model)
