"""Restore only the reviewed experiment checkpoint and its fitted preprocessing."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from citypulse_ai.forecasting.mlp import MLPForecaster, load_mlp
from citypulse_ai.forecasting.xgboost_model import FEATURES

RUN = "run-20261012T125502822756Z"
CHECKPOINT_SHA256 = "af2fe71d011d8313d73b8b74e9d692e1edcda6467373074bfafd8789a9eaaff8"
RECORD_SHA256 = "5c1bbf0777097c9bb15dc3ced734bfcf441baebd4df4b3cb78a3bb11d9e9270b"
MODEL_NAME = "citywide-pytorch-mlp"
MODEL_VERSION = f"{RUN}:{CHECKPOINT_SHA256[:12]}"
ARTIFACT_DIRECTORY = (Path(__file__).resolve().parents[3] / "outputs" /
                      "phase3-pytorch-mlp-jan-feb-2024" / RUN)


@dataclass(frozen=True)
class ServingModel:
    forecaster: MLPForecaster
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION

    def predict(self, features):
        # Reuse the experiment's scaler/inverse transform and physical clipping.
        # The outer inference context also covers its existing no_grad adapter.
        with torch.inference_mode():
            predictions = np.maximum(self.forecaster.predict(features[FEATURES]), 0)
        if predictions.shape != (24,) or not np.isfinite(predictions).all():
            raise ValueError("Model did not produce 24 finite predictions")
        return predictions


def load_serving_model(directory: Path | None = None) -> ServingModel:
    """Directory is developer configuration, never an HTTP request parameter.

    Fingerprints pin the trusted existing experiment; changed or missing files
    fail closed. There is no random-weight fallback or preprocessing fit.
    """
    directory = Path(directory if directory is not None else
                     os.environ.get("CITYPULSE_MLP_ARTIFACT_DIR", ARTIFACT_DIRECTORY))
    for name, expected in (("best_model.pt", CHECKPOINT_SHA256), ("experiment.json", RECORD_SHA256)):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required trusted artifact unavailable: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Trusted artifact checksum mismatch: {name}")
    record = json.loads((directory / "experiment.json").read_text())
    torch.set_num_threads(1)
    model = load_mlp(directory / "best_model.pt")  # CPU, strict state_dict, weights_only=True
    for scaler, size in ((model.feature_scaler, 9), (model.target_scaler, 1)):
        if (scaler.mean_.shape != (size,) or scaler.scale_.shape != (size,)
                or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all()
                or not (scaler.scale_ > 0).all()):
            raise ValueError("Invalid fitted scaler statistics")
    if (not np.array_equal(model.target_scaler.mean_, record["target_mean"])
            or not np.array_equal(model.target_scaler.scale_, record["target_scale"])
            or any(not torch.isfinite(weight).all() for weight in model.network.parameters())):
        raise ValueError("Checkpoint parameters differ from trusted experiment contract")
    model.network.eval()
    return ServingModel(model)
