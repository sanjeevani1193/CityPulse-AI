"""Small CPU MLP experiment using the existing causal midnight features."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from citypulse_ai.forecasting.baselines import LOCAL_ZONE
from citypulse_ai.forecasting.experiment import END
from citypulse_ai.forecasting.xgboost_model import (
    FEATURES, VALIDATION_START, TEST_START, forecast_features,
    training_rows, evaluate_model,
)


class DemandMLP(nn.Module):
    """One feature row predicts one hourly count; batch output is (N, 1)."""

    def __init__(self, input_features=len(FEATURES)):
        super().__init__()
        if input_features <= 0:
            raise ValueError("Input feature count must be positive")
        self.layers = nn.Sequential(
            nn.Linear(input_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.layers[0].in_features:
            raise ValueError("Inputs must have shape (batch, input_features)")
        return self.layers(inputs)


@dataclass
class MLPForecaster:
    """Adapter for the existing next-day predictor and metric evaluator."""

    network: DemandMLP
    feature_scaler: StandardScaler
    target_scaler: StandardScaler

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != FEATURES or not np.isfinite(features.to_numpy()).all():
            raise ValueError("Prediction requires ordered, finite forecast features")
        inputs = torch.as_tensor(self.feature_scaler.transform(features), dtype=torch.float32, device="cpu")
        self.network.eval()
        with torch.no_grad():
            scaled = self.network(inputs).numpy()
        return self.target_scaler.inverse_transform(scaled).ravel().astype(float)


@dataclass
class TrainingResult:
    forecaster: MLPForecaster
    history: pd.DataFrame
    best_epoch: int
    training_rows: int
    dropped_training_rows: int
    duration_seconds: float
    settings: dict
    checkpoint_path: Path


def validation_rows(history: pd.Series) -> pd.DataFrame:
    """Build validation targets separately from features available at issuance."""
    blocks = []
    for origin in pd.date_range(VALIDATION_START, TEST_START, freq="D", inclusive="left"):
        block = forecast_features(history, origin)
        block["target"] = history.reindex(pd.DatetimeIndex(block.hour_start)).to_numpy(dtype=float, na_value=np.nan)
        blocks.append(block)
    rows = pd.concat(blocks, ignore_index=True)
    if not np.isfinite(rows.target).all() or not rows.target.ge(0).all():
        raise ValueError("Validation requires all finite nonnegative targets")
    return rows


def fit_mlp(validation_history: pd.Series, checkpoint_path, *, max_epochs=200,
            patience=20, learning_rate=0.001, batch_size=64, seed=42,
            min_delta=0.00001) -> TrainingResult:
    """Fit January only; select an epoch using validation scaled MSE, never test.

    Loss curves compare full-dataset, post-epoch evaluation MSE in standardized
    target units, before the evaluator's predeclared nonnegative clipping.
    Save only the selected state and January scaler statistics; reject overwrite.
    """
    if validation_history.empty or not (validation_history.index < TEST_START).all():
        raise ValueError("Training/selection history must exclude all test observations")
    if any(not isinstance(value, int) or value <= 0 for value in (max_epochs, patience, batch_size)):
        raise ValueError("Epochs, patience and batch size must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("Use a positive learning rate and nonnegative minimum improvement")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

    train, dropped = training_rows(validation_history)
    validation = validation_rows(validation_history)
    feature_scaler = StandardScaler().fit(train[FEATURES])
    target_scaler = StandardScaler().fit(train[["target"]].to_numpy())
    x_train = torch.tensor(feature_scaler.transform(train[FEATURES]), dtype=torch.float32)
    y_train = torch.tensor(target_scaler.transform(train[["target"]].to_numpy()), dtype=torch.float32)
    x_validation = torch.tensor(feature_scaler.transform(validation[FEATURES]), dtype=torch.float32)
    y_validation = torch.tensor(target_scaler.transform(validation[["target"]].to_numpy()), dtype=torch.float32)

    # CPU, one thread, deterministic kernels and a seeded shuffle generator.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = DemandMLP().to("cpu")
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    records = []
    best_loss, patience_reference = float("inf"), float("inf")
    best_state, best_epoch, stalled = None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; no checkpoint saved")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_function(model(x_train), y_train).item()
            validation_loss = loss_function(model(x_validation), y_validation).item()
        if not np.isfinite([train_loss, validation_loss]).all():
            raise ValueError("Nonfinite evaluation loss; no checkpoint saved")
        records.append({"epoch": epoch, "train_mse_scaled": train_loss,
                        "validation_mse_scaled": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, deepcopy(model.state_dict())
        if validation_loss < patience_reference - min_delta:
            patience_reference, stalled = validation_loss, 0
        else:
            stalled += 1
        if stalled >= patience:
            break
    duration = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    settings = {"max_epochs": max_epochs, "patience": patience, "learning_rate": learning_rate,
                "batch_size": batch_size, "seed": seed, "min_delta": min_delta,
                "device": "cpu", "threads": 1, "selection_metric": "validation_mse_scaled"}
    checkpoint = {
        "state_dict": best_state, "features": FEATURES, "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss, "settings": settings,
        "feature_mean": torch.tensor(feature_scaler.mean_),
        "feature_scale": torch.tensor(feature_scaler.scale_),
        "target_mean": torch.tensor(target_scaler.mean_),
        "target_scale": torch.tensor(target_scaler.scale_),
        "training_rows": len(train), "dropped_training_rows": dropped,
        "training_first_utc": str(train.hour_start.min()),
        "training_last_utc": str(train.hour_start.max()),
        "timezone": LOCAL_ZONE, "horizon_elapsed_hours": 24,
        "torch_version": str(torch.__version__),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing files, including concurrent runs.
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return TrainingResult(MLPForecaster(model, feature_scaler, target_scaler),
                          pd.DataFrame(records), best_epoch, len(train), dropped,
                          duration, settings, checkpoint_path)


def load_mlp(checkpoint_path) -> MLPForecaster:
    """Restore CPU weights and train-fitted scaling, using weights-only loading."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES or checkpoint["horizon_elapsed_hours"] != 24:
        raise ValueError("Checkpoint forecast features/contract do not match")
    model = DemandMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_scaler, target_scaler = StandardScaler(), StandardScaler()
    for scaler, prefix in ((feature_scaler, "feature"), (target_scaler, "target")):
        scaler.mean_ = checkpoint[f"{prefix}_mean"].numpy()
        scaler.scale_ = checkpoint[f"{prefix}_scale"].numpy()
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.n_samples_seen_ = checkpoint["training_rows"]
    feature_scaler.feature_names_in_ = np.array(FEATURES, dtype=object)
    return MLPForecaster(model, feature_scaler, target_scaler)


def run_mlp_experiment(series: pd.Series, checkpoint_path, **training_options):
    """Fit without test data, then evaluate the frozen best epoch on both splits."""
    result = fit_mlp(series.loc[series.index < TEST_START].copy(), checkpoint_path, **training_options)
    blocks, metrics = [], []
    for split, left, right in (("validation", VALIDATION_START, TEST_START), ("test", TEST_START, END)):
        predictions, scores = evaluate_model(result.forecaster, series, left, right)
        blocks.append(predictions.assign(split=split))
        metrics.append({"split": split, "model": "PyTorch MLP", **scores})
    return result, pd.concat(blocks, ignore_index=True), pd.DataFrame(metrics)
