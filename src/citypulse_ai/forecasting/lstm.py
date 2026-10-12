"""Univariate, midnight-issued 48-to-24 LSTM development experiment."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from citypulse_ai.forecasting.baselines import LOCAL_ZONE, error_metrics, utc_boundary
from citypulse_ai.forecasting.experiment import START, END
from citypulse_ai.forecasting.xgboost_model import VALIDATION_START, TEST_START

LOOKBACK = 48
HORIZON = 24


def configure_cpu(seed=42):
    """Call before TensorFlow initializes devices; retain an explicit CPU scope."""
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def observed_context(series: pd.Series, origin) -> np.ndarray:
    """Return only the 48 completed hours before a NYC-local midnight."""
    origin = utc_boundary(origin)
    if origin.tz_convert(LOCAL_ZONE) != origin.tz_convert(LOCAL_ZONE).normalize():
        raise ValueError("Forecast issuance must be NYC local midnight")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None or series.index.has_duplicates:
        raise ValueError("History needs a unique timezone-aware hourly index")
    if series.index.hasnans or not series.index.equals(series.index.floor('h')):
        raise ValueError("History timestamps must be nonmissing hour boundaries")
    hours = pd.date_range(origin - pd.Timedelta(hours=LOOKBACK), periods=LOOKBACK, freq='h')
    values = series.reindex(hours).to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("All previous 48 observed hours must be finite and nonnegative")
    return values.reshape(LOOKBACK, 1)


@dataclass
class Sequences:
    origins: pd.DatetimeIndex
    inputs: np.ndarray
    targets: np.ndarray
    dropped_warmup: int


def prepare_sequences(series: pd.Series, start, end, *, drop_initial_warmup=False) -> Sequences:
    """One sample per local midnight; require all targets within the split.

    Only leading origins without enough pre-window history may be skipped.
    Internal missing hours are errors rather than silently dropped sequences.
    """
    left, right = utc_boundary(start), utc_boundary(end)
    local_left, local_right = left.tz_convert(LOCAL_ZONE), right.tz_convert(LOCAL_ZONE)
    if left >= right or local_left != local_left.normalize() or local_right != local_right.normalize():
        raise ValueError("Use increasing NYC local-midnight split boundaries")
    inputs, targets, origins, dropped = [], [], [], 0
    if series.empty:
        raise ValueError("Demand history cannot be empty")
    for origin in pd.date_range(local_left, local_right, freq='D', inclusive='left'):
        origin = origin.tz_convert('UTC')
        if drop_initial_warmup and origin - pd.Timedelta(hours=LOOKBACK) < series.index.min():
            dropped += 1
            continue
        context = observed_context(series, origin)
        target_hours = pd.date_range(origin, periods=HORIZON, freq='h')
        if not (target_hours < right).all():
            raise ValueError("24 elapsed-hour targets extend outside the split")
        truth = series.reindex(target_hours).to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(truth).all() or (truth < 0).any():
            raise ValueError("Every target hour must be observed, finite and nonnegative")
        inputs.append(context)
        targets.append(truth)
        origins.append(origin)
    if not origins:
        raise ValueError("No eligible midnight sequences")
    return Sequences(pd.DatetimeIndex(origins), np.stack(inputs), np.stack(targets), dropped)


def build_model(seed=42):
    """48 hourly scalar inputs -> 32-unit LSTM -> 24 demand predictions."""
    keras.utils.set_random_seed(seed)
    with tf.device('/CPU:0'):
        model = keras.Sequential([
            keras.Input(shape=(LOOKBACK, 1)),
            keras.layers.LSTM(32), keras.layers.Dense(HORIZON),
        ], name='citywide_lstm')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def tensor_dataset(inputs, targets, *, batch_size, shuffle=False, seed=42):
    """Keep tf.data CPU resource usage bounded, including private worker threads."""
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(len(inputs), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    return dataset.batch(batch_size).with_options(options)


@dataclass
class LSTMForecaster:
    model: object
    scaler: StandardScaler

    def predict_context(self, context):
        context = np.asarray(context, dtype=float)
        if context.shape != (LOOKBACK, 1) or not np.isfinite(context).all() or (context < 0).any():
            raise ValueError("Context must contain 48 finite nonnegative scalar hours")
        inputs = self.scaler.transform(context).astype('float32')[None, :, :]
        with tf.device('/CPU:0'):
            scaled = self.model(inputs, training=False).numpy()
        if scaled.shape != (1, HORIZON) or not np.isfinite(scaled).all():
            raise ValueError("Model must return 24 finite predictions")
        raw = self.scaler.inverse_transform(scaled.reshape(-1, 1)).ravel().astype(float)
        return np.maximum(raw, 0)  # Same predeclared physical constraint as prior models.


def forecast_next_day(forecaster: LSTMForecaster, series, origin):
    """Issue all 24 predictions together, without reading forecast-day targets."""
    origin = utc_boundary(origin)
    prediction = forecaster.predict_context(observed_context(series, origin))
    return pd.DataFrame({'origin': origin,
                         'hour_start': pd.date_range(origin, periods=HORIZON, freq='h'),
                         'prediction': prediction})


def evaluate_lstm(forecaster, series, start, end):
    # Sequence preparation validates complete targets and exact split boundaries.
    sequences = prepare_sequences(series, start, end)
    blocks = []
    for origin, context, truth in zip(sequences.origins, sequences.inputs, sequences.targets):
        blocks.append(pd.DataFrame({'origin': origin,
                                    'hour_start': pd.date_range(origin, periods=HORIZON, freq='h'),
                                    'prediction': forecaster.predict_context(context), 'actual': truth}))
    frame = pd.concat(blocks, ignore_index=True)
    return frame, {'forecast_origins': len(sequences.origins), 'target_hours': len(frame),
                   **error_metrics(frame.actual, frame.prediction)}


@dataclass
class LSTMTrainingResult:
    forecaster: LSTMForecaster
    history: pd.DataFrame
    best_epoch: int
    training_sequences: int
    validation_sequences: int
    dropped_warmup: int
    duration_seconds: float
    train_mse_scaled: float
    validation_mse_scaled: float
    checkpoint_path: Path
    settings: dict


def fit_lstm(selection_history, checkpoint_path, *, max_epochs=80, patience=10,
             batch_size=4, seed=42):
    """January-only optimization/scaling; validation chooses weights, never test."""
    if selection_history.empty or not (selection_history.index < TEST_START).all():
        raise ValueError("Training/selection history must exclude test observations")
    if any(not isinstance(value, int) or value <= 0 for value in (max_epochs, patience, batch_size)):
        raise ValueError("Epochs, patience and batch size must be positive integers")
    checkpoint_path = Path(checkpoint_path)
    if not str(checkpoint_path).endswith('.weights.h5'):
        raise ValueError("Use a .weights.h5 checkpoint filename")
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint exists: {checkpoint_path}")
    keras.utils.set_random_seed(seed)
    january = selection_history.loc[(selection_history.index >= START) & (selection_history.index < VALIDATION_START)]
    train = prepare_sequences(january, START, VALIDATION_START, drop_initial_warmup=True)
    validation = prepare_sequences(selection_history, VALIDATION_START, TEST_START)
    # One shared univariate scaler fits all observed January training hours.
    if not np.isfinite(january.to_numpy()).all() or not january.ge(0).all():
        raise ValueError("January scaler data must be observed finite nonnegative demand")
    scaler = StandardScaler().fit(january.to_numpy(dtype=float).reshape(-1, 1))
    def scale(values):
        return scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype('float32')
    train_dataset = tensor_dataset(scale(train.inputs), scale(train.targets),
                                   batch_size=batch_size, shuffle=True, seed=seed)
    validation_dataset = tensor_dataset(scale(validation.inputs), scale(validation.targets), batch_size=batch_size)
    model = build_model(seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                       min_delta=0.00001, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_loss',
                                         save_best_only=True, save_weights_only=True),
    ]
    started = time.perf_counter()
    with tf.device('/CPU:0'):
        fitted = model.fit(train_dataset, validation_data=validation_dataset,
                           epochs=max_epochs, callbacks=callbacks, verbose=0)
        # ModelCheckpoint keeps the absolute minimum, independent of min_delta.
        model.load_weights(checkpoint_path)
        train_loss = float(model.evaluate(tensor_dataset(scale(train.inputs), scale(train.targets),
                                                        batch_size=batch_size), verbose=0))
        validation_loss = float(model.evaluate(validation_dataset, verbose=0))
    duration = time.perf_counter() - started
    history = pd.DataFrame({'epoch': range(1, len(fitted.history['loss']) + 1),
                            'train_mse_scaled': fitted.history['loss'],
                            'validation_mse_scaled': fitted.history['val_loss']})
    if not np.isfinite(history[['train_mse_scaled', 'validation_mse_scaled']].to_numpy()).all():
        raise ValueError("Nonfinite training or validation loss")
    best_epoch = int(history.loc[history.validation_mse_scaled.idxmin(), 'epoch'])
    settings = {'seed': seed, 'max_epochs': max_epochs, 'patience': patience,
                'min_delta': 0.00001, 'batch_size': batch_size, 'learning_rate': 0.001,
                'device': 'cpu', 'selection_metric': 'validation_mse_scaled',
                'lookback': LOOKBACK, 'horizon': HORIZON}
    return LSTMTrainingResult(LSTMForecaster(model, scaler), history, best_epoch,
                              len(train.origins), len(validation.origins), train.dropped_warmup,
                              duration, train_loss, validation_loss, checkpoint_path, settings)


def run_lstm_experiment(series, checkpoint_path, **training_options):
    result = fit_lstm(series.loc[series.index < TEST_START].copy(), checkpoint_path, **training_options)
    blocks, metrics = [], []
    for split, left, right in (('validation', VALIDATION_START, TEST_START), ('test', TEST_START, END)):
        predictions, scores = evaluate_lstm(result.forecaster, series, left, right)
        blocks.append(predictions.assign(split=split))
        metrics.append({'split': split, 'model': 'Keras LSTM', **scores})
    return result, pd.concat(blocks, ignore_index=True), pd.DataFrame(metrics)
