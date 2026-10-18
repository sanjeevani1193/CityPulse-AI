"""Bridge the existing framework-neutral station windows to CPU TFT batches."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from citypulse_ai.features.station_dataset import (
    KNOWN_FEATURES, OBSERVED_FEATURES, CountScaler, fit_count_scalers, utc,
)


def fit_training_scalers(panel, *, train_start, train_end):
    """Use unique trusted training hours, rejecting corrupt trusted values."""
    start, end = utc(train_start), utc(train_end)
    trusted = panel.loc[panel.hour_start_utc.ge(start) & panel.hour_start_utc.lt(end)
                        & panel.label_observed_mask, 'ride_count']
    values = trusted.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError('Trusted training counts must be finite nonnegative integers')
    return fit_count_scalers(panel, train_start=start, train_end=end)


@dataclass
class TFTBatch:
    static_station_code: torch.Tensor
    encoder_known: torch.Tensor
    encoder_observed: torch.Tensor
    decoder_known: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    means: torch.Tensor
    scales: torch.Tensor

    def inputs(self):
        # Targets and label masks are intentionally absent from model arguments.
        return (self.static_station_code, self.encoder_known,
                self.encoder_observed, self.decoder_known)


def windows_to_batch(windows, scalers):
    if not windows:
        raise ValueError('A nonempty window list is required')
    for w in windows:
        if (w.metadata['scaling'] != 'frozen_training_only'
                or tuple(w.metadata['feature_order']['known']) != KNOWN_FEATURES
                or tuple(w.metadata['feature_order']['observed']) != OBSERVED_FEATURES):
            raise ValueError('Windows require the exact feature schema and frozen scaling')
    def stack(name, dtype):
        return torch.as_tensor(np.stack([getattr(w, name) for w in windows]), dtype=dtype, device='cpu')
    means = torch.tensor([scalers[w.station_id].mean for w in windows], dtype=torch.float32)
    scales = torch.tensor([scalers[w.station_id].scale for w in windows], dtype=torch.float32)
    if not torch.isfinite(means).all() or not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('Invalid frozen scaler')
    mask = stack('target_mask', torch.bool)
    raw = stack('target', torch.float32)
    # Only trusted values are transformed; missing placeholders never become labels.
    targets = torch.zeros_like(raw)
    targets[mask] = ((raw[mask] - means[:, None, None].expand_as(raw)[mask])
                     / scales[:, None, None].expand_as(raw)[mask])
    if not torch.isfinite(targets).all():
        raise ValueError('Nonfinite trusted targets or invalid scalers')
    return TFTBatch(stack('static_station_code', torch.long), stack('encoder_known', torch.float32),
        stack('encoder_observed', torch.float32), stack('decoder_known', torch.float32),
        targets, mask, means, scales)


def save_preprocessing(path, scalers, station_codes):
    """Small JSON beside, never inside, the model state; refuse overwrites."""
    if set(scalers) != set(station_codes):
        raise ValueError('Station vocabulary and scalers differ')
    data = {'format_version': 1, 'station_codes': station_codes,
        'known_features': list(KNOWN_FEATURES), 'observed_features': list(OBSERVED_FEATURES),
        'scalers': {station: {'mean': s.mean, 'scale': s.scale, 'sample_count': s.sample_count,
            'train_start_utc': s.train_start_utc.isoformat(), 'train_end_utc': s.train_end_utc.isoformat()}
            for station, s in scalers.items()}}
    _decode_preprocessing(data)
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _decode_preprocessing(data):
    if (data['format_version'] != 1 or data['known_features'] != list(KNOWN_FEATURES)
            or data['observed_features'] != list(OBSERVED_FEATURES)):
        raise ValueError('Preprocessing feature schema differs')
    codes = data['station_codes']
    if (not codes or set(codes) != set(data['scalers']) or len(set(codes.values())) != len(codes)
            or any(not isinstance(s, str) or not s.strip() or type(c) is not int or c < 1 for s, c in codes.items())):
        raise ValueError('Exact station IDs and unique positive codes required')
    scalers = {}
    for station, value in data['scalers'].items():
        s = CountScaler(value['mean'], value['scale'], value['sample_count'],
                        utc(value['train_start_utc']), utc(value['train_end_utc']))
        if (not np.isfinite([s.mean, s.scale]).all() or s.scale <= 0
                or type(s.sample_count) is not int or s.sample_count < 1
                or s.train_start_utc >= s.train_end_utc):
            raise ValueError('Invalid frozen scaler')
        scalers[station] = s
    return scalers, codes


def load_preprocessing(path):
    return _decode_preprocessing(json.loads(Path(path).read_text()))
