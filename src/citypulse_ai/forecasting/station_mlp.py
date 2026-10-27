"""Fresh pooled station MLP; training/validation only, bounded feature batches."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.final_station_dataset import LazyStationWindows
from citypulse_ai.features.station_dataset import utc
from citypulse_ai.forecasting.mlp import DemandMLP
from citypulse_ai.forecasting.station_xgboost import FEATURES, materialize, paired_metrics, write_json


@dataclass(frozen=True)
class Config:
    stations: int = 100
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = .0001
    batch_size: int = 2048
    learning_rate: float = .001
    seed: int = 42
    cpu_threads: int = 2


class DevelopmentWindows(LazyStationWindows):
    """Verify only required train/validation artifacts; never open test panels.

    The original manifest's trusted SHA binds the frozen full dataset. This
    scoped verification is deliberately not advertised as rechecking test data.
    The existing reader opens window indices (keys/masks, not target values).
    """
    def __init__(self, root, split):
        if split not in ('train', 'validation'):
            raise ValueError('Test split is not authorized')
        super().__init__(root, split)

    def verify(self):
        end = utc(self.manifest['splits']['validation'][1])
        required = {'preprocessing.json', 'windows.parquet'}
        required.update(r['name'] for r in self.manifest['panels'] if
            pd.Timestamp(r['name'].removeprefix('panel-').removesuffix('.parquet')+'-01',
                         tz='America/New_York').tz_convert('UTC') < end)
        receipts = {r['name']: r for r in self.manifest['artifacts']}
        for name in required:
            r = receipts[name]; p = self.root/name
            if (Path(name).name != name or p.is_symlink() or p.stat().st_size != r['size_bytes']
                    or fingerprint(p) != r['sha256']):
                raise ValueError('Development artifact changed: '+name)


def fit_preprocessing(features, targets, codes, *, chunk=8192):
    """Train-row moments; missing numeric values get standardized zero + a flag."""
    count = np.zeros(47, dtype=np.int64); mean = np.zeros(47); m2 = np.zeros(47)
    yn = 0; ym = ym2 = 0.
    for start in range(0, len(targets), chunk):
        raw = np.asarray(features[start:start+chunk, 1:], dtype=float)
        if raw.shape[1] != 47 or np.isinf(raw).any():
            raise ValueError('Invalid training feature schema')
        valid = np.isfinite(raw); n = valid.sum(axis=0)
        sums = np.where(valid, raw, 0).sum(axis=0)
        block_mean = np.divide(sums, n, out=np.zeros(47), where=n > 0)
        block_m2 = np.square(np.where(valid, raw-block_mean, 0)).sum(axis=0)
        total = count+n; delta = block_mean-mean
        mean += np.divide(delta*n, total, out=np.zeros(47), where=total > 0)
        m2 += block_m2+np.divide(delta**2*count*n, total, out=np.zeros(47), where=total > 0)
        count = total
        y = np.asarray(targets[start:start+chunk], dtype=float)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Invalid trusted training labels')
        bn = len(y); bm = float(y.mean()); delta_y = bm-ym; total_y = yn+bn
        ym2 += float(np.square(y-bm).sum())+delta_y**2*yn*bn/total_y
        ym += delta_y*bn/total_y; yn = total_y
    if not yn:
        raise ValueError('No training rows')
    scale = np.sqrt(np.divide(m2, count, out=np.zeros(47), where=count > 0))
    scale[scale < 1e-12] = 1
    return {'format_version': 1, 'feature_names': list(FEATURES), 'station_codes': codes,
            'numeric_mean': mean.tolist(), 'numeric_scale': scale.tolist(),
            'numeric_observed_rows': count.tolist(), 'training_rows': yn,
            'target_mean': ym, 'target_scale': float(np.sqrt(ym2/yn)) if ym2/yn > 1e-24 else 1.,
            'missing_policy': 'standardized zero placeholder plus independent missing flag',
            'fit_scope': 'training forecast-target rows only; no validation fitting'}


def encode(raw, preprocessing):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 48 or np.isinf(raw).any():
        raise ValueError('Expected 48 raw features')
    codes = raw[:, 0]; size = len(preprocessing['station_codes'])
    if not np.isfinite(codes).all() or not np.equal(codes, np.floor(codes)).all() or not ((codes >= 1) & (codes <= size)).all():
        raise ValueError('Unknown frozen station code')
    numeric = raw[:, 1:]; missing = ~np.isfinite(numeric)
    values = (numeric-np.asarray(preprocessing['numeric_mean']))/np.asarray(preprocessing['numeric_scale'])
    values[missing] = 0
    one_hot = np.zeros((len(raw), size), dtype=np.float32)
    one_hot[np.arange(len(raw)), codes.astype(int)-1] = 1
    return np.concatenate([one_hot, values.astype(np.float32), missing.astype(np.float32)], axis=1)


def predict(model, features, preprocessing, batch_size=2048):
    output = np.empty(len(features), dtype=np.float64); model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.from_numpy(encode(features[start:start+batch_size], preprocessing))
            prediction = model(values).squeeze(-1).numpy().astype(float)
            output[start:start+len(values)] = prediction*preprocessing['target_scale']+preprocessing['target_mean']
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite predictions')
    return output


def restore(path, preprocessing_path):
    prep = json.loads(Path(preprocessing_path).read_text())
    cp = torch.load(path, map_location='cpu', weights_only=True)
    if cp['preprocessing_sha256'] != fingerprint(Path(preprocessing_path)):
        raise ValueError('Checkpoint preprocessing differs')
    model = DemandMLP(len(prep['station_codes'])+94).cpu()
    model.load_state_dict(cp['state_dict']); model.eval()
    return cp, model, prep


def train(root, out, reference, *, config=Config(), max_seconds=9000,
          progress=lambda stage: None, publish=None):
    """One fixed configuration; no test entry point and no inherited weights."""
    if min(config.max_epochs, config.patience, config.batch_size, config.cpu_threads, config.stations) < 1:
        raise ValueError('Positive training limits required')
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Existing run; inspect completed checkpoints instead of retrying')
    torch.set_num_threads(config.cpu_threads); torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    import pyarrow as pa
    pa.set_cpu_count(2); pa.set_io_thread_count(2)
    started = time.monotonic(); deadline = started+max_seconds
    fixed = json.loads(Path(reference).read_text()); dataset_sha = fingerprint(root/'dataset.json')
    if fixed['dataset_manifest_sha256'] != dataset_sha or fixed['test_evaluated']:
        raise ValueError('Use the frozen pre-test validation reference')
    out.mkdir(parents=True)
    manifest = json.loads((root/'dataset.json').read_text())
    codes = json.loads((root/'preprocessing.json').read_text())['station_codes']
    if len(codes) != config.stations or set(codes.values()) != set(range(1, config.stations+1)):
        raise ValueError('Frozen station vocabulary mismatch')
    contract = {'architecture': f'{config.stations+94} -> 32 ReLU -> 16 ReLU -> 1',
                'config': asdict(config), 'fresh_initialization': True, 'optimizer': 'Adam',
                'training_loss': 'global training-target standardized MSE',
                'selection': 'lowest validation clipped MAE; exact ties keep earlier epoch',
                'early_stopping': 'patience epochs without min_delta improvement',
                'dataset_manifest_sha256': dataset_sha, 'splits': manifest['splits'],
                'reference_sha256': fingerprint(Path(reference)),
                'source_sha256': fingerprint(Path(__file__)), 'test_evaluated': False}
    write_json(out/'configuration.json', contract)
    arrays = {}; metas = {}
    for split in ('train', 'validation'):
        progress('features_'+split)
        metas[split] = materialize(root, split, out/('scratch-'+split), progress=progress,
                                  reader_factory=DevelopmentWindows)
        folder = out/('scratch-'+split)
        arrays[split] = (np.load(folder/'features.npy', mmap_mode='r'), np.load(folder/'labels.npy', mmap_mode='r'))
    v = metas['validation']; expected = fixed['validation']
    if (metas['train']['forecast_windows'] != fixed['train_windows']
            or v['forecast_windows'] != expected['forecast_windows'] or v['rows'] != expected['targets']
            or v['paired_key_sha256'] != expected['paired_key_sha256']):
        raise ValueError('Paired validation population differs; fitting prohibited')
    keys = pd.read_parquet(out/'scratch-validation/keys.parquet')
    baselines = {f'lag_{lag}': paired_metrics(keys.actual, keys[f'lag{lag}']) for lag in (24, 168)}
    for name, scores in baselines.items():
        for metric in ('targets', 'MAE', 'RMSE', 'WAPE_percent'):
            if not np.isclose(scores[metric], expected['baselines'][name][metric], rtol=0, atol=1e-9):
                raise ValueError('Paired baseline differs; fitting prohibited')
    x, y = arrays['train']; vx, vy = arrays['validation']
    preprocessing = fit_preprocessing(x, y, codes)
    preprocessing.update(dataset_manifest_sha256=dataset_sha, training_period=manifest['splits']['train'])
    write_json(out/'preprocessing.json', preprocessing)
    torch.manual_seed(config.seed)
    model = DemandMLP(config.stations+94).cpu()
    initial_sha = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []; best = None; stale = 0; patience_reference = float('inf'); receipts = []
    stop_reason = 'fixed_epoch_budget'
    for epoch in range(1, config.max_epochs+1):
        anticipated = history[-1]['epoch_seconds']*1.3 if history else 60
        if time.monotonic()+anticipated+600 > deadline:
            stop_reason = 'runtime_before_next_epoch'; break
        progress(f'train_epoch_{epoch}'); epoch_started = time.monotonic(); total_loss = 0.; seen = 0
        model.train(); order = np.random.default_rng(config.seed+epoch).permutation(len(y))
        for start in range(0, len(order), config.batch_size):
            if time.monotonic() > deadline-600:
                stop_reason = 'partial_epoch_discarded'; break
            indices = order[start:start+config.batch_size]
            inputs = torch.from_numpy(encode(x[indices], preprocessing))
            targets = torch.from_numpy(((np.asarray(y[indices], dtype=float)-preprocessing['target_mean'])/
                                       preprocessing['target_scale']).astype(np.float32)).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total_loss += float(loss.detach())*len(indices); seen += len(indices)
        if stop_reason == 'partial_epoch_discarded':
            break
        if seen != len(y):
            raise ValueError('Incomplete epoch')
        progress(f'validate_epoch_{epoch}')
        prediction = predict(model, vx, preprocessing, config.batch_size)
        raw = paired_metrics(vy, prediction); clipped = paired_metrics(vy, np.maximum(prediction, 0))
        record = {'epoch': epoch, 'training_mse_scaled': total_loss/seen, 'training_rows': seen,
                  'validation': {'raw': raw, 'clipped': clipped}, 'epoch_seconds': time.monotonic()-epoch_started}
        history.append(record); score = clipped['MAE']
        if best is None or score < best['validation_clipped_MAE']:
            best = {'epoch': epoch, 'checkpoint': f'epoch-{epoch:03d}.pt', 'validation_clipped_MAE': score}
        if score < patience_reference-config.min_delta:
            stale = 0; patience_reference = score
        else:
            stale += 1
        cp = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
              'torch_rng_state': torch.get_rng_state(), 'configuration': contract, 'history': history.copy(),
              'best': best.copy(), 'stale_epochs': stale, 'patience_reference': patience_reference,
              'preprocessing_sha256': fingerprint(out/'preprocessing.json'), 'initial_state_sha256': initial_sha}
        path = out/f'epoch-{epoch:03d}.pt'
        with path.open('xb') as stream:
            torch.save(cp, stream)
        if publish:
            receipts.append(publish(path))
        write_json(out/'history.json', history); write_json(out/'checkpoint-receipts.json', receipts)
        if stale >= config.patience:
            stop_reason = 'validation_early_stopping'; break
    if best is None:
        raise RuntimeError('No completed epoch; no checkpoint selected')
    cp, selected, prep = restore(out/best['checkpoint'], out/'preprocessing.json')
    prediction = predict(selected, vx, prep, config.batch_size)
    keys['prediction_raw'] = prediction; keys['prediction_clipped'] = np.maximum(prediction, 0)
    keys.to_parquet(out/'validation-predictions.parquet', index=False, compression='zstd')
    scores = {'raw': paired_metrics(keys.actual, prediction), 'clipped': paired_metrics(keys.actual, np.maximum(prediction, 0))}
    if scores != history[best['epoch']-1]['validation']:
        raise ValueError('Selected checkpoint metrics differ on reload')
    probe = np.unique(np.linspace(0, len(vx)-1, min(100, len(vx)), dtype=int))
    probe_values = np.asarray(vx[probe])
    probe_expected = predict(selected, probe_values, prep, config.batch_size)
    if not np.allclose(probe_expected, prediction[probe], rtol=1e-6, atol=1e-5):
        raise ValueError('Probe differs materially across batch shapes')
    np.save(out/'reload-probe.npy', probe_values); np.save(out/'reload-expected.npy', probe_expected)
    report = {**contract, 'status': 'station_mlp_validation_complete', 'history': history,
              'epochs_completed': len(history), 'best': best, 'stop_reason': stop_reason,
              'training_complete': stop_reason in ('fixed_epoch_budget', 'validation_early_stopping'),
              'parameters': sum(p.numel() for p in selected.parameters()), 'torch': str(torch.__version__),
              'training': metas['train'], 'validation': {**v, **scores}, 'baselines': baselines,
              'initial_state_sha256': initial_sha, 'checkpoint_sha256': fingerprint(out/best['checkpoint']),
              'elapsed_seconds': time.monotonic()-started, 'test_evaluated': False}
    write_json(out/'experiment.json', report)
    return report
