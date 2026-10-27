"""Small station MLP fixtures: scaling, missingness, causal data and restoration."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.forecasting.mlp import DemandMLP
from citypulse_ai.forecasting.station_mlp import (
    Config, DevelopmentWindows, encode, fit_preprocessing, predict, restore, train,
)
from citypulse_ai.forecasting.station_xgboost import materialize, paired_metrics
from tests.test_final_station_dataset import saved
from tests.test_station_xgboost import rename_panels


def test_train_only_moments_missing_indicators_and_station_one_hot():
    x = np.ones((10, 48), dtype=np.float32); x[:, 0] = 1
    x[:, 1] = np.arange(10); x[0, 2] = np.nan; x[:, 47] = np.nan
    y = np.arange(10, dtype=np.float32)
    p = fit_preprocessing(x, y, {'001': 1, 'exact.00_': 2}, chunk=3)
    assert p['numeric_mean'][0] == pytest.approx(4.5)
    assert p['numeric_scale'][0] == pytest.approx(np.std(np.arange(10)))
    assert p['numeric_observed_rows'][-1] == 0 and p['numeric_scale'][-1] == 1
    before = json.dumps(p, sort_keys=True)
    validation = x.copy(); validation[:, 1] = 99999; validation[0, 0] = 2
    result = encode(validation, p)
    assert result.shape == (10, 96) and result[0, :2].tolist() == [0, 1]
    assert result[0, 3] == 0 and result[0, 50] == 1  # numeric missing value and flag
    assert json.dumps(p, sort_keys=True) == before
    validation[0, 0] = 3
    with pytest.raises(ValueError, match='station code'):
        encode(validation, p)


def test_native_100_station_architecture_fresh_seed_and_finite_gradients():
    torch.manual_seed(42); a = DemandMLP(194)
    torch.manual_seed(42); b = DemandMLP(194)
    assert sum(p.numel() for p in a.parameters()) == 6785
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    optimizer = torch.optim.Adam(a.parameters(), lr=.001)
    loss = a(torch.ones(24, 194)).square().mean(); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in a.parameters())
    optimizer.step()
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert a(torch.zeros(24, 194)).shape == (24, 1)


def test_development_verification_never_hashes_test_quarter(saved, monkeypatch):
    root, *_ = saved; rename_panels(root)
    # Add a deliberately unavailable test-quarter object. Its bytes must never
    # be opened; the trusted full manifest still names it for eventual testing.
    p = root/'dataset.json'; d = json.loads(p.read_text())
    receipt = {'name': 'panel-2025-10.parquet', 'size_bytes': 999, 'sha256': 'unopened'}
    d['panels'].append(receipt); d['artifacts'].append(receipt); p.write_text(json.dumps(d))
    data = DevelopmentWindows(root, 'validation')
    assert len(data) > 0
    with pytest.raises(ValueError, match='Test split'):
        DevelopmentWindows(root, 'test')
    (root/'panel-2024-01.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact changed'):
        DevelopmentWindows(root, 'train')


def reference(root, destination):
    a = materialize(root, 'train', destination/'train', reader_factory=DevelopmentWindows)
    b = materialize(root, 'validation', destination/'validation', reader_factory=DevelopmentWindows)
    keys = pd.read_parquet(destination/'validation/keys.parquet')
    value = {'dataset_manifest_sha256': fingerprint(root/'dataset.json'), 'train_windows': a['forecast_windows'],
             'test_evaluated': False, 'validation': {'forecast_windows': b['forecast_windows'],
             'targets': b['rows'], 'paired_key_sha256': b['paired_key_sha256'],
             'baselines': {f'lag_{lag}': paired_metrics(keys.actual, keys[f'lag{lag}']) for lag in (24, 168)}}}
    path = destination/'reference.json'; path.write_text(json.dumps(value)); return path


def test_tiny_training_complete_epochs_selected_restore_and_no_test_reads(saved, tmp_path):
    root, *_ = saved; rename_panels(root)
    ref = reference(root, tmp_path)
    out = tmp_path/'run'
    report = train(root, out, ref, config=Config(stations=2, max_epochs=2, patience=2, batch_size=256), max_seconds=1200)
    assert report['epochs_completed'] == 2 and report['training_complete'] and not report['test_evaluated']
    assert all(h['training_rows'] == report['training']['rows'] for h in report['history'])
    assert report['best']['validation_clipped_MAE'] == min(h['validation']['clipped']['MAE'] for h in report['history'])
    _, network, prep = restore(out/report['best']['checkpoint'], out/'preprocessing.json')
    probe = np.load(out/'reload-probe.npy')
    np.testing.assert_array_equal(predict(network, probe, prep), np.load(out/'reload-expected.npy'))
    assert prep['training_rows'] == report['training']['rows'] and prep['target_mean'] < 9
    with pytest.raises(ValueError, match='Existing run'):
        train(root, out, ref)


def test_mismatched_reference_stops_before_optimization(saved, tmp_path, monkeypatch):
    root, *_ = saved; rename_panels(root); ref = reference(root, tmp_path)
    value = json.loads(ref.read_text()); value['validation']['paired_key_sha256'] = 'changed'; ref.write_text(json.dumps(value))
    def forbidden(*args, **kwargs):
        raise AssertionError('Optimizer constructed before reference validation')
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    with pytest.raises(ValueError, match='fitting prohibited'):
        train(root, tmp_path/'invalid', ref, config=Config(stations=2))
