"""Synthetic fixtures only; never open the real held-out dataset."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.evaluation import station_followup as f
from citypulse_ai.features.final_station_dataset import SPLITS
from citypulse_ai.features.station_dataset import build_station_panel
from citypulse_ai.forecasting.station_xgboost import paired_metrics


def phase_plan(root, artifacts, destination):
    """Synthetic phase contract, intentionally separate from production freeze."""
    from citypulse_ai.features.final_station_dataset import LazyStationWindows
    data = LazyStationWindows(root, 'test')
    contract = f.calendar_contract(*data.manifest['splits']['test'], data.codes)
    destination.mkdir()
    f.save(destination/'evaluation-plan.json', {'dataset_manifest_sha256': fingerprint(root/'dataset.json'),
        'source_sha256': fingerprint(Path(f.__file__)), 'inputs': {k: fingerprint(p) for k, p in artifacts.items()},
        'contract': contract})
    return contract


@pytest.fixture
def panel():
    origin = pd.Timestamp('2025-11-02T04:00:00Z')
    hours = pd.date_range(origin-pd.Timedelta(hours=336), periods=360, freq='h')
    observed = pd.DataFrame({'station_id': '001', 'hour_start_utc': hours,
                             'ride_count_accepted': np.arange(360)%9})
    quality = observed[['station_id', 'hour_start_utc']].assign(count_complete=True, evidence_id='synthetic')
    frame = build_station_panel(observed, {'001': 1}, start=hours[0],
                               end=hours[-1]+pd.Timedelta(hours=1), quality=quality)
    return origin, frame


def test_causal_original_features_and_fall_dst(panel):
    origin, frame = panel
    x, keys = f.feature_rows(frame, '001', 1, [origin])
    assert x.shape == (24, 48) and len(keys) == 24
    assert list(keys.target_hour_utc.dt.tz_convert('America/New_York').dt.hour).count(1) == 2
    assert keys.target_hour_utc.iloc[-1].tz_convert('America/New_York').hour == 22
    changed = frame.copy()
    changed.loc[changed.hour_start_utc.ge(origin), 'ride_count'] = 123456
    changed['station_available'] = 1.; changed['availability_known_mask'] = True
    changed['status_available_at_utc'] = origin+pd.Timedelta(hours=1)
    xx, yy = f.feature_rows(changed, '001', 1, [origin])
    np.testing.assert_array_equal(x, xx)
    assert not np.array_equal(keys.actual, yy.actual)
    broken = frame.copy(); broken.loc[broken.hour_start_utc.eq(origin), 'label_observed_mask'] = False
    with pytest.raises(ValueError, match='trusted target'): f.feature_rows(broken, '001', 1, [origin])
    with pytest.raises(ValueError, match='grid'): f.feature_rows(frame.iloc[1:], '001', 1, [origin])


def test_exact_keys_order_labels_and_once_only_marker(panel, tmp_path):
    origin, frame = panel
    _, keys = f.feature_rows(frame, '001', 1, [origin])
    contract = f.calendar_contract(origin, origin+pd.Timedelta(hours=25), {'001': 1})
    assert f.keys_and_labels(keys, contract) == contract['paired_key_sha256']
    with pytest.raises(ValueError): f.keys_and_labels(keys.iloc[::-1], contract)
    with pytest.raises(ValueError): f.keys_and_labels(pd.concat([keys.iloc[:-1], keys.iloc[:1]]), contract)
    changed = keys.copy(); changed.loc[0, 'actual'] = np.nan
    with pytest.raises(ValueError): f.keys_and_labels(changed, contract)
    f.save(tmp_path/'started.json', {'once': True})
    with pytest.raises(FileExistsError): f.save(tmp_path/'started.json', {'once': False})


@pytest.mark.parametrize('fault', [None, 'model', 'earlier_test', 'selection', 'keys', 'scaler', 'preprocessing'])
def test_freeze_before_target_reads_and_fail_closed(tmp_path, monkeypatch, fault):
    codes = {f'{i:03d}': i+1 for i in range(100)}
    contract = f.calendar_contract(*SPLITS['test'], codes)
    index = pd.DataFrame([(s, pd.Timestamp(o)) for s in sorted(codes) for o in contract['origins']],
                         columns=['station_id', 'origin_utc'])
    index = index.assign(primary_evaluation_eligible=True, input_eligible=True, trusted_targets=24)
    manifest = {'identity': 'synthetic', 'splits': SPLITS, 'cohort': {'metadata': {
        'train_start_utc': SPLITS['train'][0], 'train_end_utc_exclusive': SPLITS['train'][1]}}}
    (tmp_path/'dataset.json').write_text(json.dumps(manifest))
    (tmp_path/'preprocessing.json').write_text('{}')
    monkeypatch.setattr(f, 'DATASET_SHA', fingerprint(tmp_path/'dataset.json'))
    artifacts = {name: tmp_path/(name+'.json') for name in f.FROZEN}
    reports = {}
    for m, epoch in [('mlp', 10), ('lstm', 18)]:
        artifacts[m+'_checkpoint'].write_bytes(b'synthetic-'+m.encode())
        h = {'epoch': epoch, 'validation': {'mse_scaled': .1, 'clipped': {'MAE': .1}}}
        source = Path(f.__file__).parents[1]/'forecasting'/('station_'+m+'.py')
        reports[m] = {'history': [h], 'best': {'epoch': epoch}, 'splits': SPLITS,
            'checkpoint_sha256': fingerprint(artifacts[m+'_checkpoint']), 'dataset_manifest_sha256': f.DATASET_SHA,
            'training_complete': True, 'fresh_initialization': True, 'test_evaluated': False,
            'source_sha256': fingerprint(source), 'validation': {}, 'config': {}}
    prep = {'station_codes': codes, 'feature_names': list(f.FEATURES), 'training_period': SPLITS['train'],
            'training_rows': 1279200, 'dataset_manifest_sha256': f.DATASET_SHA}
    if fault == 'earlier_test': reports['lstm']['test_evaluated'] = True
    if fault == 'selection': reports['lstm']['history'][0]['epoch'] = 20
    if fault == 'preprocessing': prep['feature_names'] = list(reversed(f.FEATURES))
    for m in reports: artifacts[m+'_report'].write_text(json.dumps(reports[m]))
    artifacts['mlp_preprocessing'].write_text(json.dumps(prep))
    artifacts['lstm_preprocessing'] = tmp_path/'preprocessing.json'
    artifacts['prior_predictions'].write_bytes(b'synthetic-predictions')
    artifacts['prior_report'].write_text(json.dumps({'test_targets': 220800,
        'paired_key_sha256': contract['paired_key_sha256'],
        'prediction_sha256': fingerprint(artifacts['prior_predictions'])}))
    artifacts['prior_plan'].write_text(json.dumps({'contract': contract, 'validation': {}}))
    monkeypatch.setattr(f, 'FROZEN', {k: fingerprint(p) for k, p in artifacts.items()})
    if fault == 'model': artifacts['mlp_checkpoint'].write_bytes(b'changed')
    if fault == 'keys': index = index.iloc[1:]
    scaler = SimpleNamespace(train_start_utc=f.utc(SPLITS['train'][0]), train_end_utc=f.utc(SPLITS['train'][1]))
    if fault == 'scaler': scaler.train_end_utc = f.utc(SPLITS['test'][1])
    monkeypatch.setattr(f, 'MetadataWindows', lambda *a: SimpleNamespace(
        manifest=manifest, codes=codes, scalers={s: scaler for s in codes}, index=index))
    def no_targets(*a, **kw): raise AssertionError('Target body opened during freeze')
    monkeypatch.setattr(pd, 'read_parquet', no_targets)
    if fault:
        with pytest.raises(ValueError): f.freeze(tmp_path, artifacts, tmp_path/'result')
        assert not (tmp_path/'result').exists()
    else:
        plan = f.freeze(tmp_path, artifacts, tmp_path/'result')
        assert plan['contract']['targets'] == 220800 and plan['followup_after_prior_test_disclosure']
        assert not plan['fully_blind_six_model_experiment']
        with pytest.raises(ValueError, match='Existing'): f.freeze(tmp_path, artifacts, tmp_path/'result')


def test_saved_prediction_scoring_join_clipping_and_no_reinference(panel, tmp_path):
    origin, source = panel; _, frame = f.feature_rows(source, '001', 1, [origin])
    contract = f.calendar_contract(origin, origin+pd.Timedelta(hours=25), {'001': 1})
    truth = frame.actual.to_numpy(); metric = paired_metrics(truth, truth)
    prior = {'metrics': {m: dict(metric) for m in ['TFT_raw', 'TFT_clipped', 'XGBoost_raw', 'XGBoost_clipped', 'lag24', 'lag168']}}
    original_v = {'TFT': {**metric, 'nonnegative_clipped': metric},
                  'XGBoost': {'raw': metric, 'clipped': metric}, 'lag24': metric, 'lag168': metric}
    artifacts = {'prior_predictions': tmp_path/'prior.parquet', 'prior_report': tmp_path/'prior.json',
                 'prior_plan': tmp_path/'prior-plan.json'}
    frame.to_parquet(artifacts['prior_predictions'], index=False)
    artifacts['prior_report'].write_text(json.dumps(prior)); artifacts['prior_plan'].write_text('{}')
    plan = {'contract': contract, 'inputs': {k: fingerprint(p) for k, p in artifacts.items()},
            'validation': {m: {'raw': metric, 'clipped': metric} for m in ('MLP', 'LSTM')},
            'original_validation': original_v}
    f.save(tmp_path/'evaluation-plan.json', plan)
    for m in ('mlp', 'lstm'):
        values = frame.copy(); values[m+'_raw'] = truth-1
        values[m+'_clipped'] = np.maximum(truth-1, 0)
        values.to_parquet(tmp_path/(m+'-predictions.parquet'), index=False)
        f.save(tmp_path/(m+'-complete.json'), {'prediction_sha256': fingerprint(tmp_path/(m+'-predictions.parquet'))})
    result = f.summarize(artifacts, tmp_path)
    assert len(result['leaderboard']) == 6 and result['test_targets'] == 24
    assert result['metrics']['MLP_raw']['MAE'] == 1
    assert result['metrics']['MLP_clipped']['MAE'] < 1
    assert not result['existing_four_methods_rerun'] and not result['training_or_selection_performed']
    assert pd.read_parquet(tmp_path/'test-predictions.parquet').actual.equals(frame.actual)
    with pytest.raises(FileExistsError): f.summarize(artifacts, tmp_path)
