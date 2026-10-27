"""Synthetic calendar/scoring checks; never open real held-out observations."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from citypulse_ai.evaluation.station_benchmark import calendar_contract, quantile_metrics, save
from citypulse_ai.features.final_station_dataset import SPLITS


def test_final_calendar_has_identical_keys_and_explicit_fall_gap():
    codes = {'001': 1, 'exact.00_': 2}
    plan = calendar_contract(*SPLITS['test'], codes)
    assert plan['forecast_windows'] == 184 and plan['targets'] == 4416
    assert plan['unscored_station_hours'] == 2
    assert plan['unscored_hours_utc_per_station'] == ['2025-11-03T04:00:00+00:00']
    assert plan == calendar_contract(*SPLITS['test'], dict(reversed(list(codes.items()))))
    origin = pd.Timestamp('2025-11-02T04:00:00Z')
    horizon = pd.date_range(origin, periods=24, freq='h').tz_convert('America/New_York')
    assert list(horizon.hour).count(1) == 2
    assert horizon[1].utcoffset() != horizon[2].utcoffset()
    assert horizon[-1].hour == 22


def test_digest_binds_exact_station_origin_target_and_lead():
    start, end = '2025-10-01T00:00:00-04:00', '2025-10-02T00:00:00-04:00'
    result = calendar_contract(start, end, {'001': 1})
    origin = pd.Timestamp(start).tz_convert('UTC'); digest = hashlib.sha256()
    for lead in range(24):
        hour = origin+pd.Timedelta(hours=lead)
        digest.update(f'001|{origin.isoformat()}|{hour.isoformat()}|{lead+1}\n'.encode())
    assert digest.hexdigest() == result['paired_key_sha256']
    assert result['paired_key_sha256'] != calendar_contract(start, end, {'1': 1})['paired_key_sha256']


def test_original_unit_quantile_metrics_raw_and_clipped():
    actual = np.array([0., 4.]); raw = np.array([[-3., -1., 2.], [-1., 3., 5.]])
    scores = quantile_metrics(actual, raw); clipped = quantile_metrics(actual, np.maximum(raw, 0))
    assert scores['targets'] == clipped['targets'] == 2
    assert scores['MAE'] == 1 and clipped['MAE'] == .5
    assert scores['negative_median_fraction'] == .5 and clipped['negative_median_fraction'] == 0
    assert scores['interval_coverage'] == clipped['interval_coverage'] == 1
    assert clipped['pinball_by_quantile']['q50'] == .25
    with pytest.raises(ValueError, match='Invalid quantiles'):
        quantile_metrics(actual, raw[:, ::-1])
    with pytest.raises(ValueError, match='Invalid quantiles'):
        quantile_metrics(actual, np.full((2, 3), np.nan))


def test_once_only_artifacts_cannot_silently_restart(tmp_path):
    path = tmp_path/'test-started.json'
    save(path, {'began': True})
    with pytest.raises(FileExistsError):
        save(path, {'began': False})


def test_partial_split_horizon_is_not_scored():
    result = calendar_contract('2025-10-01T00:00:00-04:00', '2025-10-02T12:00:00-04:00', {'A': 1})
    assert result['targets'] == 24 and result['unscored_station_hours'] == 12


@pytest.mark.parametrize('fault', [None, 'artifact', 'earlier_test', 'keys', 'scaler', 'features'])
def test_preflight_fails_closed_without_reading_target_bodies(tmp_path, monkeypatch, fault):
    import json
    from types import SimpleNamespace
    from citypulse_ai.evaluation import station_benchmark as b
    from citypulse_ai.etl.archives import fingerprint

    codes = {f'{i:03d}': i+1 for i in range(100)}
    contract = calendar_contract(*SPLITS['test'], codes)
    index = pd.DataFrame([(s, pd.Timestamp(o)) for s in sorted(codes) for o in contract['origins']],
                         columns=['station_id', 'origin_utc'])
    index = index.assign(primary_evaluation_eligible=True, input_eligible=True, trusted_targets=24)
    scaler = SimpleNamespace(train_start_utc=pd.Timestamp(SPLITS['train'][0]),
                             train_end_utc=pd.Timestamp(SPLITS['train'][1]))
    cohort = {'metadata': {'train_start_utc': SPLITS['train'][0],
                           'train_end_utc_exclusive': SPLITS['train'][1]}}
    manifest = {'identity': 'synthetic', 'splits': SPLITS, 'cohort': cohort, 'replay': 'retrospective'}
    (tmp_path/'dataset.json').write_text(json.dumps(manifest))
    checkpoint, model = tmp_path/'epoch.pt', tmp_path/'model.ubj'
    checkpoint.write_bytes(b'synthetic-checkpoint'); model.write_bytes(b'synthetic-trees')
    sha = fingerprint(tmp_path/'dataset.json')
    monkeypatch.setattr(b, 'DATASET_SHA', sha)
    monkeypatch.setattr(b, 'TFT_SHA', fingerprint(checkpoint))
    monkeypatch.setattr(b, 'XGB_SHA', fingerprint(model))
    tft = {'dataset_manifest_sha256': sha, 'dataset_identity': 'synthetic', 'best': {'epoch': 6},
           'epochs_completed': 6, 'selected_checkpoint': {'sha256': fingerprint(checkpoint)},
           'test_evaluated': False, 'validation': {}}
    xgb = {'dataset_manifest_sha256': sha, 'dataset_identity': 'synthetic',
           'model_sha256': fingerprint(model), 'test_evaluated': False, 'refit_on_validation': False,
           'selected': {'metrics': {}}, 'baselines': {'lag24': {}, 'lag168': {}}}
    prep = {'feature_names': list(b.FEATURES), 'feature_types': list(b.FEATURE_TYPES),
            'station_codes': codes, 'dataset_manifest_sha256': sha}
    if fault == 'artifact':
        model.write_bytes(b'changed')
    elif fault == 'earlier_test':
        tft['test_evaluated'] = True
    elif fault == 'keys':
        index = index.iloc[1:]
    elif fault == 'scaler':
        scaler.train_end_utc = pd.Timestamp(SPLITS['test'][1])
    elif fault == 'features':
        prep['feature_names'] = prep['feature_names'][::-1]
    for name, value in [('tft.json', tft), ('xgb.json', xgb), ('preprocessing.json', prep)]:
        (tmp_path/name).write_text(json.dumps(value))
    monkeypatch.setattr(b, 'LazyStationWindows', lambda *args: SimpleNamespace(
        codes=codes, manifest=manifest, scalers={s: scaler for s in codes}, index=index))
    def no_read(*args, **kwargs):
        raise AssertionError('Preflight attempted to read target panel bodies')
    monkeypatch.setattr(pd, 'read_parquet', no_read)
    args = (tmp_path, checkpoint, model, tmp_path/'preprocessing.json',
            tmp_path/'tft.json', tmp_path/'xgb.json', tmp_path/'evaluation')
    if fault:
        with pytest.raises(ValueError):
            b.preflight(*args)
        assert not (tmp_path/'evaluation/evaluation-plan.json').exists()
    else:
        assert b.preflight(*args)['contract']['targets'] == 220800
