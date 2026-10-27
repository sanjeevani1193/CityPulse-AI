"""Frozen, once-only held-out evaluation; training entry points remain unchanged.

TFT and XGBoost phases run in their existing separate CPU environments. Only
the TFT phase opens test panels. The XGBoost phase consumes its causal feature
memmap and shared keys, never constructs its own target population.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.final_station_dataset import LazyStationWindows, SPLITS
from citypulse_ai.features.station_dataset import make_window, utc
from citypulse_ai.forecasting.station_xgboost import FEATURES, FEATURE_TYPES, feature_block, paired_metrics

DATASET_SHA = 'a6747adb2485276e2e7aa735dc24b1a10048e9b8526928c489a9472a532ba952'
TFT_SHA = '5396d276b1d05f84ffcd969767b4319faa659cb2107c022e3fd87d70c7fe12c6'
XGB_SHA = 'f7a64cc25e38893b0a566dab2957034fd81f8db6bdf2c712abace79015451b2a'


def save(path, value):
    """Completed artifacts are create-only; a failed run is never silently retried."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def calendar_contract(start, end, codes):
    """Target identity depends on the frozen cohort/calendar, never target values."""
    start, end = utc(start), utc(end)
    origins = pd.date_range(start.tz_convert('America/New_York'),
                            end.tz_convert('America/New_York'), freq='D', inclusive='left').tz_convert('UTC')
    origins = origins[origins + pd.Timedelta(hours=24) <= end]
    hours = pd.DatetimeIndex([o + pd.Timedelta(hours=h) for o in origins for h in range(24)])
    if not hours.is_unique or not hours.is_monotonic_increasing:
        raise ValueError('Repeated or unordered target hours')
    full = pd.date_range(start, end, freq='h', inclusive='left')
    digest = hashlib.sha256()
    for station in sorted(codes):
        for origin in origins:
            for lead in range(24):
                hour = origin + pd.Timedelta(hours=lead)
                digest.update(f'{station}|{origin.isoformat()}|{hour.isoformat()}|{lead+1}\n'.encode())
    return {'origins': [o.isoformat() for o in origins], 'station_ids': sorted(codes),
            'forecast_windows': len(origins)*len(codes), 'targets': len(hours)*len(codes),
            'paired_key_sha256': digest.hexdigest(),
            'unscored_hours_utc_per_station': [h.isoformat() for h in full.difference(hours)],
            'unscored_station_hours': (len(full)-len(hours))*len(codes)}


def preflight(root, checkpoint, xgb_model, preprocessing, tft_report, xgb_report, out):
    """Hash files and inspect saved indices/metadata only: no panel body reads."""
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Existing evaluation directory; inspect instead of retrying')
    for path, expected in [(root/'dataset.json', DATASET_SHA), (checkpoint, TFT_SHA), (xgb_model, XGB_SHA)]:
        if fingerprint(Path(path)) != expected:
            raise ValueError('Frozen artifact identity mismatch')
    data = LazyStationWindows(root, 'test')
    tft, xgb = json.loads(Path(tft_report).read_text()), json.loads(Path(xgb_report).read_text())
    prep = json.loads(Path(preprocessing).read_text())
    if (data.manifest['splits'] != SPLITS or len(data.codes) != 100
            or tft['dataset_manifest_sha256'] != DATASET_SHA or xgb['dataset_manifest_sha256'] != DATASET_SHA
            or tft['best']['epoch'] != 6 or tft['epochs_completed'] != 6
            or tft['selected_checkpoint']['sha256'] != TFT_SHA or xgb['model_sha256'] != XGB_SHA
            or tft['test_evaluated'] or xgb['test_evaluated'] or xgb['refit_on_validation']
            or tft['dataset_identity'] != data.manifest['identity'] or xgb['dataset_identity'] != data.manifest['identity']):
        raise ValueError('Frozen experiment contract mismatch')
    if (prep['feature_names'] != list(FEATURES) or prep['feature_types'] != list(FEATURE_TYPES)
            or prep['station_codes'] != data.codes or prep['dataset_manifest_sha256'] != DATASET_SHA):
        raise ValueError('XGBoost preprocessing differs')
    if any(s.train_start_utc < utc(SPLITS['train'][0]) or s.train_end_utc > utc(SPLITS['train'][1])
           for s in data.scalers.values()):
        raise ValueError('Scaler fitting leaks outside training')
    cohort = data.manifest['cohort']['metadata']
    if (utc(cohort['train_start_utc']) != utc(SPLITS['train'][0])
            or utc(cohort['train_end_utc_exclusive']) != utc(SPLITS['train'][1])):
        raise ValueError('Cohort was not selected on the fixed training period')
    contract = calendar_contract(*SPLITS['test'], data.codes)
    expected = [(s, utc(o)) for s in sorted(data.codes) for o in contract['origins']]
    index = data.index.sort_values(['station_id', 'origin_utc'])
    if (list(zip(index.station_id, index.origin_utc)) != expected
            or not index.primary_evaluation_eligible.all() or not index.input_eligible.all()
            or not index.trusted_targets.eq(24).all()):
        raise ValueError('Common key eligibility differs; stop before reading targets')
    out.mkdir(parents=True)
    plan = {'status': 'frozen_before_test_panel_reads', 'dataset_manifest_sha256': DATASET_SHA,
            'dataset_identity': data.manifest['identity'], 'TFT_sha256': TFT_SHA, 'XGBoost_sha256': XGB_SHA,
            'preprocessing_sha256': fingerprint(Path(preprocessing)), 'contract': contract,
            'splits': SPLITS, 'encoder_hours': 336, 'horizon_hours': 24,
            'point_rule': 'max(0, raw original-unit point forecast)',
            'quantile_rule': 'max(0, each raw inverse-scaled quantile)',
            'replay': data.manifest['replay'], 'selection': 'frozen before test; no refit or tuning',
            'tft_report_sha256': fingerprint(Path(tft_report)), 'xgb_report_sha256': fingerprint(Path(xgb_report)),
            'validation': {'TFT': tft['validation'], 'XGBoost': xgb['selected']['metrics'],
                           'lag24': xgb['baselines']['lag24'], 'lag168': xgb['baselines']['lag168']}}
    save(out/'evaluation-plan.json', plan)
    return plan


def tft_phase(root, checkpoint, out, *, progress=lambda stage: None):
    """One chronological, station-block inference pass; targets never enter inputs."""
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from citypulse_ai.forecasting.tft_training import restore
    from citypulse_ai.forecasting.tft_preparation import windows_to_batch
    from citypulse_ai.forecasting.tft import inverse_quantiles

    root, out = Path(root), Path(out)
    plan = json.loads((out/'evaluation-plan.json').read_text())
    if fingerprint(Path(checkpoint)) != plan['TFT_sha256'] or fingerprint(root/'dataset.json') != DATASET_SHA:
        raise ValueError('Frozen inputs changed')
    save(out/'test-started.json', {'phase': 'TFT', 'plan_sha256': fingerprint(out/'evaluation-plan.json')})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    pa.set_cpu_count(2); pa.set_io_thread_count(2)
    cp, model, optimizer = restore(checkpoint, DATASET_SHA)
    if cp['epoch'] != 6 or cp['best']['epoch'] != 6:
        raise ValueError('Wrong checkpoint epoch')
    del optimizer
    model.eval()
    data = LazyStationWindows(root, 'test')
    index = data.index.sort_values(['station_id', 'origin_utc'])
    matrix = np.lib.format.open_memmap(out/'causal-features.npy', mode='w+', dtype='float32',
                                    shape=(plan['contract']['targets'], len(FEATURES)))
    writer = None; offset = 0; digest = hashlib.sha256()
    left, right = map(utc, SPLITS['test'])
    try:
        with torch.inference_mode():
            for station, group in index.groupby('station_id', sort=True):
                progress('TFT_station_'+station)
                begin = group.origin_utc.min()-pd.Timedelta(hours=336)
                end = group.origin_utc.max()+pd.Timedelta(hours=24)
                frame = pd.read_parquet(data.paths, filters=[('station_id', '==', station),
                    ('hour_start_utc', '>=', begin), ('hour_start_utc', '<', end)]).sort_values('hour_start_utc').reset_index(drop=True)
                if not pd.DatetimeIndex(frame.hour_start_utc).equals(pd.date_range(begin, end, freq='h', inclusive='left')):
                    raise ValueError('Missing/duplicate UTC panel hours')
                counts = frame.ride_count.to_numpy(dtype=float, na_value=np.nan)
                mask = frame.label_observed_mask.to_numpy(dtype=bool)
                from citypulse_ai.features.station_dataset import KNOWN_FEATURES
                calendar = frame[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=np.float32)
                status = frame.station_available.to_numpy(dtype=float, na_value=np.nan)
                published = pd.to_datetime(frame.status_available_at_utc, utc=True)
                base = frame.availability_known_mask.to_numpy(dtype=bool)
                rows = list(group.itertuples(index=False))
                for start in range(0, len(rows), 32):
                    windows = []; baseline = []
                    for row in rows[start:start+32]:
                        origin = utc(row.origin_utc); pos = int((origin-begin)/pd.Timedelta(hours=1))
                        w = make_window(frame, station, origin, split_start=left, split_end=right,
                                        scalers=data.scalers, split_role='evaluation')
                        if not w.input_eligible or not w.target_mask.all():
                            raise ValueError('Common target/input masks changed')
                        if not mask[pos-168:pos-144].all() or not mask[pos-24:pos].all():
                            raise ValueError('Missing paired lag observations')
                        statuses = base & published.notna().to_numpy() & published.le(origin).to_numpy()
                        matrix[offset:offset+24] = feature_block(counts, mask, calendar, status, statuses, pos, data.codes[station])
                        offset += 24
                        windows.append(w)
                        baseline.append((counts[pos-24:pos], counts[pos-168:pos-144]))
                    batch = windows_to_batch(windows, data.scalers)
                    original = inverse_quantiles(model(*batch.inputs()), batch.means, batch.scales).numpy()
                    if not np.isfinite(original).all() or (np.diff(original, axis=-1) < 0).any():
                        raise ValueError('Invalid quantile outputs')
                    frames = []
                    for j, w in enumerate(windows):
                        for lead, hour in enumerate(w.target_hours_utc):
                            digest.update(f'{station}|{w.origin_utc.isoformat()}|{hour.isoformat()}|{lead+1}\n'.encode())
                        frames.append(pd.DataFrame({'station_id': station, 'origin_utc': w.origin_utc,
                            'target_hour_utc': w.target_hours_utc, 'horizon_step': np.arange(1, 25),
                            'actual': w.target[:, 0], 'trusted': w.target_mask[:, 0],
                            'tft_q10_raw': original[j, :, 0], 'tft_q50_raw': original[j, :, 1],
                            'tft_q90_raw': original[j, :, 2], 'lag24': baseline[j][0], 'lag168': baseline[j][1]}))
                    table = pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out/'tft-predictions.parquet', table.schema, compression='zstd')
                    writer.write_table(table)
    finally:
        matrix.flush()
        if writer is not None:
            writer.close()
    if offset != plan['contract']['targets'] or digest.hexdigest() != plan['contract']['paired_key_sha256']:
        raise ValueError('Output population differs from frozen keys')
    save(out/'tft-complete.json', {'targets': offset, 'paired_key_sha256': digest.hexdigest(),
        'features_sha256': fingerprint(out/'causal-features.npy'), 'predictions_sha256': fingerprint(out/'tft-predictions.parquet')})


def quantile_metrics(actual, quantiles):
    actual = np.asarray(actual, dtype=float); quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.shape != (len(actual), 3) or not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError('Invalid quantiles')
    residual = actual[:, None]-quantiles; q = np.array([.1, .5, .9])
    pinball = np.maximum(q*residual, (q-1)*residual).mean(axis=0)
    return {**paired_metrics(actual, quantiles[:, 1]),
            'pinball_by_quantile': dict(zip(['q10', 'q50', 'q90'], pinball.tolist())),
            'mean_pinball': float(pinball.mean()),
            'interval_coverage': float(((quantiles[:, 0] <= actual) & (actual <= quantiles[:, 2])).mean()),
            'nominal_interval_coverage': .8, 'negative_median_fraction': float((quantiles[:, 1] < 0).mean())}


def xgb_phase(model_path, preprocessing, out):
    """Single selected-XGBoost prediction call on the already frozen shared rows."""
    import xgboost as xgb
    out = Path(out); plan = json.loads((out/'evaluation-plan.json').read_text())
    completed = json.loads((out/'tft-complete.json').read_text())
    if (fingerprint(Path(model_path)) != plan['XGBoost_sha256']
            or fingerprint(Path(preprocessing)) != plan['preprocessing_sha256']
            or fingerprint(out/'causal-features.npy') != completed['features_sha256']
            or fingerprint(out/'tft-predictions.parquet') != completed['predictions_sha256']):
        raise ValueError('Evaluation/model inputs changed')
    if xgb.__version__ != '3.2.0':
        raise ValueError('Use the recorded native XGBoost 3.2.0 environment')
    save(out/'xgb-started.json', {'model_sha256': plan['XGBoost_sha256']})
    model = xgb.Booster(); model.load_model(model_path); model.set_param({'nthread': 2, 'device': 'cpu'})
    if model.feature_names != list(FEATURES) or model.feature_types != list(FEATURE_TYPES) or model.num_boosted_rounds() != 300:
        raise ValueError('XGBoost saved schema/round count changed')
    values = np.load(out/'causal-features.npy', mmap_mode='r')
    prediction = model.inplace_predict(values)
    frame = pd.read_parquet(out/'tft-predictions.parquet')
    if len(frame) != len(prediction) or not frame.trusted.all() or frame.duplicated(['station_id', 'origin_utc', 'target_hour_utc']).any():
        raise ValueError('Invalid common prediction population')
    frame['xgb_raw'] = prediction; frame['xgb_clipped'] = np.maximum(prediction, 0)
    for name in ['q10', 'q50', 'q90']:
        frame['tft_'+name+'_clipped'] = np.maximum(frame['tft_'+name+'_raw'], 0)
    raw = frame[['tft_q10_raw', 'tft_q50_raw', 'tft_q90_raw']].to_numpy()
    clipped = np.maximum(raw, 0)
    metrics = {'TFT_raw': quantile_metrics(frame.actual, raw), 'TFT_clipped': quantile_metrics(frame.actual, clipped),
               'XGBoost_raw': paired_metrics(frame.actual, prediction),
               'XGBoost_clipped': paired_metrics(frame.actual, np.maximum(prediction, 0)),
               'lag24': paired_metrics(frame.actual, frame.lag24), 'lag168': paired_metrics(frame.actual, frame.lag168)}
    frame.to_parquet(out/'test-predictions.parquet', index=False, compression='zstd')
    v = plan['validation']
    comparison = []
    validations = {'TFT_raw': v['TFT'], 'TFT_clipped': v['TFT']['nonnegative_clipped'],
                   'XGBoost_raw': v['XGBoost']['raw'], 'XGBoost_clipped': v['XGBoost']['clipped'],
                   'lag24': v['lag24'], 'lag168': v['lag168']}
    for name, scores in metrics.items():
        for field in ['MAE', 'RMSE', 'WAPE_percent']:
            before, after = validations[name][field], scores[field]
            comparison.append({'model': name, 'metric': field, 'validation': before, 'test': after,
                               'test_minus_validation': after-before, 'validation_targets': 220800,
                               'test_targets': len(frame)})
    pd.DataFrame(comparison).to_csv(out/'validation-versus-test.csv', index=False)
    report = {'status': 'final_held_out_test_complete', 'plan_sha256': fingerprint(out/'evaluation-plan.json'),
              'paired_key_sha256': plan['contract']['paired_key_sha256'], 'metrics': metrics,
              'test_targets': len(frame), 'forecast_windows': plan['contract']['forecast_windows'],
              'unscored_station_hours': plan['contract']['unscored_station_hours'],
              'test_evaluated': True, 'training_or_selection_performed': False,
              'prediction_sha256': fingerprint(out/'test-predictions.parquet'), 'xgboost_version': xgb.__version__}
    save(out/'experiment.json', report)
    return report
