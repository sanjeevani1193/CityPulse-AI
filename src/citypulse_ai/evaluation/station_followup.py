"""Once-only MLP/LSTM test extension after the initial test results were disclosed.

Frameworks load in separate CPU processes. Training entry points stay closed to
test data. Existing XGBoost/TFT/baseline results are read, never inferred again.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.evaluation.station_benchmark import DATASET_SHA, calendar_contract, save
from citypulse_ai.features.final_station_dataset import LazyStationWindows, SPLITS
from citypulse_ai.features.station_dataset import KNOWN_FEATURES, utc
from citypulse_ai.forecasting.station_xgboost import FEATURES, feature_block, paired_metrics

FROZEN = {
    'mlp_checkpoint': '8bf098d0f3eca06f3fb569931433233c6fb221aff48017ff8d34b5acf414f63f',
    'mlp_preprocessing': 'a3c046a057d41895c7750d98f166e9d31ce533d461381e813cda377d4dde33c4',
    'mlp_report': '570a6133eee5e010a44a2a80326eb2db43c447603b19b0729193cb830e01be8d',
    'lstm_checkpoint': '81dde6d7faf480e7e870fb2234a0208828f1f6cb857b95a18c589913d4c531ec',
    'lstm_preprocessing': 'f788108df037662b5bbc0739f6f22f105e798e2839bed1e67100cf570b0467fe',
    'lstm_report': 'c2eeee5bdb521297caffe9ec8c49e5b0763dca4b68e96d4ae9043aa9d202b670',
    'prior_report': '33c179392f60ed2253accb588425faca893f54b956e7253f2ea1a7d7a876259b',
    'prior_plan': 'fb5a5aba7e658661fa1750ef040771bd62d40b834d48e3999b81b661e0a3f4f9',
    'prior_predictions': 'ccbd2ca6989f85437078bed249a36507c60dfc627d9e2a0139d424e42902cc30',
}
KEYS = ['station_id', 'origin_utc', 'target_hour_utc', 'horizon_step']


class MetadataWindows(LazyStationWindows):
    """Preflight verifies index/scalers without decoding any target panel."""
    def verify(self):
        for r in self.manifest['artifacts']:
            if r['name'] in ('preprocessing.json', 'windows.parquet'):
                p = self.root/r['name']
                if p.is_symlink() or p.stat().st_size != r['size_bytes'] or fingerprint(p) != r['sha256']:
                    raise ValueError('Frozen metadata differs')


def freeze(root, artifacts, out):
    """Freeze identities, configurations and calendar keys before target reads."""
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Existing follow-up; inspect saved state instead of retrying')
    if fingerprint(root/'dataset.json') != DATASET_SHA or set(artifacts) != set(FROZEN):
        raise ValueError('Frozen dataset/artifact contract differs')
    for name, expected in FROZEN.items():
        if fingerprint(Path(artifacts[name])) != expected:
            raise ValueError('Frozen artifact differs: '+name)
    reports = {m: json.loads(Path(artifacts[m+'_report']).read_text()) for m in ('mlp', 'lstm')}
    data = MetadataWindows(root, 'test')
    if data.manifest['splits'] != SPLITS or len(data.codes) != 100:
        raise ValueError('Frozen splits/stations differ')
    cohort = data.manifest['cohort']['metadata']
    a, b = map(utc, SPLITS['train'])
    if utc(cohort['train_start_utc']) != a or utc(cohort['train_end_utc_exclusive']) != b:
        raise ValueError('Station selection is not training-only')
    if any(s.train_start_utc != a or s.train_end_utc != b for s in data.scalers.values()):
        raise ValueError('Scaler training period differs')
    for m, epoch in [('mlp', 10), ('lstm', 18)]:
        r = reports[m]
        selected = min(r['history'], key=lambda h: h['validation']['clipped']['MAE'] if m == 'mlp'
                       else h['validation']['mse_scaled'])['epoch']
        if (r['test_evaluated'] or not r['training_complete'] or not r['fresh_initialization']
                or r['dataset_manifest_sha256'] != DATASET_SHA or r['splits'] != SPLITS
                or r['best']['epoch'] != epoch or selected != epoch
                or r['checkpoint_sha256'] != FROZEN[m+'_checkpoint']):
            raise ValueError('Frozen selection/development identity differs: '+m)
        source = Path(__file__).parents[1]/'forecasting'/('station_'+m+'.py')
        if fingerprint(source) != r['source_sha256']:
            raise ValueError('Training adapter changed: '+m)
    prep = json.loads(Path(artifacts['mlp_preprocessing']).read_text())
    if (prep['station_codes'] != data.codes or prep['feature_names'] != list(FEATURES)
            or prep['training_period'] != SPLITS['train'] or prep['training_rows'] != 1279200
            or prep['dataset_manifest_sha256'] != DATASET_SHA):
        raise ValueError('MLP preprocessing differs')
    if fingerprint(root/'preprocessing.json') != FROZEN['lstm_preprocessing']:
        raise ValueError('LSTM preprocessing differs')
    contract = calendar_contract(*SPLITS['test'], data.codes)
    prior = json.loads(Path(artifacts['prior_report']).read_text())
    prior_plan = json.loads(Path(artifacts['prior_plan']).read_text())
    expected = [(s, utc(o)) for s in contract['station_ids'] for o in contract['origins']]
    index = data.index.sort_values(['station_id', 'origin_utc'])
    if (list(zip(index.station_id, index.origin_utc)) != expected
            or not index.primary_evaluation_eligible.all() or not index.input_eligible.all()
            or not index.trusted_targets.eq(24).all()
            or contract['targets'] != 220800 or prior['test_targets'] != 220800
            or prior['paired_key_sha256'] != contract['paired_key_sha256']
            or prior['prediction_sha256'] != FROZEN['prior_predictions']
            or prior_plan['contract'] != contract):
        raise ValueError('Existing/common test keys differ')
    plan = {'status': 'frozen_before_followup_test_reads', 'dataset_manifest_sha256': DATASET_SHA,
            'dataset_identity': data.manifest['identity'], 'inputs': dict(FROZEN), 'contract': contract,
            'encoder_hours': 336, 'horizon_hours': 24, 'splits': SPLITS,
            'point_rule': 'max(0, raw original-unit point forecast)',
            'followup_after_prior_test_disclosure': True, 'fully_blind_six_model_experiment': False,
            'selection_or_training_allowed': False,
            'validation': {m.upper(): r['validation'] for m, r in reports.items()},
            'original_validation': prior_plan['validation'],
            'configurations': {m: r['config'] for m, r in reports.items()},
            'source_sha256': fingerprint(Path(__file__))}
    out.mkdir(parents=True)
    save(out/'evaluation-plan.json', plan)
    return plan


def keys_and_labels(frame, contract):
    if (len(frame) != contract['targets'] or frame.duplicated(KEYS).any()
            or not np.isfinite(frame.actual).all() or (frame.actual < 0).any()):
        raise ValueError('Invalid common keys/labels')
    digest = hashlib.sha256()
    for r in frame.itertuples(index=False):
        origin, hour = utc(r.origin_utc), utc(r.target_hour_utc)
        if not 1 <= r.horizon_step <= 24 or hour != origin+pd.Timedelta(hours=r.horizon_step-1):
            raise ValueError('Invalid elapsed-hour target')
        digest.update(f'{r.station_id}|{origin.isoformat()}|{hour.isoformat()}|{r.horizon_step}\n'.encode())
    if digest.hexdigest() != contract['paired_key_sha256']:
        raise ValueError('Common key digest differs')
    return digest.hexdigest()


def feature_rows(frame, station, code, origins):
    """Reuse the original 48 causal features, one station in memory."""
    frame = frame.sort_values('hour_start_utc').reset_index(drop=True)
    hours = pd.DatetimeIndex(frame.hour_start_utc)
    if (not hours.equals(pd.date_range(hours[0], hours[-1], freq='h'))
            or not frame.station_id.eq(station).all() or not frame.station_code.eq(code).all()):
        raise ValueError('Station/UTC grid differs')
    counts = frame.ride_count.to_numpy(dtype=float, na_value=np.nan)
    mask = frame.label_observed_mask.to_numpy(dtype=bool)
    calendar = frame[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=np.float32)
    status = frame.station_available.to_numpy(dtype=float, na_value=np.nan)
    base = frame.availability_known_mask.to_numpy(dtype=bool)
    published = pd.to_datetime(frame.status_available_at_utc, utc=True)
    features, keys = [], []
    for value in origins:
        origin = utc(value); p = int((origin-hours[0])/pd.Timedelta(hours=1))
        local = origin.tz_convert('America/New_York')
        if local != local.normalize():
            raise ValueError('NYC midnight required')
        available = base & published.notna().to_numpy() & published.le(origin).to_numpy()
        features.append(feature_block(counts, mask, calendar, status, available, p, code))
        if (not mask[p:p+24].all() or not np.isfinite(counts[p:p+24]).all()
                or (counts[p:p+24] < 0).any()):
            raise ValueError('Missing trusted target')
        keys.append(pd.DataFrame({'station_id': station, 'origin_utc': origin,
            'target_hour_utc': hours[p:p+24], 'horizon_step': np.arange(1, 25), 'actual': counts[p:p+24]}))
    return np.concatenate(features), pd.concat(keys, ignore_index=True)


def _begin(root, artifacts, out, model):
    out = Path(out); plan = json.loads((out/'evaluation-plan.json').read_text())
    if (fingerprint(Path(root)/'dataset.json') != plan['dataset_manifest_sha256']
            or fingerprint(Path(__file__)) != plan['source_sha256']):
        raise ValueError('Frozen evaluator/dataset changed')
    for name in (model+'_checkpoint', model+'_preprocessing', model+'_report'):
        if fingerprint(Path(artifacts[name])) != plan['inputs'][name]:
            raise ValueError('Frozen model input changed')
    save(out/(model+'-started.json'), {'plan_sha256': fingerprint(out/'evaluation-plan.json'),
                                     'checkpoint_sha256': plan['inputs'][model+'_checkpoint']})
    return plan


def _write_prediction(writer, out, model, frame, prediction):
    import pyarrow as pa
    import pyarrow.parquet as pq
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(frame) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError('Invalid frozen predictions')
    frame[model+'_raw'] = prediction; frame[model+'_clipped'] = np.maximum(prediction, 0)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(Path(out)/(model+'-predictions.parquet'), table.schema, compression='zstd')
    writer.write_table(table)
    return writer


def _complete(out, artifacts, model, plan):
    out = Path(out); path = out/(model+'-predictions.parquet'); frame = pd.read_parquet(path)
    digest = keys_and_labels(frame, plan['contract'])
    if fingerprint(Path(artifacts[model+'_checkpoint'])) != plan['inputs'][model+'_checkpoint']:
        raise ValueError('Checkpoint mutated during inference')
    save(out/(model+'-complete.json'), {'targets': len(frame), 'paired_key_sha256': digest,
         'prediction_sha256': fingerprint(path), 'training_or_selection_performed': False})


def mlp_phase(root, artifacts, out, *, progress=lambda s: None):
    import torch
    from citypulse_ai.forecasting.station_mlp import restore, predict
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    plan = _begin(root, artifacts, out, 'mlp')
    cp, model, prep = restore(artifacts['mlp_checkpoint'], artifacts['mlp_preprocessing'])
    if cp['epoch'] != 10:
        raise ValueError('MLP checkpoint epoch differs')
    data = LazyStationWindows(root, 'test'); writer = None
    try:
        for station, group in data.index.sort_values(['station_id', 'origin_utc']).groupby('station_id', sort=True):
            progress('MLP_station_'+station)
            begin = group.origin_utc.min()-pd.Timedelta(hours=336); end = group.origin_utc.max()+pd.Timedelta(hours=24)
            frame = pd.read_parquet(data.paths, filters=[('station_id', '==', station),
                ('hour_start_utc', '>=', begin), ('hour_start_utc', '<', end)])
            x, keys = feature_rows(frame, station, data.codes[station], group.origin_utc)
            writer = _write_prediction(writer, out, 'mlp', keys, predict(model, x, prep))
    finally:
        if writer is not None: writer.close()
    _complete(out, artifacts, 'mlp', plan)


def lstm_phase(root, artifacts, out, *, progress=lambda s: None):
    from tensorflow import keras
    from citypulse_ai.forecasting.station_lstm import configure_cpu, DevelopmentData
    configure_cpu()
    plan = _begin(root, artifacts, out, 'lstm')
    model = keras.models.load_model(artifacts['lstm_checkpoint'])
    if model.count_params() != 8953 or model.output_shape != (None, 24):
        raise ValueError('LSTM architecture differs')
    data = LazyStationWindows(root, 'test')
    data.index = data.index.sort_values(['station_id', 'origin_utc']).reset_index(drop=True)
    writer = None
    try:
        # Reuse the exact development batch implementation without opening a
        # test pathway in its training-only constructor.
        for x, _, mask, meta in DevelopmentData.batches(data, 64):
            progress('LSTM_station_'+meta['station_id'])
            if not mask.all(): raise ValueError('Incomplete common target mask')
            raw = np.asarray(model(x, training=False), dtype=float)*meta['scale']+meta['mean']
            frames = [pd.DataFrame({'station_id': meta['station_id'], 'origin_utc': origin,
                'target_hour_utc': meta['target_hours'][i], 'horizon_step': np.arange(1, 25),
                'actual': meta['actual'][i]}) for i, origin in enumerate(meta['origins'])]
            writer = _write_prediction(writer, out, 'lstm', pd.concat(frames, ignore_index=True), raw)
    finally:
        if writer is not None: writer.close()
    _complete(out, artifacts, 'lstm', plan)


def summarize(artifacts, out):
    """Score saved forecasts only; preserve existing four-method measurements."""
    out = Path(out); plan = json.loads((out/'evaluation-plan.json').read_text())
    for name in ('prior_predictions', 'prior_report', 'prior_plan'):
        if fingerprint(Path(artifacts[name])) != plan['inputs'][name]:
            raise ValueError('Saved benchmark changed')
    frame = pd.read_parquet(artifacts['prior_predictions'])
    keys_and_labels(frame, plan['contract'])
    prior = json.loads(Path(artifacts['prior_report']).read_text())
    scores = dict(prior['metrics'])
    for model in ('mlp', 'lstm'):
        path = out/(model+'-predictions.parquet')
        completed = json.loads((out/(model+'-complete.json')).read_text())
        if fingerprint(path) != completed['prediction_sha256']:
            raise ValueError('Follow-up prediction changed')
        part = pd.read_parquet(path); keys_and_labels(part, plan['contract'])
        if not np.array_equal(frame.actual.to_numpy(), part.actual.to_numpy()):
            raise ValueError('Actuals differ across models')
        for mode in ('raw', 'clipped'):
            column = model+'_'+mode; frame[column] = part[column].to_numpy()
            if mode == 'clipped' and not np.array_equal(frame[column], np.maximum(part[model+'_raw'], 0)):
                raise ValueError('Clipping rule differs')
            scores[model.upper()+'_'+mode] = paired_metrics(frame.actual, frame[column])
    if (out/'test-predictions.parquet').exists(): raise FileExistsError('Combined predictions already saved')
    frame.to_parquet(out/'test-predictions.parquet', index=False, compression='zstd')
    rows = []
    validations = {'MLP_'+k: v for k, v in plan['validation']['MLP'].items() if k in ('raw', 'clipped')}
    validations.update({'LSTM_'+k: v for k, v in plan['validation']['LSTM'].items() if k in ('raw', 'clipped')})
    v = plan['original_validation']
    validations.update(TFT_raw=v['TFT'], TFT_clipped=v['TFT']['nonnegative_clipped'],
        XGBoost_raw=v['XGBoost']['raw'], XGBoost_clipped=v['XGBoost']['clipped'], lag24=v['lag24'], lag168=v['lag168'])
    for method, values in scores.items():
        for metric in ('MAE', 'RMSE', 'WAPE_percent'):
            before, after = validations[method][metric], values[metric]
            rows.append({'method': method, 'metric': metric, 'validation': before, 'test': after,
                         'test_minus_validation': after-before,
                         'validation_targets': validations[method]['targets'], 'test_targets': len(frame)})
    pd.DataFrame(rows).to_csv(out/'validation-versus-test.csv', index=False)
    methods = ['XGBoost_clipped', 'TFT_clipped', 'MLP_clipped', 'LSTM_clipped', 'lag24', 'lag168']
    leaderboard = sorted([{'method': m, **{k: scores[m][k] for k in ('targets', 'MAE', 'RMSE', 'WAPE_percent')}}
                         for m in methods], key=lambda r: r['MAE'])
    pd.DataFrame(leaderboard).to_csv(out/'leaderboard.csv', index=False)
    report = {'status': 'six_method_followup_complete', 'test_targets': len(frame),
        'forecast_windows': plan['contract']['forecast_windows'],
        'paired_key_sha256': plan['contract']['paired_key_sha256'], 'metrics': scores, 'leaderboard': leaderboard,
        'prediction_sha256': fingerprint(out/'test-predictions.parquet'),
        'plan_sha256': fingerprint(out/'evaluation-plan.json'), 'frozen_inputs': plan['inputs'],
        'followup_after_prior_test_disclosure': True, 'fully_blind_six_model_experiment': False,
        'training_or_selection_performed': False, 'existing_four_methods_rerun': False,
        'test_evaluated': True, 'unscored_station_hours': plan['contract']['unscored_station_hours']}
    save(out/'experiment.json', report)
    return report
