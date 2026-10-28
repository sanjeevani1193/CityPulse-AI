"""Descriptive post-hoc diagnostics from saved forecasts; never model inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from citypulse_ai.api.station_catalog import catalog
from citypulse_ai.evaluation.station_followup import keys_and_labels
from citypulse_ai.evaluation.station_benchmark import quantile_metrics
from citypulse_ai.features.final_station_dataset import load_preprocessing, SPLITS
from citypulse_ai.evaluation.station_benchmark import calendar_contract
from citypulse_ai.forecasting.station_xgboost import paired_metrics

COLUMNS = {'XGBoost': 'xgb_clipped', 'MLP': 'mlp_clipped', 'TFT': 'tft_q50_clipped',
           'LSTM': 'lstm_clipped', 'Lag-24': 'lag24', 'Lag-168': 'lag168'}
IDS = dict(zip(COLUMNS, ['xgboost', 'pytorch-mlp', 'pytorch-tft', 'tensorflow-lstm', 'lag-24', 'lag-168']))


def activity_groups(scalers):
    """Fixed training-mean rank thirds: lowest34, middle33, highest33; ID ties."""
    stations = sorted(scalers, key=lambda s: (scalers[s].mean, s))
    if len(stations) != 100: raise ValueError('Require frozen 100-station scalers')
    return {s: 'lower' if i < 34 else 'middle' if i < 67 else 'higher' for i, s in enumerate(stations)}


def tables(frame, scalers):
    groups = activity_groups(scalers); local = frame.target_hour_utc.dt.tz_convert('America/New_York')
    keys = pd.DataFrame({'station_id': frame.station_id, 'hour': local.dt.hour,
        'day_type': np.where(local.dt.dayofweek < 5, 'weekday', 'weekend'),
        'month': local.dt.strftime('%Y-%m'), 'activity': frame.station_id.map(groups),
        'period': np.where((local.dt.dayofweek < 5) & local.dt.hour.isin([7, 8, 9, 16, 17, 18]),
                           'weekday_commute_peak', 'off_peak')})
    outputs = {key: [] for key in ['overall', 'station_id', 'hour', 'day_type', 'month', 'activity', 'period', 'station_distribution']}
    for name, column in COLUMNS.items():
        prediction = frame[column].to_numpy(dtype=float); actual = frame.actual.to_numpy(dtype=float)
        if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError('Invalid fixed clipped forecasts')
        signed = prediction-actual; data = keys.assign(absolute_error=np.abs(signed), signed_error=signed)
        outputs['overall'].append({'model': name, **paired_metrics(actual, prediction), 'mean_signed_error': float(signed.mean())})
        for key in ['station_id', 'hour', 'day_type', 'month', 'activity', 'period']:
            result = data.groupby(key, sort=True).agg(MAE=('absolute_error', 'mean'),
                mean_signed_error=('signed_error', 'mean'), targets=('absolute_error', 'size')).reset_index()
            result.insert(0, 'model', name); outputs[key].append(result)
        station_mae = data.groupby('station_id').absolute_error.mean()
        outputs['station_distribution'].append({'model': name, 'stations': len(station_mae),
            'minimum': station_mae.min(), 'p10': station_mae.quantile(.1), 'median': station_mae.median(),
            'p90': station_mae.quantile(.9), 'maximum': station_mae.max(), 'mean': station_mae.mean()})
    return {k: pd.DataFrame(v) if k in ('overall', 'station_distribution') else pd.concat(v, ignore_index=True)
            for k, v in outputs.items()}


def analyze(predictions, preprocessing, out, figure_dir):
    spec = catalog(); source = Path(predictions)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != spec['diagnostics']['sha256']: raise ValueError('Saved prediction checksum differs')
    frame = pd.read_parquet(source); scalers, codes = load_preprocessing(preprocessing)
    if hashlib.sha256(Path(preprocessing).read_bytes()).hexdigest() != spec['artifacts']['preprocessing.json']['sha256']:
        raise ValueError('Training activity/preprocessing identity differs')
    contract = calendar_contract(*SPLITS['test'], codes); keys_and_labels(frame, contract)
    results = tables(frame, scalers)
    for row in results['overall'].to_dict('records'):
        expected = spec['benchmark']['metrics'][IDS[row['model']]]
        for metric in ['MAE', 'RMSE', 'WAPE_percent']:
            if not np.isclose(row[metric], expected[metric], rtol=0, atol=1e-12):
                raise ValueError('Original benchmark metric changed')
    uncertainty = quantile_metrics(frame.actual, frame[['tft_q10_clipped', 'tft_q50_clipped', 'tft_q90_clipped']].to_numpy())
    frozen = spec['benchmark']['TFT_uncertainty']
    if not np.isclose(uncertainty['interval_coverage'], frozen['interval_coverage'], atol=1e-12, rtol=0):
        raise ValueError('Saved TFT coverage changed')
    for q, value in frozen['pinball_by_quantile'].items():
        if not np.isclose(uncertainty['pinball_by_quantile'][q], value, atol=1e-12, rtol=0):
            raise ValueError('Saved TFT pinball changed')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    for name, table in results.items(): table.to_csv(out/(name+'.csv'), index=False)
    validation = spec['benchmark']['validation_metrics']['xgboost']
    summary = {'source_sha256': digest, 'paired_key_sha256': contract['paired_key_sha256'], 'targets': len(frame),
        'grouping': 'training CountScaler mean rank ascending, station-ID ties: lower34/middle33/higher33',
        'commute_peak': 'weekday target hours07–09 and16–18 NYC; off_peak includes all remaining scored hours',
        'test_actual_mean': float(frame.actual.to_numpy(dtype=float).mean()),
        'validation_actual_mean_derived_from_frozen_metrics': 100*validation['MAE']/validation['WAPE_percent'],
        'TFT_uncertainty_unchanged': frozen, 'descriptive_posthoc_only': True,
        'training_or_model_inference_performed': False, 'original_metrics_unchanged': True}
    (out/'provenance.json').write_text(json.dumps(summary, indent=2)+'\n')
    plots(results, figure_dir)
    return summary, results


def plots(results, destination):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/forecasting-v1/matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': 'citypulse-v1', 'font.size': 10})
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    def finish(fig, name):
        fig.tight_layout(); fig.savefig(destination/name, metadata={'Date': None}); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5)); table = results['overall']
    ax.bar(table.model, table.mean_signed_error); ax.axhline(0, color='black', linewidth=.8)
    ax.set_ylabel('Mean prediction − actual (starts/hour)'); ax.set_title('Forecast bias — fixed held-out forecasts')
    finish(fig, 'bias.svg')
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, group in results['hour'].groupby('model', sort=False): ax.plot(group.hour, group.MAE, label=model)
    ax.set(xlabel='NYC-local target hour', ylabel='MAE (starts/hour)', xticks=range(0,24,2), title='Error by local hour')
    ax.legend(ncol=3); finish(fig, 'hourly-mae.svg')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([results['station_id'].loc[results['station_id'].model.eq(m), 'MAE'] for m in COLUMNS], tick_labels=list(COLUMNS))
    ax.set(ylabel='Station MAE (starts/hour)', title='Distribution across the same 100 stations')
    finish(fig, 'station-mae.svg')
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, group in results['month'].groupby('model', sort=False): ax.plot(group.month, group.MAE, marker='o', label=model)
    ax.set(ylabel='MAE (starts/hour)', title='Monthly error — identical scored targets'); ax.legend(ncol=3)
    finish(fig, 'monthly-mae.svg')
