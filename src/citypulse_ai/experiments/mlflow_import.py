"""Import existing JSON metadata only; no training, model imports or artifact copies."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source paths must stay within the project')
    return path


def read_json(path):
    if path.stat().st_size > 100_000:
        raise ValueError('Only small metadata JSON records may be imported')
    return json.loads(path.read_text())


def text(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')) if isinstance(value, (dict, list, bool)) else str(value)


def build_plans(root, configuration):
    root = Path(root).resolve()
    sources = {p: digest(local_path(root, p)) for p in configuration['contract_sources']}
    plans = []
    for item in configuration['records']:
        source = local_path(root, item['path'])
        record = read_json(source)
        record_sha = digest(source)
        manifest = read_json(local_path(root, item['manifest'])) if item.get('manifest') else None
        if manifest and manifest['original_experiment_sha256'] != record_sha:
            raise ValueError('Recovery manifest does not identify the original experiment')
        for model in item['models']:
            params = dict(configuration['evaluation_contract'])
            metrics = {}
            for split in ('validation', 'test'):
                rows = [r for r in record['metrics'] if r['model'] == model and r['split'] == split]
                if len(rows) != 1:
                    raise ValueError(f'Expected exactly one {model}/{split} metric row')
                for name in ('MAE', 'RMSE', 'WAPE_percent'):
                    value = rows[0][name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError('Metrics must be finite nonnegative saved numbers')
                    metrics[f'{split}_{name}'] = value
                for name in ('forecast_origins', 'target_hours'):
                    params[f'{split}_{name}'] = rows[0][name]
            tags = {'model_name': model, 'stage': 'preliminary', 'training_code_commit': 'not recorded',
                    'source_record': item['path'], 'source_record_sha256': record_sha,
                    'data_provenance': text(record.get('provenance', {})),
                    'contract_metadata_sources': text(sources),
                    'period_metadata_origin': 'inspected implementation/docs; not original JSON split fields'}
            checkpoint = None
            if model.startswith('Lag '):
                tags['framework'] = 'seasonal-naive'
                params.update(lag_elapsed_hours=int(model.split()[1]), training='no fitting',
                              input_configuration='actual demand at target minus lag')
            elif model == 'XGBoost':
                tags['framework'] = 'XGBoost'
                params.update(record['selected_parameters'])
                if not manifest:
                    raise ValueError('XGBoost requires its associated recovery manifest')
                if manifest['hyperparameters'] != record['selected_parameters']:
                    raise ValueError('Recovered hyperparameters differ from original record')
                params.update(feature_order=manifest['feature_order'], preprocessing=manifest['preprocessing'])
                tags['recovery_method'] = manifest['method']
                tags['manifest_sha256'] = digest(local_path(root, item['manifest']))
                checkpoint = local_path(root, manifest['artifact_location'])
                if digest(checkpoint) != manifest['artifact_sha256']:
                    raise ValueError('XGBoost checkpoint checksum differs from manifest')
            elif model == 'PyTorch MLP':
                tags['framework'] = 'PyTorch'
                params.update(record['settings'], architecture=record['architecture'])
                meta_path = local_path(root, item['checkpoint_metadata'])
                meta = read_json(meta_path)
                checkpoint = local_path(root, record['checkpoint_relative_path'])
                if digest(checkpoint) != meta['checkpoint_sha256']:
                    raise ValueError('MLP checkpoint metadata checksum differs')
                params.update(feature_order=meta['features'], training_first_utc=meta['training_first_utc'],
                              training_last_utc=meta['training_last_utc'])
                tags['checkpoint_metadata_sha256'] = digest(meta_path)
            elif model == 'Keras LSTM':
                tags['framework'] = 'TensorFlow/Keras'
                params.update(record['settings'], architecture=item['documented_architecture'],
                              input_shape=[record['settings']['lookback'], 1])
                tags['architecture_metadata_source'] = item['architecture_source']
                tags['architecture_source_sha256'] = digest(local_path(root, item['architecture_source']))
                checkpoint = local_path(root, record['checkpoint_relative_path'])
            else:
                raise ValueError(f'Unsupported existing model: {model}')
            if checkpoint:
                params['checkpoint_reference'] = str(checkpoint.relative_to(root))
                tags['checkpoint_sha256'] = digest(checkpoint)
            else:
                tags['checkpoint_status'] = 'not applicable; seasonal-naive'
            for name in ('training_rows', 'dropped_training_rows', 'training_sequences', 'validation_sequences',
                         'test_sequences', 'parameters', 'epochs_run', 'best_epoch', 'scaler_training_hours'):
                if name in record and not model.startswith('Lag '):
                    params[name] = record[name]
            tags['recorded_versions'] = text(record.get('versions', {}))
            for name in ('test_previously_inspected', 'development_only', 'exploratory_development_experiment'):
                if name in record:
                    tags[name] = text(record[name])
            params = {key: text(value) for key, value in params.items()}
            key = hashlib.sha256(f'{record_sha}:{model}'.encode()).hexdigest()
            plan_sha = hashlib.sha256(text({'params': params, 'metrics': metrics, 'tags': tags}).encode()).hexdigest()
            plans.append({'key': key, 'plan_sha': plan_sha, 'model': model, 'params': params, 'metrics': metrics, 'tags': tags})
    if len({p['key'] for p in plans}) != len(plans):
        raise ValueError('Configuration lists a source/model twice')
    return plans


@contextmanager
def import_lock(store):
    store.mkdir(parents=True, exist_ok=True)
    with (store/'import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def import_plans(store, experiment_name, plans, import_commit='not available'):
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    from mlflow.tracking import MlflowClient
    store = Path(store).resolve()
    with import_lock(store):
        client = MlflowClient(tracking_uri=f'sqlite:///{store / "tracking.sqlite"}')
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(
            experiment_name, artifact_location=(store/'artifacts').as_uri())
        results = []
        for plan in plans:
            found = client.search_runs([experiment_id], f"tags.citypulse_import_key = '{plan['key']}'", max_results=2)
            if len(found) > 1:
                raise ValueError('Duplicate existing import keys; refusing to change runs')
            created = not found
            run = found[0] if found else client.create_run(experiment_id, tags={
                **plan['tags'], 'mlflow.runName': plan['model'], 'citypulse_import_key': plan['key'],
                'citypulse_plan_sha256': plan['plan_sha'], 'import_code_commit': import_commit,
                'import_timestamp_ms': str(int(time.time()*1000)), 'import_state': 'incomplete'})
            if run.data.tags['citypulse_plan_sha256'] != plan['plan_sha']:
                raise ValueError('Metadata changed for an immutable source/model; refusing overwrite')
            if any(run.data.tags.get(name) != value for name,value in plan['tags'].items()):
                raise ValueError('Existing provenance tags differ')
            run_id = run.info.run_id
            for name,value in plan['params'].items():
                if name in run.data.params and run.data.params[name] != value:
                    raise ValueError('Existing run parameter differs')
                if name not in run.data.params:
                    client.log_param(run_id, name, value)
            for name,value in plan['metrics'].items():
                history = client.get_metric_history(run_id, name)
                if history and (len(history) != 1 or history[0].value != value):
                    raise ValueError('Existing metric history differs; refusing overwrite')
                if not history:
                    client.log_metric(run_id, name, value, timestamp=int(run.data.tags['import_timestamp_ms']), step=0)
            client.set_tag(run_id, 'import_state', 'complete')
            client.set_terminated(run_id, status='FINISHED')
            saved = client.get_run(run_id)
            if saved.data.params != plan['params'] or saved.data.metrics != plan['metrics']:
                raise ValueError('Stored metadata verification failed')
            if client.list_artifacts(run_id):
                raise ValueError('Metadata-only run unexpectedly contains copied artifacts')
            results.append({'model': plan['model'], 'run_id': run_id, 'created': created})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/mlflow_existing_experiments.json')
    parser.add_argument('--prepare-mlp-metadata', action='store_true',
                        help='Run in the existing PyTorch environment; extract small metadata only')
    args = parser.parse_args()
    configuration = read_json(args.config)
    if args.prepare_mlp_metadata:
        import torch
        for item in configuration['records']:
            if 'checkpoint_metadata' not in item:
                continue
            record = read_json(local_path(ROOT, item['path']))
            checkpoint = local_path(ROOT, record['checkpoint_relative_path'])
            saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
            metadata = {key: saved[key] for key in ('features', 'training_first_utc', 'training_last_utc',
                        'timezone', 'horizon_elapsed_hours', 'settings')}
            metadata['checkpoint_sha256'] = digest(checkpoint)
            destination = local_path(ROOT, item['checkpoint_metadata'])
            if destination.exists():
                if read_json(destination) != metadata:
                    raise ValueError('Existing checkpoint metadata differs; refusing overwrite')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('x') as stream:
                    stream.write(json.dumps(metadata, indent=2)+'\n')
            print(f'Verified small checkpoint metadata: {destination}')
        return
    plans = build_plans(ROOT, configuration)  # Validate all sources before store mutation.
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    results = import_plans(ROOT/'outputs/mlflow', configuration['experiment_name'], plans, commit)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
