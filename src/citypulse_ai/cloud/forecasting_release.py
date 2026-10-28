"""Small pinned serving artifacts and a bounded validation demo; no compute startup."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import pandas as pd
import numpy as np

from citypulse_ai.api.station_catalog import catalog
from citypulse_ai.cloud.gcs import GCSStorage


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    return digest.hexdigest()


def download(record, destination, store):
    destination = Path(destination)
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size != record['size_bytes'] or sha(destination) != record['sha256']:
            raise ValueError('Existing local artifact differs; never overwrite it')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    key = record['cloud_path'].split('/', 3)[3]
    blob = store.bucket.blob(key, generation=int(record['generation']))
    blob.reload(timeout=60, retry=None)
    if blob.size != record['size_bytes'] or blob.metadata.get('sha256') != record['sha256']:
        raise ValueError('Pinned GCS artifact identity differs')
    store.download_file(key, destination, generation=int(record['generation']), expected_sha256=record['sha256'])


def zip_member(record, member, destination, store):
    """Seek only to one verified member; never download the complete dataset ZIP."""
    key = record['cloud_path'].split('/', 3)[3]
    blob = store.bucket.blob(key, generation=int(record['generation'])); blob.reload(timeout=60, retry=None)
    if blob.size != record['size_bytes'] or blob.metadata.get('sha256') != record['sha256']:
        raise ValueError('Pinned dataset recovery identity differs')
    if member['size_bytes'] > 32*1024**2:
        raise ValueError('Development member exceeds the bounded demo budget')
    with blob.open('rb', chunk_size=1024**2, retry=None, timeout=60) as remote:
        with zipfile.ZipFile(remote) as archive:
            info = archive.getinfo(member['name'])
            if info.file_size != member['size_bytes']:
                raise ValueError('Frozen member size differs')
            with archive.open(info) as source, Path(destination).open('xb') as target:
                shutil.copyfileobj(source, target, length=1024**2)
    if sha(destination) != member['sha256']:
        raise ValueError('Extracted member SHA-256 differs')


def provision(root, *, demo=False, diagnostics=False, store=None):
    root = Path(root); spec = catalog()
    store = store or GCSStorage('citypulse-ai-511315', 'citypulse-ai-511315-data')
    for name, record in spec['artifacts'].items(): download(record, root/name, store)
    root.mkdir(parents=True, exist_ok=True)
    (root/'provision-receipt.json').write_text(json.dumps({'artifacts': spec['artifacts'],
        'dataset_manifest_sha256': spec['dataset_manifest_sha256'], 'weights_or_preprocessing_fitted': False}, indent=2)+'\n')
    if diagnostics:
        download(spec['diagnostics'], root.parent/'saved-test-predictions.parquet', store)
    if demo:
        development = spec['development']; origin = pd.Timestamp(development['origin']).tz_convert('UTC')
        station = development['station_id']; start = origin-pd.Timedelta(hours=336)
        # Only this newly owned, reproducible scratch directory is cleaned up.
        with tempfile.TemporaryDirectory(prefix='citypulse-v1-demo-') as scratch:
            folder = Path(scratch); panel = folder/'validation-panel.parquet'
            zip_member(development['dataset_recovery'], development['validation_panel'], panel, store)
            original = pd.read_parquet(panel, filters=[('station_id', '==', station),
                ('hour_start_utc', '>=', start), ('hour_start_utc', '<', origin+pd.Timedelta(hours=24))]).sort_values('hour_start_utc')
            frame = original.loc[original.hour_start_utc.lt(origin)]
            if len(frame) != 336 or not frame.label_observed_mask.all():
                raise ValueError('Development history is not fully verified')
            history = [{'timestamp': r.hour_start_utc.isoformat(), 'ride_count': int(r.ride_count)} for r in frame.itertuples()]
            from citypulse_ai.features.final_station_dataset import load_preprocessing
            from citypulse_ai.features.station_dataset import make_window
            scalers, _ = load_preprocessing(root/'preprocessing.json')
            window = make_window(original, station, origin, split_start='2025-07-01T00:00:00-04:00',
                                 split_end='2025-10-01T00:00:00-04:00', scalers=scalers)
            np.savez_compressed(root.parent/'demo-original-window.npz', station=window.static_station_code[None],
                encoder_known=window.encoder_known[None], encoder_observed=window.encoder_observed[None],
                decoder_known=window.decoder_known[None], mean=scalers[station].mean, scale=scalers[station].scale)
            request = {'station_id': station, 'origin': development['origin'], 'history': history}
            expected = {'lag-24': frame.ride_count.iloc[-24:].astype(float).tolist(),
                        'lag-168': frame.ride_count.iloc[168:192].astype(float).tolist()}
            for model, record in development['validation_references'].items():
                path = folder/(model+'.parquet'); download(record, path, store)
                reference = pd.read_parquet(path, filters=[('station_id', '==', station), ('origin_utc', '==', origin)]).sort_values('horizon_step')
                if len(reference) != 24 or reference.horizon_step.tolist() != list(range(1, 25)):
                    raise ValueError('Saved validation reference is incomplete')
                expected[model] = reference.prediction_clipped.astype(float).tolist()
            (root.parent/'demo-request.json').write_text(json.dumps(request, indent=2)+'\n')
            (root.parent/'demo-references.json').write_text(json.dumps({'expected': expected,
                'source': 'frozen original validation forecasts and certified historical lag counts',
                'TFT_saved_validation_reference_available': False, 'development': development,
                'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}, indent=2)+'\n')
    return {'artifact_bytes': sum(r['size_bytes'] for r in spec['artifacts'].values()),
            'artifacts': len(spec['artifacts']), 'demo_created': demo, 'diagnostics_restored': diagnostics,
            'full_dataset_downloaded': False}
