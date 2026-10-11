"""Offline validation of trusted backup receipts; never contact GCS or delete files."""

import hashlib
import json
from pathlib import Path
import re

from citypulse_ai.etl.archives import fingerprint


def revision_for(specification: dict) -> str:
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def output_hashes(entry: dict) -> dict:
    return {name: digest for name, digest in sorted(entry['output_sha256'].items())
            if Path(name).suffix in {'.parquet', '.json'} or name == 'rejected_trips.csv'}


def source_files(entry: dict, raw_root: Path) -> dict:
    """Use the retained official ZIP, or exact original CSVs when no ZIP was retained."""
    source = entry['source']
    if source.get('sha256'):
        key = source['key']
        if Path(key).name != key or not key.endswith('.zip'):
            raise ValueError('Unsafe source ZIP name')
        return {key: {'path': str(Path(raw_root) / 'archives' / key),
                      'sha256': source['sha256'], 'role': 'original_zip'}}
    files = {}
    for item in entry.get('csv_files', []):
        path = Path(item.get('path') or item['input_file'])
        digest = item.get('sha256') or item['input_sha256']
        if path.name in files or path.suffix != '.csv':
            raise ValueError('Source CSV names must be unique')
        files[path.name] = {'path': str(path), 'sha256': digest, 'role': 'original_csv'}
    if not files:
        raise ValueError('No original source hashes recorded')
    return files


def checked_receipts(publication: dict, expected: dict, month: str) -> dict:
    """A status string alone is insufficient evidence of a recoverable publication."""
    if publication.get('status') != 'published' or not expected:
        raise ValueError('Complete verified cloud publication is required')
    files = publication.get('files', {})
    if set(files) != set(expected):
        raise ValueError('Missing or unexpected backup artifacts')
    for name, digest in expected.items():
        f = files[name]
        generation = str(f.get('generation', ''))
        prefix = (f"gs://{publication['bucket']}/{publication['prefix']}/"
                  f"source_archive_month={month}/revision={publication['revision']}/")
        if (Path(name).name != name or not re.fullmatch('[0-9a-f]{64}', digest)
                or f.get('sha256') != digest or f.get('verification_status') != 'verified'
                or f.get('checks', {}).get('sha256_match') is not True
                or not generation.isdecimal() or int(generation) <= 0
                or not isinstance(f.get('size_bytes'), int) or f['size_bytes'] < 0
                or f.get('cloud_path') != prefix + name):
            raise ValueError(f'Incomplete or inconsistent receipt: {name}')
        if name.endswith('.parquet') and any(f.get('checks', {}).get(k) is not True
                                            for k in ('row_count_match', 'schema_match')):
            raise ValueError(f'Parquet verification incomplete: {name}')
    return files


def output_receipts(entry: dict) -> dict:
    if entry.get('status') != 'processed':
        raise ValueError('Full processed partition required')
    hashes = output_hashes(entry)
    if not {'validated_trips.parquet', 'hourly_rides.parquet', 'rejected_trips.csv', 'report.json'} <= hashes.keys():
        raise ValueError('Required historical artifacts missing')
    p = entry.get('cloud_publication', {})
    spec = {'source_archive_month': entry['month'], 'project': p.get('project'),
            'bucket': p.get('bucket'), 'prefix': p.get('prefix'), 'artifact_sha256': hashes,
            'source': entry['source'], 'local_start_month_counts': entry['verification']['start_month_counts']}
    if (any(p.get(k) != v for k, v in spec.items()) or p.get('revision') != revision_for(spec)
            or entry['input_rows'] != entry['accepted_rows'] + entry['rejected_rows']
            or sum(spec['local_start_month_counts'].values()) != entry['accepted_rows']):
        raise ValueError('Publication identity or reconciliation differs from ETL manifest')
    return checked_receipts(p, hashes, entry['month'])


def check_surviving_outputs(entry: dict) -> None:
    """Cloud evidence permits absent files, never silently ignores changed survivors."""
    for name, digest in entry['output_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe output filename')
        path = Path(entry['output_dir']) / name
        if path.exists() and (not path.is_file() or fingerprint(path) != digest):
            raise ValueError(f'Local output changed: {path}')


def backup_spec(entry: dict, files: dict, *, project: str, bucket: str, prefix: str) -> dict:
    return {'source_archive_month': entry['month'], 'project': project, 'bucket': bucket,
            'prefix': prefix.strip('/'), 'source': entry['source'],
            'artifacts': files, 'original_zip_available': bool(entry['source'].get('sha256'))}


def source_receipts(entry: dict) -> dict:
    p = entry.get('source_backup', {})
    if p.get('status') != 'published':
        raise ValueError('Complete source and processing-manifest backup required')
    files = p.get('artifacts', {})
    spec = backup_spec(entry, files, project=p.get('project'), bucket=p.get('bucket'),
                       prefix=p.get('prefix', ''))
    if p.get('revision') != revision_for(spec) or any(p.get(k) != v for k, v in spec.items()):
        raise ValueError('Source backup identity differs from manifest')
    # Check the exact source set; never pretend CSV backups are an original ZIP.
    originals = {n: f for n, f in files.items() if f.get('role') in {'original_zip', 'original_csv'}}
    expected = source_files(entry, Path('.'))
    if {n: (f['sha256'], f['role']) for n, f in originals.items()} != {
            n: (f['sha256'], f['role']) for n, f in expected.items()}:
        raise ValueError('Original source backups are incomplete')
    if files.get('processing_manifest.json', {}).get('role') != 'processing_manifest':
        raise ValueError('Processing manifest snapshot backup required')
    for name, digest in entry['output_sha256'].items():
        if name.endswith('.json'):
            f = files.get(name, {})
            if f.get('role') != 'audit_metadata' or f.get('sha256') != digest:
                raise ValueError(f'Audit backup required: {name}')
    return checked_receipts(p, {n: f['sha256'] for n, f in files.items()}, entry['month'])


