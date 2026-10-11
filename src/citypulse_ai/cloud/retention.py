"""Create-only source backups and explicit, generation-pinned recovery. No deletion."""

import json
import math
from pathlib import Path

from citypulse_ai.cloud.gcs import GCSOperationError, GCSStorage, sha256_file
from citypulse_ai.cloud.publish import publish_verified_artifact, parse_destination
from citypulse_ai.etl.archives import check_disk, validate_month, write_json
from citypulse_ai.etl.historical import load_manifest, now
from citypulse_ai.etl.storage_state import (
    backup_spec, source_receipts, check_surviving_outputs, output_receipts, revision_for, source_files,
)


def publish_sources(month: str, output_root: Path, *, raw_root: Path, project: str,
                    bucket: str, prefix: str = 'raw/historical', max_file_bytes: int,
                    store: GCSStorage | None = None, upload_timeout: float = 600,
                    upload_attempts: int = 3) -> dict:
    """Back up original sources, audit JSONs and an immutable per-month manifest snapshot.

    Run under the historical single-writer lock. Repeated attempts recover the
    same object names. The mutable live manifest is never uploaded as a snapshot
    of its own upload status: the immutable snapshot excludes source_backup.
    """
    validate_month(month)
    if not math.isfinite(upload_timeout) or upload_timeout <= 0 or not 1 <= upload_attempts <= 5:
        raise ValueError('Upload timeout must be finite/positive and attempts between 1 and 5')
    if not project.strip() or not bucket.strip() or not prefix.strip('/') or max_file_bytes <= 0:
        raise ValueError('Explicit project, bucket, prefix and positive size limit required')
    root = Path(output_root).resolve()
    manifest = load_manifest(root); entry = manifest['archives'][month]
    if entry.get('status') != 'processed':
        raise ValueError('Full processed source partition required')
    check_surviving_outputs(entry)
    files = source_files(entry, raw_root)
    for name, digest in entry['output_sha256'].items():
        if name.endswith('.json'):
            if name in files or Path(name).name != name:
                raise ValueError('Backup filename collision')
            files[name] = {'path': str(Path(entry['output_dir']) / name), 'sha256': digest,
                           'role': 'audit_metadata'}
    clean_entry = {k: v for k, v in entry.items() if k != 'source_backup'}
    snapshot = {'version': manifest['version'], 'modelling_specification': manifest['modelling_specification'],
                'archives': {month: clean_entry}}
    content = (json.dumps(snapshot, indent=2, sort_keys=True) + '\n').encode()
    import hashlib
    digest = hashlib.sha256(content).hexdigest()
    folder = root / 'source-backup-metadata' / month
    path = folder / f'{digest}.json'
    # Preflight original files before creating even the snapshot or cloud client.
    for f in files.values():
        local = Path(f['path'])
        if local.stat().st_size >= max_file_bytes or sha256_file(local) != f['sha256']:
            raise ValueError(f'Source changed or exceeds backup size limit: {local}')
    if len(content) >= max_file_bytes:
        raise ValueError('Manifest snapshot exceeds backup size limit')
    folder.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing immutable snapshot changed')
    else:
        with path.open('xb') as stream:
            stream.write(content)
    files['processing_manifest.json'] = {'path': str(path), 'sha256': digest,
                                         'role': 'processing_manifest'}
    spec = backup_spec(entry, files, project=project, bucket=bucket, prefix=prefix)
    revision = revision_for(spec)
    publication = entry.get('source_backup')
    if publication and publication.get('revision') != revision:
        raise ValueError('Source backup identity changed; preserve earlier backup and investigate')
    if publication is None:
        publication = {**spec, 'revision': revision, 'status': 'pending', 'files': {}}
        entry['source_backup'] = publication
    publication['status'] = 'publishing'; publication.pop('error', None)
    publication.pop('published_at_utc', None)
    write_json(root / 'manifest.json', manifest)
    current = None
    try:
        cloud = store if store is not None else GCSStorage(project, bucket, upload_timeout=upload_timeout)
        if cloud.bucket_name != bucket:
            raise ValueError('Cloud bucket differs from configured source backup destination')
        for name, f in sorted(files.items()):
            key = f'{spec["prefix"]}/source_archive_month={month}/revision={revision}/{name}'
            current = publication['files'].setdefault(name, {})
            generation = current.get('generation')
            current.update({'cloud_path': f'gs://{bucket}/{key}', 'sha256': f['sha256'],
                            'verification_status': 'publishing'})
            current.pop('error', None)
            write_json(root / 'manifest.json', manifest)
            metadata = {'sha256': f['sha256'], 'source_archive_month': month,
                        'source_filename': name, 'backup_role': f['role'],
                        'publication_revision': revision, 'source_archive_url': entry['source']['url'],
                        'original_zip_available': str(spec['original_zip_available']).lower()}
            def checkpoint(info):
                current.update(info)
                write_json(root / 'manifest.json', manifest)
            current.update(publish_verified_artifact(Path(f['path']), key, store=cloud,
                           metadata=metadata, max_bytes=max_file_bytes, expected_generation=generation,
                           on_inspected=checkpoint, upload_attempts=upload_attempts))
            write_json(root / 'manifest.json', manifest)
        publication.update({'status': 'published', 'published_at_utc': now()})
        source_receipts(entry)
        write_json(root / 'manifest.json', manifest)
        return publication
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        publication.update({'status': status, 'error': f'{type(exc).__name__}: {exc}'})
        if current is not None:
            current.update({'verification_status': status, 'error': publication['error']})
        write_json(root / 'manifest.json', manifest)
        raise


def inspect_remote(store: GCSStorage, entry: dict, publication: dict, name: str, receipt: dict) -> tuple[str, dict]:
    bucket, key = parse_destination(receipt['cloud_path'])
    if bucket != store.bucket_name:
        raise ValueError('Restore/resume cloud bucket differs from receipt')
    info = store.object_info(key)
    metadata = info['metadata']
    if (info['generation'] != receipt['generation'] or info['size_bytes'] != receipt['size_bytes']
            or metadata.get('sha256') != receipt['sha256']
            or metadata.get('source_archive_month') != entry['month']
            or metadata.get('publication_revision') != publication['revision']
            or metadata.get('source_filename') != name
            or metadata.get('source_archive_url') != entry['source']['url']):
        raise GCSOperationError(f'Cloud generation or provenance changed: {key}')
    if publication is entry.get('source_backup'):
        if (metadata.get('backup_role') != publication['artifacts'][name]['role']
                or metadata.get('original_zip_available') != str(publication['original_zip_available']).lower()):
            raise GCSOperationError(f'Cloud source backup role changed: {key}')
    else:
        if (metadata.get('coverage_status') != 'source_archive_only_not_complete_start_months'
                or metadata.get('source_archive_sha256_status') !=
                ('available' if entry['source'].get('sha256') else 'unavailable')):
            raise GCSOperationError(f'Cloud coverage provenance changed: {key}')
        if metadata.get('local_start_month_counts') != json.dumps(entry['verification']['start_month_counts'], sort_keys=True):
            raise GCSOperationError(f'Cloud start-month provenance changed: {key}')
        if entry['source'].get('sha256') and metadata.get('source_archive_sha256') != entry['source']['sha256']:
            raise GCSOperationError(f'Cloud archive hash changed: {key}')
    return key, info


def verify_cloud_partition(entry: dict, store: GCSStorage) -> dict:
    """Prior SHA-verified receipts plus unchanged live generations; no byte download."""
    receipts = output_receipts(entry)
    check_surviving_outputs(entry)
    for name, f in receipts.items():
        inspect_remote(store, entry, entry['cloud_publication'], name, f)
    return {'month': entry['month'], 'status': 'skipped_cloud_verified',
            'output_dir': entry['output_dir']}


def restore_partition(entry: dict, destination: Path, *, store: GCSStorage,
                      kind: str = 'outputs', names: list[str] | None = None,
                      reserve_bytes: int = 5 * 1024**3) -> Path:
    """Restore immutable files to a persistent cache; retain partial/corrupt copies.

    Existing files must hash-match; they are never overwritten or deleted. A
    failed partial download requires a new destination on retry. No ETL is run.
    """
    if kind not in {'outputs', 'sources'}:
        raise ValueError('Restore kind must be outputs or sources')
    receipts = output_receipts(entry) if kind == 'outputs' else source_receipts(entry)
    publication = entry['cloud_publication' if kind == 'outputs' else 'source_backup']
    check_surviving_outputs(entry)
    chosen = list(receipts) if names is None else names
    if not chosen or not set(chosen) <= receipts.keys():
        raise ValueError('Requested files are absent from verified receipts')
    destination = Path(destination).resolve()
    check_disk(destination, sum(receipts[n]['size_bytes'] for n in chosen
                               if not (destination / n).exists()), reserve_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        f = receipts[name]; path = destination / name
        key, info = inspect_remote(store, entry, publication, name, f)
        if path.exists():
            if not path.is_file() or sha256_file(path) != f['sha256']:
                raise GCSOperationError(f'Restore destination collision: {path}')
        else:
            store.download_file(key, path, expected_sha256=f['sha256'], generation=int(f['generation']))
            if sha256_file(path) != f['sha256']:
                raise GCSOperationError(f'Restored SHA-256 mismatch: {path}')
        if store.object_info(key) != info:
            raise GCSOperationError(f'Cloud object changed during restore: {key}')
    return destination
