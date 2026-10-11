"""Synthetic monthly ETL plus mocked cloud transfers; no ADC/network required."""

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip('google.cloud.storage', reason="Install the cloud extra")
from citypulse_ai.cloud import historical as publishing
from citypulse_ai.cloud.gcs import GCSOperationError, GCSStorage, sha256_file
from citypulse_ai.etl import historical_cli
from citypulse_ai.etl.archives import write_json
from citypulse_ai.etl.historical import load_manifest, register_existing
from citypulse_ai.etl.pipeline import run_pipeline


@pytest.fixture
def partition(tmp_path):
    csv_path = tmp_path / 'synthetic.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ride_id', 'started_at', 'ended_at', 'start_station_id', 'end_station_id'])
        writer.writerows([
            ['boundary', '2025-01-31 23:59:00.123456789', '2025-02-01 00:01:00.123456789', '6569.09_', ''],
            ['feb', '2025-02-01 01:00:00', '2025-02-01 01:10:00', '001.01', ''],
            ['bad', '2025-02-01 02:00:00', '2025-02-01 02:10:00', '', ''],
        ])
    source = tmp_path / 'source'
    run_pipeline(csv_path, source, chunk_size=1)
    root = tmp_path / 'history'
    register_existing('2025-02', source, root)
    return root, source


@pytest.fixture
def cloud():
    store = Mock(spec=GCSStorage)
    store.bucket_name = 'test-bucket'
    objects = {}
    def upload(path, name, *, metadata):
        if name in objects:
            raise GCSOperationError('PreconditionFailed: collision')
        objects[name] = {'bytes': Path(path).read_bytes(), 'metadata': dict(metadata),
                         'generation': str(1000 + len(objects))}
        return sha256_file(path)
    def info(name):
        obj = objects[name]
        return {'generation': obj['generation'], 'size_bytes': len(obj['bytes']),
                'metadata': dict(obj['metadata'])}
    def download(name, path, *, expected_sha256, generation):
        assert str(generation) == objects[name]['generation']
        with Path(path).open('xb') as stream:
            stream.write(objects[name]['bytes'])
        return path
    store.upload_file.side_effect = upload
    store.object_exists.side_effect = lambda name: name in objects
    store.object_info.side_effect = info
    store.download_file.side_effect = download
    return store, objects


def publish(root, store, **kwargs):
    return publishing.publish_month('2025-02', root, project='test-project',
                                    bucket='test-bucket', store=store, **kwargs)


def state(root):
    return load_manifest(root)['archives']['2025-02']


def test_success_preserves_sources_and_records_boundary_provenance(partition, cloud):
    root, source = partition
    store, objects = cloud
    before = {p.name: sha256_file(p) for p in source.iterdir() if p.is_file()}
    result = publish(root, store)
    assert result['status'] == 'published'
    assert state(root)['status'] == 'processed'
    assert result['local_start_month_counts'] == {'2025-01': 1, '2025-02': 1}
    assert all(item['verification_status'] == 'verified' for item in result['files'].values())
    assert all(item['generation'].isdigit() for item in result['files'].values())
    assert set(result['files']) == {'validated_trips.parquet', 'hourly_rides.parquet',
                                   'february_2025_hourly_rides.parquet', 'rejected_trips.csv', 'report.json'}
    for name, obj in objects.items():
        assert '/source_archive_month=2025-02/revision=' in name
        assert obj['metadata']['coverage_status'] == 'source_archive_only_not_complete_start_months'
        assert obj['metadata']['source_archive_sha256_status'] == 'unavailable'
        assert json.loads(obj['metadata']['local_start_month_counts']) == {'2025-01': 1, '2025-02': 1}
    assert {p.name: sha256_file(p) for p in source.iterdir() if p.is_file()} == before
    assert all(Path(f['download_path']).exists() for f in result['files'].values())
    assert state(root)['cloud_publication'] == result


@pytest.mark.parametrize('failure', [KeyboardInterrupt(), GCSOperationError('response lost')])
def test_retry_after_upload_completed_but_response_lost(partition, cloud, failure):
    root, _ = partition
    store, objects = cloud
    original = store.upload_file.side_effect
    def interrupted(path, name, **kwargs):
        original(path, name, **kwargs)
        raise failure
    store.upload_file.side_effect = interrupted
    with pytest.raises(type(failure)):
        publish(root, store)
    expected_status = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed'
    assert state(root)['cloud_publication']['status'] == expected_status
    assert state(root)['status'] == 'processed' and len(objects) == 1
    store.upload_file.side_effect = original
    result = publish(root, store)
    assert result['status'] == 'published'
    assert store.upload_file.call_count == len(result['files'])  # Existing object verified, not uploaded twice.


def test_retry_from_publishing_checkpoint_without_rerunning_etl(partition, cloud):
    root, _ = partition
    store, _ = cloud
    result = publish(root, store)
    manifest = load_manifest(root)
    manifest['archives']['2025-02']['cloud_publication']['status'] = 'publishing'
    write_json(root / 'manifest.json', manifest)
    uploads = store.upload_file.call_count
    resumed = publish(root, store)
    assert resumed['status'] == 'published' and store.upload_file.call_count == uploads
    assert resumed['revision'] == result['revision']


def test_upload_failure_keeps_local_etl_processed(partition, cloud):
    root, _ = partition
    store, objects = cloud
    store.upload_file.side_effect = GCSOperationError('Forbidden')
    with pytest.raises(GCSOperationError):
        publish(root, store)
    entry = state(root)
    assert entry['status'] == 'processed' and entry['cloud_publication']['status'] == 'failed'
    assert not objects
    assert 'Forbidden' in entry['cloud_publication']['error']


def test_hash_mismatch_does_not_mark_partition_published(partition, cloud):
    root, _ = partition
    store, objects = cloud
    def corrupt(name, path, **kwargs):
        Path(path).write_bytes(b'corrupt download')
        return path
    store.download_file.side_effect = corrupt
    with pytest.raises(GCSOperationError, match='SHA-256 mismatch'):
        publish(root, store)
    assert state(root)['cloud_publication']['status'] == 'failed'
    assert len(objects) == 1  # No rollback/deletion of the uploaded object.
    failed = next(iter(state(root)['cloud_publication']['files'].values()))
    assert failed['generation'] == next(iter(objects.values()))['generation']


def test_collision_never_overwrites_existing_bytes(partition, cloud):
    root, _ = partition
    store, objects = cloud
    store.upload_file.side_effect = GCSOperationError('network failed before upload')
    with pytest.raises(GCSOperationError):
        publish(root, store)
    item = next(iter(state(root)['cloud_publication']['files'].values()))
    name = item['cloud_path'].split('test-bucket/', 1)[1]
    objects[name] = {'bytes': b'unrelated', 'metadata': {}, 'generation': '99'}
    calls = store.upload_file.call_count
    with pytest.raises(GCSOperationError, match='collision'):
        publish(root, store)
    assert objects[name]['bytes'] == b'unrelated'
    assert store.upload_file.call_count == calls


@pytest.mark.parametrize('change', ['generation', 'missing'])
def test_verified_objects_are_not_replaced_on_retry(partition, cloud, change):
    root, _ = partition
    store, objects = cloud
    publish(root, store)
    name = next(iter(objects))
    if change == 'generation':
        objects[name]['generation'] = '987654'
    else:
        objects.pop(name)  # Simulate external cloud state; application never deletes.
    calls = store.upload_file.call_count
    with pytest.raises(GCSOperationError):
        publish(root, store)
    assert store.upload_file.call_count == calls
    assert state(root)['cloud_publication']['status'] == 'failed'


def test_generation_change_during_download_rejected(partition, cloud):
    root, _ = partition
    store, objects = cloud
    original = store.download_file.side_effect
    def changed(name, path, **kwargs):
        original(name, path, **kwargs)
        objects[name]['generation'] = '987654'
        return path
    store.download_file.side_effect = changed
    with pytest.raises(GCSOperationError, match='changed during verification'):
        publish(root, store)
    assert state(root)['cloud_publication']['status'] == 'failed'


def test_rejected_partition_or_changed_source_never_contacts_cloud(partition, cloud):
    root, source = partition
    store, _ = cloud
    (source / 'report.json').write_text('modified')
    with pytest.raises(ValueError, match='unchanged'):
        publish(root, store)
    store.object_exists.assert_not_called()
    assert 'cloud_publication' not in state(root)


def test_size_limit_preflight_is_all_or_nothing(partition, cloud):
    root, _ = partition
    store, _ = cloud
    with pytest.raises(ValueError, match='size limit'):
        publish(root, store, max_file_bytes=1)
    store.object_exists.assert_not_called()
    assert 'cloud_publication' not in state(root)


def test_changed_destination_preserves_existing_manifest(partition, cloud):
    root, _ = partition
    store, _ = cloud
    publish(root, store)
    before = (root / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='destination changed'):
        publish(root, store, prefix='other-prefix')
    assert (root / 'manifest.json').read_bytes() == before


def test_cloud_import_is_unnecessary_for_local_cli(monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__
    def no_cloud(name, *args, **kwargs):
        if name.startswith('citypulse_ai.cloud') or name.startswith('google'):
            raise AssertionError('Local-only command imported cloud code')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_cloud)
    process = Mock(return_value={'status': 'processed'})
    monkeypatch.setattr(historical_cli, 'process_month', process)
    historical_cli.main(['--month', '2025-02', '--output-root', str(tmp_path)])
    process.assert_called_once()


def test_cli_publishes_only_after_local_success(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(historical_cli, 'process_month', lambda *a, **k: events.append('etl') or {'status': 'processed'})
    monkeypatch.setattr(publishing, 'publish_month', lambda *a, **k: events.append('publish') or {'status': 'published'})
    historical_cli.main(['--month', '2025-02', '--output-root', str(tmp_path), '--cloud-publish',
                         '--cloud-project', 'p', '--cloud-bucket', 'b'])
    assert events == ['etl', 'publish']


def test_cli_failed_etl_never_publishes(monkeypatch, tmp_path):
    monkeypatch.setattr(historical_cli, 'process_month', Mock(side_effect=RuntimeError('ETL failed')))
    action = Mock()
    monkeypatch.setattr(publishing, 'publish_month', action)
    with pytest.raises(RuntimeError, match='ETL failed'):
        historical_cli.main(['--month', '2025-02', '--output-root', str(tmp_path), '--cloud-publish',
                             '--cloud-project', 'p', '--cloud-bucket', 'b'])
    action.assert_not_called()


def test_existing_only_never_calls_processing(partition, cloud, monkeypatch):
    root, _ = partition
    store, _ = cloud
    action = publishing.publish_month
    monkeypatch.setattr(publishing, 'publish_month', lambda *a, **k: action(*a, **k, store=store))
    process = Mock(side_effect=AssertionError('Existing-only mode called ETL'))
    monkeypatch.setattr(historical_cli, 'process_month', process)
    historical_cli.main(['--month', '2025-02', '--output-root', str(root), '--cloud-publish',
                         '--cloud-project', 'test-project', '--cloud-bucket', 'test-bucket', '--publish-existing-only'])
    process.assert_not_called()
    assert state(root)['cloud_publication']['status'] == 'published'


@pytest.mark.parametrize('extra', [[], ['--sample-rows-per-file', '1'], ['--plan'], ['--probe-boundary']])
def test_bad_cloud_cli_arguments_fail_before_etl(tmp_path, monkeypatch, extra):
    process = Mock()
    monkeypatch.setattr(historical_cli, 'process_month', process)
    config = [] if not extra else ['--cloud-project', 'p', '--cloud-bucket', 'b']
    with pytest.raises(SystemExit):
        historical_cli.main(['--month', '2025-02', '--output-root', str(tmp_path), '--cloud-publish', *config, *extra])
    process.assert_not_called()


def test_verified_files_survive_later_failure_and_are_not_reuploaded(partition, cloud):
    root, _ = partition
    store, objects = cloud
    original = store.upload_file.side_effect
    def fail_second(path, name, **kwargs):
        if len(objects) == 1:
            raise GCSOperationError('second file failed')
        return original(path, name, **kwargs)
    store.upload_file.side_effect = fail_second
    with pytest.raises(GCSOperationError):
        publish(root, store)
    files = state(root)['cloud_publication']['files']
    assert [f['verification_status'] for f in files.values()] == ['verified', 'failed']
    generation = next(f['generation'] for f in files.values() if f['verification_status'] == 'verified')
    store.upload_file.side_effect = original
    result = publish(root, store)
    assert result['status'] == 'published'
    assert result['files']['february_2025_hourly_rides.parquet']['generation'] == generation
    assert len(objects) == len(result['files'])


def test_unprocessed_month_never_contacts_cloud(partition, cloud):
    root, _ = partition
    store, _ = cloud
    manifest = load_manifest(root)
    manifest['archives']['2025-02']['status'] = 'sample_processed'
    write_json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='processed full'):
        publish(root, store)
    store.object_exists.assert_not_called()


def test_missing_start_month_audit_never_contacts_cloud(partition, cloud):
    root, _ = partition
    store, _ = cloud
    manifest = load_manifest(root)
    manifest['archives']['2025-02']['verification']['start_month_counts'] = {'2025-02': 1}
    write_json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='reconcile'):
        publish(root, store)
    store.object_exists.assert_not_called()


def transient_upload_error():
    from requests.exceptions import ConnectionError
    error = GCSOperationError('write timed out')
    error.__cause__ = ConnectionError('The write operation timed out')
    return error


def test_transient_retry_is_bounded_with_exponential_backoff(partition, cloud, monkeypatch):
    root, _ = partition
    store, objects = cloud
    sleeps = []
    monkeypatch.setattr('citypulse_ai.cloud.publish.time.sleep', sleeps.append)
    store.upload_file.side_effect = transient_upload_error()
    with pytest.raises(GCSOperationError, match='write timed out'):
        publish(root, store)
    assert store.upload_file.call_count == 3
    assert sleeps == [1, 2]
    assert not objects
    publication = state(root)['cloud_publication']
    assert publication['status'] == 'failed'
    first = next(iter(publication['files'].values()))
    assert first['upload_attempts'] == 3
    assert first['upload_elapsed_seconds'] >= 0
    assert state(root)['status'] == 'processed'


def test_transient_retry_succeeds_without_reuploading_verified_files(partition, cloud, monkeypatch):
    root, _ = partition
    store, objects = cloud
    original = store.upload_file.side_effect
    calls = 0
    def once(path, name, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise transient_upload_error()
        return original(path, name, **kwargs)
    store.upload_file.side_effect = once
    sleeps = []
    monkeypatch.setattr('citypulse_ai.cloud.publish.time.sleep', sleeps.append)
    result = publish(root, store)
    assert result['status'] == 'published'
    assert sleeps == [1]
    assert calls == len(objects) + 1
    store.upload_file.reset_mock()
    publish(root, store)
    store.upload_file.assert_not_called()


def test_transient_lost_response_verifies_existing_object_without_retry(partition, cloud):
    root, _ = partition
    store, objects = cloud
    original = store.upload_file.side_effect
    def lost(path, name, **kwargs):
        original(path, name, **kwargs)
        raise transient_upload_error()
    store.upload_file.side_effect = lost
    assert publish(root, store)['status'] == 'published'
    assert store.upload_file.call_count == len(objects)
    assert store.download_file.call_count == len(objects)


def test_retry_collision_is_not_treated_as_success(partition, cloud):
    root, _ = partition
    store, objects = cloud
    original = store.upload_file.side_effect
    def collision(path, name, **kwargs):
        original(path, name, **kwargs)
        objects[name]['metadata']['sha256'] = '0' * 64
        raise transient_upload_error()
    store.upload_file.side_effect = collision
    with pytest.raises(GCSOperationError, match='collision'):
        publish(root, store)
    assert store.upload_file.call_count == 1
    assert state(root)['cloud_publication']['status'] == 'failed'
    store.download_file.assert_not_called()


@pytest.mark.parametrize('options', [
    ['--cloud-upload-timeout', 'nan'], ['--cloud-upload-timeout', '0'],
    ['--cloud-upload-attempts', '0'], ['--cloud-upload-attempts', '6'],
])
def test_invalid_upload_configuration_fails_before_processing(options, monkeypatch):
    process = Mock()
    monkeypatch.setattr(historical_cli, 'process_month', process)
    with pytest.raises(SystemExit):
        historical_cli.main(['--month', '2024-01', '--cloud-publish', '--cloud-project',
                             'project', '--cloud-bucket', 'bucket', *options])
    process.assert_not_called()


def test_generation_precondition_race_requires_full_verification(partition, cloud):
    from google.api_core.exceptions import PreconditionFailed
    root, _ = partition
    store, objects = cloud
    original = store.upload_file.side_effect
    def raced(path, name, **kwargs):
        original(path, name, **kwargs)
        error = GCSOperationError('object appeared during upload')
        error.__cause__ = PreconditionFailed('generation zero failed')
        raise error
    store.upload_file.side_effect = raced
    assert publish(root, store)['status'] == 'published'
    assert store.upload_file.call_count == len(objects)
    assert store.download_file.call_count == len(objects)


def test_permission_error_is_not_retried(partition, cloud):
    from google.api_core.exceptions import Forbidden
    root, _ = partition
    store, _ = cloud
    error = GCSOperationError('permission denied')
    error.__cause__ = Forbidden('permission denied')
    store.upload_file.side_effect = error
    with pytest.raises(GCSOperationError, match='permission denied'):
        publish(root, store)
    assert store.upload_file.call_count == 1
