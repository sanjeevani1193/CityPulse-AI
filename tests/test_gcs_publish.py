"""Publication guardrails and round-trip checks without contacting Google Cloud."""

import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip('google.cloud.storage', reason="Install the cloud extra: pip install -e '.[cloud]'")
from citypulse_ai.cloud import publish
from citypulse_ai.cloud.gcs import GCSOperationError, GCSStorage, sha256_file


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / 'hourly.parquet'
    table = pa.table({'start_station_id': ['6569.09_', '6569.09'], 'ride_count': [1, 2]})
    table = table.replace_schema_metadata({b'coverage_policy': b'observed_hours_only'})
    pq.write_table(table, path)
    return path


@pytest.fixture
def store():
    cloud = Mock(spec=GCSStorage)
    cloud.bucket_name = 'test-bucket'
    remote = {}
    def upload(path, name, *, metadata):
        remote['path'] = path
        remote['metadata'] = dict(metadata)
        return sha256_file(path)
    def download(name, path, *, expected_sha256):
        with Path(path).open('xb') as stream:
            stream.write(remote['path'].read_bytes())
        assert sha256_file(path) == expected_sha256
        return path
    cloud.upload_file.side_effect = upload
    cloud.object_exists.return_value = True
    cloud.object_metadata.side_effect = lambda name: remote['metadata']
    cloud.download_file.side_effect = download
    return cloud


@pytest.fixture(autouse=True)
def retained_download_directory(tmp_path, monkeypatch):
    def directory(**kwargs):
        path = tmp_path / 'download'
        path.mkdir()
        return str(path)
    monkeypatch.setattr(publish.tempfile, 'mkdtemp', directory)


def run(artifact, store, tmp_path):
    return publish.publish_parquet(
        artifact, 'gs://test-bucket/curated/forecasting/hourly.parquet',
        project='test-project', report_path=tmp_path / 'receipt.json', store=store,
    )


def test_publish_preserves_bytes_schema_provenance_and_receipt(artifact, store, tmp_path):
    (tmp_path / 'report.json').write_text('{"validated": true}')
    (tmp_path / 'verification.json').write_text('{"rows": 2}')
    before = artifact.stat()
    result = run(artifact, store, tmp_path)
    assert result['verified'] and result['uploaded'] and result['object_exists']
    assert result['row_count'] == 2
    assert all(result['checks'].values()) and result['metadata_match']
    assert result['object_metadata']['source_filename'] == 'hourly.parquet'
    assert result['object_metadata']['report_sha256'] == sha256_file(tmp_path / 'report.json')
    assert result['object_metadata']['verification_sha256'] == sha256_file(tmp_path / 'verification.json')
    assert result['sha256'] == sha256_file(artifact)
    assert Path(result['download_path']).read_bytes() == artifact.read_bytes()
    assert artifact.stat().st_mtime_ns == before.st_mtime_ns
    assert json.loads((tmp_path / 'receipt.json').read_text()) == result
    assert pq.read_schema(result['download_path']).metadata[b'coverage_policy'] == b'observed_hours_only'
    store.upload_file.assert_called_once()


@pytest.mark.parametrize('uri', ['https://bucket/file', 'gs://bucket/', 'gs:///file', 'gs://bucket/file?x=y', 'gs://bucket/file#fragment'])
def test_invalid_destination_fails_before_upload(artifact, store, tmp_path, uri):
    with pytest.raises(ValueError, match='Destination'):
        publish.publish_parquet(artifact, uri, project='p', report_path=tmp_path / 'receipt', store=store)
    store.upload_file.assert_not_called()


def test_exact_object_name_is_preserved():
    assert publish.parse_destination('gs://bucket/curated/file name.parquet') == ('bucket', 'curated/file name.parquet')


@pytest.mark.parametrize('size', [publish.MAX_ARTIFACT_BYTES, publish.MAX_ARTIFACT_BYTES + 1])
def test_100_mib_limit_precedes_parquet_read_or_upload(store, tmp_path, size):
    source = tmp_path / 'large.parquet'
    with source.open('wb') as stream:
        stream.truncate(size)  # Sparse synthetic file, not a real dataset.
    with pytest.raises(ValueError, match='under 100 MiB'):
        run(source, store, tmp_path)
    store.upload_file.assert_not_called()


def test_invalid_parquet_fails_before_upload(store, tmp_path):
    source = tmp_path / 'invalid.parquet'
    source.write_bytes(b'not parquet')
    with pytest.raises(pa.ArrowInvalid):
        run(source, store, tmp_path)
    store.upload_file.assert_not_called()


def test_existing_receipt_is_not_overwritten(artifact, store, tmp_path):
    receipt = tmp_path / 'receipt.json'
    receipt.write_bytes(b'keep this')
    with pytest.raises(FileExistsError):
        run(artifact, store, tmp_path)
    assert receipt.read_bytes() == b'keep this'
    store.upload_file.assert_not_called()


def test_existing_cloud_object_failure_recorded_without_retry(artifact, store, tmp_path):
    store.upload_file.side_effect = GCSOperationError('PreconditionFailed: object exists')
    with pytest.raises(GCSOperationError, match='PreconditionFailed'):
        run(artifact, store, tmp_path)
    result = json.loads((tmp_path / 'receipt.json').read_text())
    assert not result['uploaded'] and not result['verified']
    store.upload_file.assert_called_once()
    store.download_file.assert_not_called()


def test_wrong_bucket_never_uploads(artifact, store, tmp_path):
    store.bucket_name = 'wrong'
    with pytest.raises(ValueError, match='bucket'):
        run(artifact, store, tmp_path)
    store.upload_file.assert_not_called()


def test_changed_source_digest_is_not_verified(artifact, store, tmp_path):
    store.upload_file.side_effect = None
    store.upload_file.return_value = '0' * 64
    with pytest.raises(GCSOperationError, match='Source changed'):
        run(artifact, store, tmp_path)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['uploaded'] and not receipt['verified']
    store.download_file.assert_not_called()


def test_remote_metadata_mismatch_fails(artifact, store, tmp_path):
    store.object_metadata.side_effect = lambda name: {'sha256': 'wrong'}
    with pytest.raises(GCSOperationError, match='provenance'):
        run(artifact, store, tmp_path)
    store.download_file.assert_not_called()


def test_download_failure_retains_receipt(artifact, store, tmp_path):
    store.download_file.side_effect = GCSOperationError('Forbidden')
    with pytest.raises(GCSOperationError, match='Forbidden'):
        run(artifact, store, tmp_path)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['uploaded'] and not receipt['verified']
    assert 'Forbidden' in receipt['error']


def test_structural_verification_detects_changed_data(artifact, tmp_path):
    copy = tmp_path / 'changed.parquet'
    pq.write_table(pa.table({'ride_count': [99]}), copy)
    with pytest.raises(GCSOperationError, match="'row_count_match': False"):
        publish.verify_parquet(artifact, copy)
    assert copy.exists()


def test_schema_metadata_difference_is_detected(artifact, tmp_path):
    copy = tmp_path / 'changed.parquet'
    pq.write_table(pq.read_table(artifact).replace_schema_metadata({b'coverage_policy': b'changed'}), copy)
    with pytest.raises(GCSOperationError, match="'schema_match': False"):
        publish.verify_parquet(artifact, copy)


def test_cli_passes_arguments_and_prints_receipt(artifact, tmp_path, monkeypatch, capsys):
    action = Mock(return_value={'verified': True})
    monkeypatch.setattr(publish, 'publish_parquet', action)
    assert publish.main([str(artifact), 'gs://bucket/file', '--project', 'p', '--report', str(tmp_path / 'r')]) == 0
    assert json.loads(capsys.readouterr().out) == {'verified': True}
    action.assert_called_once_with(artifact, 'gs://bucket/file', project='p', report_path=tmp_path / 'r')


def test_cli_returns_failure_without_upload_for_invalid_destination(artifact, tmp_path, capsys):
    assert publish.main([str(artifact), 'invalid', '--project', 'p', '--report', str(tmp_path / 'r')]) == 1
    assert 'Publication failed' in capsys.readouterr().err
