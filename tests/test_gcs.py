"""GCS behavior tested without ADC, cloud requests, or real datasets."""

import hashlib
from unittest.mock import Mock, patch

import pytest

pytest.importorskip('google.cloud.storage', reason="Install the cloud extra: pip install -e '.[cloud]'")
from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from requests.exceptions import ConnectionError

from citypulse_ai.cloud.gcs import (
    ChecksumMismatchError, GCSOperationError, GCSStorage, sha256_file,
)


@pytest.fixture
def gcs():
    client = Mock()
    store = GCSStorage('test-project', 'test-bucket', client=client)
    return store, client.bucket.return_value.blob.return_value


def test_default_client_uses_adc():
    with patch('citypulse_ai.cloud.gcs.storage.Client') as client:
        GCSStorage('test-project', 'test-bucket')
    client.assert_called_once_with(project='test-project')


def test_missing_adc_has_actionable_error():
    with patch('citypulse_ai.cloud.gcs.storage.Client', side_effect=DefaultCredentialsError('missing')):
        with pytest.raises(GCSOperationError, match='ADC'):
            GCSStorage('test-project', 'test-bucket')


def test_hash_empty_and_multiblock_files(tmp_path):
    path = tmp_path / 'payload'
    path.write_bytes(b'')
    assert sha256_file(path) == hashlib.sha256(b'').hexdigest()
    data = b'citypulse\n' * 250_000
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_upload_sends_exact_bytes_and_prevents_overwrite(gcs, tmp_path):
    store, blob = gcs
    data = b'CityPulse AI\n'
    path = tmp_path / 'source.txt'
    path.write_bytes(data)
    captured = []
    blob.upload_from_file.side_effect = lambda stream, **kwargs: captured.append(stream.read())
    assert store.upload_file(path, 'smoke/test.txt') == hashlib.sha256(data).hexdigest()
    assert captured == [data]
    assert blob.upload_from_file.call_args.kwargs == {
        'if_generation_match': 0, 'checksum': None, 'timeout': 600, 'retry': None,
    }
    blob.delete.assert_not_called()


def test_missing_source_never_uploads(gcs, tmp_path):
    store, blob = gcs
    with pytest.raises(FileNotFoundError):
        store.upload_file(tmp_path / 'missing', 'test.txt')
    blob.upload_from_file.assert_not_called()


@pytest.mark.parametrize('failure', [Forbidden('denied'), PreconditionFailed('exists'), RefreshError('expired'), ConnectionError('offline')])
def test_upload_failure_is_clear_and_never_deletes(gcs, tmp_path, failure):
    store, blob = gcs
    source = tmp_path / 'source'
    source.write_text('tiny')
    blob.upload_from_file.side_effect = failure
    with pytest.raises(GCSOperationError, match='Upload failed') as error:
        store.upload_file(source, 'target')
    assert error.value.__cause__ is failure
    blob.delete.assert_not_called()


@pytest.mark.parametrize('exists', [True, False])
def test_object_exists(gcs, exists):
    store, blob = gcs
    blob.exists.return_value = exists
    assert store.object_exists('target') is exists
    blob.exists.assert_called_once_with(timeout=60, retry=None)


def test_existence_permission_error_is_not_false(gcs):
    store, blob = gcs
    blob.exists.side_effect = Forbidden('denied')
    with pytest.raises(GCSOperationError, match='Existence check failed'):
        store.object_exists('target')


def test_verified_download_accepts_uppercase_digest(gcs, tmp_path):
    store, blob = gcs
    data = b'tiny payload\n'
    blob.download_to_file.side_effect = lambda stream, **kwargs: stream.write(data)
    target = tmp_path / 'download'
    assert store.download_file('target', target, expected_sha256=hashlib.sha256(data).hexdigest().upper()) == target
    assert target.read_bytes() == data
    blob.download_to_file.assert_called_once()
    assert blob.download_to_file.call_args.kwargs == {'checksum': None, 'timeout': 60, 'retry': None}
    blob.delete.assert_not_called()


def test_checksum_mismatch_retains_download(gcs, tmp_path):
    store, blob = gcs
    blob.download_to_file.side_effect = lambda stream, **kwargs: stream.write(b'changed')
    target = tmp_path / 'download'
    with pytest.raises(ChecksumMismatchError, match='File retained'):
        store.download_file('target', target, expected_sha256='0' * 64)
    assert target.read_bytes() == b'changed'
    blob.delete.assert_not_called()


def test_existing_destination_is_preserved(gcs, tmp_path):
    store, blob = gcs
    target = tmp_path / 'existing'
    target.write_bytes(b'keep me')
    with pytest.raises(FileExistsError):
        store.download_file('target', target, expected_sha256='0' * 64)
    assert target.read_bytes() == b'keep me'
    blob.download_to_file.assert_not_called()


@pytest.mark.parametrize('failure', [NotFound('missing'), Forbidden('denied'), RefreshError('expired'), ConnectionError('offline')])
def test_download_failure_retains_partial_bytes(gcs, tmp_path, failure):
    store, blob = gcs
    def partial_download(stream, **kwargs):
        stream.write(b'partial')
        raise failure
    blob.download_to_file.side_effect = partial_download
    target = tmp_path / 'partial'
    with pytest.raises(GCSOperationError, match='Download failed') as error:
        store.download_file('target', target, expected_sha256='0' * 64)
    assert error.value.__cause__ is failure
    assert target.read_bytes() == b'partial'
    blob.delete.assert_not_called()


@pytest.mark.parametrize('digest', ['', 'g' * 64, 'a' * 63])
def test_bad_digest_rejected_before_download(gcs, tmp_path, digest):
    store, blob = gcs
    target = tmp_path / 'download'
    with pytest.raises(ValueError, match='64 hexadecimal'):
        store.download_file('target', target, expected_sha256=digest)
    assert not target.exists()
    blob.download_to_file.assert_not_called()


def test_blank_object_name_rejected(gcs):
    store, blob = gcs
    with pytest.raises(ValueError, match='Object name'):
        store.object_exists('  ')
    blob.exists.assert_not_called()


def test_upload_records_metadata_with_measured_digest(gcs, tmp_path):
    store, blob = gcs
    source = tmp_path / 'source'
    source.write_bytes(b'tiny')
    digest = store.upload_file(source, 'target', metadata={'source_filename': 'source', 'sha256': 'wrong'})
    assert blob.metadata == {'source_filename': 'source', 'sha256': digest}
    blob.delete.assert_not_called()


def test_object_metadata_is_read_only(gcs):
    store, blob = gcs
    blob.metadata = {'sha256': 'hash', 'row_count': '2'}
    assert store.object_metadata('target') == blob.metadata
    blob.reload.assert_called_once_with(timeout=60, retry=None)
    blob.patch.assert_not_called()
    blob.delete.assert_not_called()


def test_object_metadata_permission_error_is_clear(gcs):
    store, blob = gcs
    blob.reload.side_effect = Forbidden('denied')
    with pytest.raises(GCSOperationError, match='Metadata read failed'):
        store.object_metadata('target')


def test_object_info_records_generation_size_and_metadata(gcs):
    store, blob = gcs
    blob.generation = 1234567890123456
    blob.size = 36
    blob.metadata = {'sha256': 'digest'}
    assert store.object_info('target') == {
        'generation': '1234567890123456', 'size_bytes': 36, 'metadata': {'sha256': 'digest'},
    }
    blob.reload.assert_called_once_with(timeout=60, retry=None)
    blob.delete.assert_not_called()


def test_generation_pinned_download(gcs, tmp_path):
    store, blob = gcs
    data = b'old generation'
    blob.download_to_file.side_effect = lambda stream, **kwargs: stream.write(data)
    store.download_file('target', tmp_path / 'copy', expected_sha256=hashlib.sha256(data).hexdigest(), generation=42)
    store.bucket.blob.assert_called_with('target', generation=42)
    assert blob.download_to_file.call_args.kwargs['if_generation_match'] == 42
    blob.delete.assert_not_called()


def test_object_info_network_error_is_clear(gcs):
    store, blob = gcs
    blob.reload.side_effect = ConnectionError('offline')
    with pytest.raises(GCSOperationError, match='Object inspection failed'):
        store.object_info('target')


@pytest.mark.parametrize('generation', [0, -1])
def test_invalid_download_generation_never_creates_file(gcs, tmp_path, generation):
    store, blob = gcs
    target = tmp_path / 'copy'
    with pytest.raises(ValueError, match='positive'):
        store.download_file('target', target, expected_sha256='0' * 64, generation=generation)
    assert not target.exists()
    blob.download_to_file.assert_not_called()


def test_configurable_upload_timeout_preserves_precondition(tmp_path):
    client = Mock()
    store = GCSStorage('project', 'bucket', client=client, upload_timeout=123)
    source = tmp_path / 'source'
    source.write_text('tiny')
    store.upload_file(source, 'target')
    kwargs = client.bucket.return_value.blob.return_value.upload_from_file.call_args.kwargs
    assert kwargs['timeout'] == 123
    assert kwargs['if_generation_match'] == 0
    assert kwargs['retry'] is None


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_upload_timeout_never_initializes_adc(timeout):
    with patch('citypulse_ai.cloud.gcs.storage.Client') as client:
        with pytest.raises(ValueError, match='timeout'):
            GCSStorage('project', 'bucket', upload_timeout=timeout)
        client.assert_not_called()
