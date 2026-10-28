"""Bounded synthetic recovery checks; never contact GCS."""
from io import BytesIO
import hashlib
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from citypulse_ai.cloud.forecasting_release import download, zip_member


def record(content):
    return {'cloud_path': 'gs://existing-bucket/frozen', 'generation': '42',
            'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}


class Store:
    def __init__(self, content):
        self.content = content; self.calls = []; self.spec = record(content)
        self.bucket = self
    def blob(self, key, generation):
        assert key == 'frozen' and generation == 42
        self.calls.append('metadata')
        return SimpleNamespace(size=len(self.content), metadata={'sha256': self.spec['sha256']},
            reload=lambda **kwargs: None, open=self.open)
    def open(self, mode, **kwargs):
        assert mode == 'rb' and kwargs['chunk_size'] == 1024**2
        return BytesIO(self.content)
    def download_file(self, key, destination, **kwargs):
        assert kwargs == {'generation': 42, 'expected_sha256': self.spec['sha256']}
        self.calls.append('download'); destination.write_bytes(self.content)


def test_idempotent_pinned_download_and_no_overwrite(tmp_path):
    store = Store(b'frozen'); path = tmp_path/'weights'
    download(store.spec, path, store); download(store.spec, path, store)
    assert store.calls == ['metadata', 'download'] and path.read_bytes() == b'frozen'
    path.write_bytes(b'differ')
    with pytest.raises(ValueError, match='never overwrite'): download(store.spec, path, store)
    assert path.read_bytes() == b'differ'


def test_remote_identity_fails_closed(tmp_path):
    store = Store(b'frozen'); spec = {**store.spec, 'sha256': '0'*64}
    with pytest.raises(ValueError, match='identity differs'): download(spec, tmp_path/'out', store)
    assert 'download' not in store.calls and not (tmp_path/'out').exists()


def test_selected_zip_member_and_integrity(tmp_path):
    data = BytesIO()
    with ZipFile(data, 'w') as archive:
        archive.writestr('selected.parquet', b'small fixture')
        archive.writestr('unrelated.parquet', b'not extracted')
    store = Store(data.getvalue()); member = {**record(b'small fixture'), 'name': 'selected.parquet'}
    path = tmp_path/'member'
    zip_member(store.spec, member, path, store)
    assert path.read_bytes() == b'small fixture' and list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match='SHA-256'):
        zip_member(store.spec, {**member, 'sha256': '0'*64}, tmp_path/'invalid', store)
    with pytest.raises(ValueError, match='budget'):
        zip_member(store.spec, {**member, 'size_bytes': 33*1024**2}, tmp_path/'large', store)
