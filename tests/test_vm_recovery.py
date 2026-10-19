"""Tiny synthetic partitions, local bundles and mocked generation-pinned GCS."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.test_cloud_retention import setup, outputs, sources
from citypulse_ai.etl import vm_recovery as recovery
from citypulse_ai.etl.archives import fingerprint, write_json
from citypulse_ai.etl.historical import load_manifest
from citypulse_ai.etl.storage_state import revision_for
from citypulse_ai.etl.temporary_reconciliation import reconcile_temporary
from citypulse_ai.cloud.gcs import GCSOperationError


@pytest.fixture
def prepared(setup,tmp_path):
    root,raw,output,store,objects=setup
    outputs(root,store);sources(root,raw,store)
    manifest=load_manifest(root)
    manifest['archives']['2025-02']['output_dir']='/Users/example/old-mac/processed'
    manifest['archives']['2024-09']={'month':'2024-09','status':'interrupted',
        'output_dir':'/Users/example/old-mac/incomplete','source':{'sha256':'a'*64},
        'attempts':[{'status':'interrupted','output_dir':'/Users/example/old-mac/incomplete'}]}
    write_json(root/'manifest.json',manifest)
    bundle=tmp_path/'bundle';digest=recovery.export_bundle(root,bundle)
    store.reset_mock()
    return root,output,store,objects,bundle,digest,manifest


def test_exact_mapping_receipts_preserved_and_original_untouched(prepared,tmp_path):
    root,_,store,objects,bundle,digest,original=prepared
    vm=tmp_path/'vm';result=recovery.restore_bundle(bundle,vm,digest,store)
    current=load_manifest(vm)
    assert result['restored_processed_months']==['2025-02']
    expected=deepcopy(original)
    for month,e in expected['archives'].items():e['output_dir']=str(vm/f'source_month={month}')
    assert current==expected and load_manifest(root)==original
    assert (vm/'source_month=2025-02/validated_trips.parquet').exists()
    assert (vm/'provenance/source_month=2025-02/processing_manifest.json').exists()
    assert not (vm/'canonical').exists() and not list(vm.rglob('*.duckdb'))
    assert not list(vm.rglob('*.csv.part'))
    store.upload_file.assert_not_called()
    for call in store.download_file.call_args_list:
        key=call.args[0]
        assert str(call.kwargs['generation'])==objects[key]['generation']
        assert call.kwargs['expected_sha256']==objects[key]['metadata']['sha256']
    downloads=store.download_file.call_count
    assert recovery.restore_bundle(bundle,vm,digest,store)==result
    assert store.download_file.call_count==downloads


def test_restore_resumes_after_interrupted_download_and_preserves_advanced_state(prepared,tmp_path):
    _,_,store,_,bundle,digest,_=prepared
    vm=tmp_path/'vm';original=store.download_file.side_effect;calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt()
        return original(*args,**kwargs)
    store.download_file.side_effect=interrupt
    with pytest.raises(KeyboardInterrupt):recovery.restore_bundle(bundle,vm,digest,store)
    assert (vm/'recovery-origin.json').exists()
    store.download_file.side_effect=original
    recovery.restore_bundle(bundle,vm,digest,store)
    m=load_manifest(vm);m['archives']['2024-09']['attempts'].append({'status':'interrupted','note':'new VM attempt'})
    write_json(vm/'manifest.json',m)
    recovery.restore_bundle(bundle,vm,digest,store)
    assert load_manifest(vm)==m


def test_changed_or_missing_remote_backup_refused(prepared,tmp_path):
    _,_,store,objects,bundle,digest,_=prepared
    for obj in objects.values():obj['generation']='999999'
    with pytest.raises(GCSOperationError,match='generation|Generation|Cloud'):
        recovery.restore_bundle(bundle,tmp_path/'vm',digest,store)
    store.download_file.assert_not_called()


def test_missing_remote_object_refused(prepared,tmp_path):
    _,_,store,_,bundle,digest,_=prepared
    store.object_info.side_effect=GCSOperationError('Missing recorded object')
    with pytest.raises(GCSOperationError,match='Missing'):
        recovery.restore_bundle(bundle,tmp_path/'vm',digest,store)
    store.download_file.assert_not_called()


def test_wrong_bundle_checksum_or_corrupt_file_never_downloads(prepared,tmp_path):
    _,_,store,_,bundle,digest,_=prepared
    with pytest.raises(ValueError,match='SHA-256'):
        recovery.restore_bundle(bundle,tmp_path/'vm','0'*64,store)
    (bundle/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='Invalid recovery file'):
        recovery.restore_bundle(bundle,tmp_path/'vm',digest,store)
    store.download_file.assert_not_called()


def test_replacement_vm_can_reuse_former_absolute_path(prepared,tmp_path):
    _,_,store,_,bundle,_,_=prepared
    replacement=tmp_path/'replacement-vm-root'
    # The exported pathname belongs to a former machine; no local source tree
    # exists there. Its trusted transfer digest covers the original-root field.
    index=json.loads((bundle/'bundle.json').read_text())
    index['original_root']=str(replacement)
    write_json(bundle/'bundle.json',index);digest=fingerprint(bundle/'bundle.json')
    recovery.restore_bundle(bundle,replacement,digest,store)
    downloads=store.download_file.call_count
    recovery.restore_bundle(bundle,replacement,digest,store)
    assert store.download_file.call_count==downloads


def test_invalid_receipts_and_original_root_refused(prepared,tmp_path):
    root,_,store,_,bundle,digest,manifest=prepared
    with pytest.raises(ValueError,match='separate'):
        recovery.restore_bundle(bundle,root,digest,store)
    manifest['archives']['2025-02'].pop('source_backup')
    write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='backup required'):
        recovery.export_bundle(root,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_unowned_root_and_symlink_mapping_refused(prepared,tmp_path):
    _,_,store,_,bundle,digest,_=prepared
    vm=tmp_path/'vm';vm.mkdir();(vm/'unrelated.txt').write_text('preserve')
    with pytest.raises(ValueError,match='owned'):
        recovery.restore_bundle(bundle,vm,digest,store)
    assert (vm/'unrelated.txt').read_text()=='preserve'
    other=tmp_path/'other';recovery.restore_bundle(bundle,other,digest,store)
    current=load_manifest(other);current['archives']['2025-02']['output_dir']='/outside'
    write_json(other/'manifest.json',current)
    with pytest.raises(ValueError,match='changed'):
        recovery.restore_bundle(bundle,other,digest,store)
    with pytest.raises(ValueError,match='Unsafe'):
        recovery.safe_path(other,'../outside')
    link=other/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):
        recovery.safe_path(other,'link/file')


def test_complete_and_intermediate_checkpoints_restore_and_reuse(setup,tmp_path):
    root,raw,output,store,_=setup
    outputs(root,store);sources(root,raw,store)
    first=reconcile_temporary(root,reserve_bytes=0,spill_limit_bytes=1024**2)
    bundle=tmp_path/'bundle';digest=recovery.export_bundle(root,bundle)
    index=json.loads((bundle/'bundle.json').read_text())
    assert any('/checkpoints/' in f['name'] for f in index['files'])
    vm=tmp_path/'vm';recovery.restore_bundle(bundle,vm,digest,store)
    second=reconcile_temporary(vm,reserve_bytes=0,spill_limit_bytes=1024**2)
    assert second['resumed'] and second['revision']==first['revision']
    assert second['start_month_counts']==first['start_month_counts']
    assert second['accepted_rows']==first['accepted_rows']==2
    # An interrupted stage bundle (without a completed audit) is also valid.
    # Build it separately rather than deleting the original audit.
    incomplete=tmp_path/'incomplete';incomplete.mkdir()
    write_json(incomplete/'manifest.json',load_manifest(root))
    import shutil
    shutil.copytree(root/'reconciliation-audits/checkpoints',incomplete/'reconciliation-audits/checkpoints')
    partial=tmp_path/'partial';sha=recovery.export_bundle(incomplete,partial)
    recovery.restore_bundle(partial,tmp_path/'vm-partial',sha,store)
    assert recovery.validate_audits(tmp_path/'vm-partial')
    resumed=reconcile_temporary(tmp_path/'vm-partial',reserve_bytes=0,spill_limit_bytes=1024**2)
    assert resumed['accepted_rows']==first['accepted_rows']
    assert resumed['start_month_counts']==first['start_month_counts']


def test_insufficient_disk_stops_before_cloud_download(prepared,tmp_path,monkeypatch):
    _,_,store,_,bundle,digest,_=prepared
    def no_disk(*args):raise OSError('Insufficient disk reserve')
    monkeypatch.setattr(recovery,'check_disk',no_disk)
    with pytest.raises(OSError,match='reserve'):
        recovery.restore_bundle(bundle,tmp_path/'vm',digest,store)
    store.download_file.assert_not_called()


def test_corrupt_checkpoint_integrity_is_not_repaired_silently(setup,tmp_path):
    root,raw,_,store,_=setup
    outputs(root,store);sources(root,raw,store)
    reconcile_temporary(root,reserve_bytes=0,spill_limit_bytes=1024**2)
    p=next((root/'reconciliation-audits/checkpoints').rglob('source-counts.json'))
    b=json.loads(p.read_text());b['payload']['identity']='0'*64
    b['sha256']=revision_for(b['payload']);write_json(p,b)
    with pytest.raises(ValueError,match='checkpoint'):
        recovery.export_bundle(root,tmp_path/'bundle')
