"""Synthetic one-month orchestration and safety guards; no real network/data."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from citypulse_ai.etl import monthly_cycle as cycle
from citypulse_ai.etl.archives import fingerprint


def test_general_start_gate_cannot_be_disabled(tmp_path,monkeypatch):
    monkeypatch.setattr(cycle.shutil,'disk_usage',lambda p:SimpleNamespace(free=cycle.START_GATE-1))
    with pytest.raises(ValueError,match='28 GiB'):cycle.preflight('2024-09',tmp_path,tmp_path/'work',Mock())


def test_resource_checks_each_stop_without_retry(tmp_path):
    guard=cycle.BoundedRuntime('2024-09',tmp_path,tmp_path/'work',tmp_path/'report');guard.initial_swap=0
    base={'free_bytes':28*cycle.GIB,'workspace_bytes':0,'rss_bytes':100,'swap_allocated_bytes':0}
    guard.check(base)
    for key,value in [('free_bytes',16*cycle.GIB-1),('workspace_bytes',10*cycle.GIB),
                      ('rss_bytes',2*cycle.GIB+1),('swap_allocated_bytes',2*cycle.GIB+1)]:
        with pytest.raises(RuntimeError,match='no automatic retry'):guard.check({**base,key:value})


def test_september_exception_does_not_apply_to_future_months(tmp_path):
    with pytest.raises(ValueError,match='another month'):
        cycle.preflight('2024-10',tmp_path,tmp_path/'work',Mock(),september_budget=True)


def test_only_current_owned_verified_download_is_deleted(tmp_path,monkeypatch):
    owned=tmp_path/'current';owned.mkdir()
    p=owned/'verification.parquet';p.write_bytes(b'copy')
    old=tmp_path/'prior-copy';old.write_bytes(b'preserve')
    source=tmp_path/'source.zip';source.write_bytes(b'original')
    def receipt(path):return {'download_path':str(path),'sha256':fingerprint(path),'size_bytes':path.stat().st_size,
        'generation':'1','cloud_path':'gs://existing/artifact','verification_status':'verified','checks':{'sha256_match':True}}
    publication={'files':{'current':receipt(p),'old':receipt(old),'source':receipt(source)}}
    monkeypatch.setattr(cycle,'verify_publication',Mock(return_value=[]))
    removed=cycle.cleanup_verification_scratch(publication,owned,Mock(),{},'source_backup')
    assert [r['path'] for r in removed]==[str(p)] and not p.exists()
    assert old.read_bytes()==b'preserve' and source.read_bytes()==b'original'


def test_failed_remote_or_hash_verification_never_deletes(tmp_path,monkeypatch):
    p=tmp_path/'copy';p.write_bytes(b'copy')
    r={'download_path':str(p),'sha256':'0'*64,'size_bytes':4,'generation':'1','cloud_path':'gs://b/k',
       'verification_status':'verified','checks':{'sha256_match':True}}
    monkeypatch.setattr(cycle,'verify_publication',Mock(side_effect=ValueError('remote changed')))
    with pytest.raises(ValueError,match='remote changed'):
        cycle.cleanup_verification_scratch({'files':{'x':r}},tmp_path,Mock(),{},'source_backup')
    monkeypatch.setattr(cycle,'verify_publication',Mock(return_value=[]))
    with pytest.raises(ValueError,match='disposable'):
        cycle.cleanup_verification_scratch({'files':{'x':r}},tmp_path,Mock(),{},'source_backup')
    assert p.read_bytes()==b'copy'


def test_monthly_cli_has_no_bulk_or_lower_gate_options():
    with pytest.raises(SystemExit):cycle.main(['--month','2024-09','2024-10'])
    with pytest.raises(SystemExit):cycle.main(['--month','2024-09','--reserve-gib','0'])


def test_cycle_stage_order_cached_resume_and_protected_inputs(tmp_path,monkeypatch):
    root=tmp_path/'historical';root.mkdir();work=tmp_path/'work';report=tmp_path/'report'
    folder=tmp_path/'processed';folder.mkdir()
    (folder/'report.json').write_text('{}')
    entry={'month':'2024-09','source':{'bytes':10},'output_dir':str(folder),
        'output_sha256':{'report.json':'unused'},'input_rows':3,'accepted_rows':2,'rejected_rows':1,
        'verification':{'start_month_counts':{'2024-08':1,'2024-09':1}}}
    manifest={'archives':{'2024-09':entry}};events=[]
    plan={'manifest':{'archives':{}},'inputs':[],'canonical_hashes':{},'month':'2024-09',
          'free_before_bytes':30*cycle.GIB,'archive_envelope_bytes':9*cycle.GIB,'cost_budget':{},
          'source':{'month':'2024-09','bytes':10}}
    monkeypatch.setattr(cycle,'preflight',lambda *args:plan)
    class Guard:
        forbid_sql=False
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def stage(self,name):events.append(name)
    monkeypatch.setattr(cycle,'BoundedRuntime',Guard)
    monkeypatch.setattr(cycle,'load_manifest',lambda root:manifest)
    def ingest(*args,**kwargs):events.append('ETL');return {'status':'processed'}
    monkeypatch.setattr(cycle,'process_month',ingest)
    def publish_processed(*args,**kwargs):
        assert kwargs['upload_attempts']==1
        entry['cloud_publication']={'status':'published','files':{}}
    def publish_raw(*args,**kwargs):entry['source_backup']={'status':'published','files':{}}
    monkeypatch.setattr(cycle,'publish_month',publish_processed);monkeypatch.setattr(cycle,'publish_sources',publish_raw)
    monkeypatch.setattr(cycle,'verify_publication',lambda *args:[])
    monkeypatch.setattr(cycle,'cleanup_verification_scratch',lambda *args:[])
    monkeypatch.setattr(cycle,'source_cleanup_candidates',lambda *args:{'files':[],'not_deleted':True})
    calls=[]
    def reconcile(*args,**kwargs):
        assert kwargs['memory_limit']=='512MB' and kwargs['reserve_bytes']==15*cycle.GIB
        calls.append(1);return {'revision':'r','resumed':len(calls)>1,'accepted_rows':2}
    monkeypatch.setattr(cycle,'reconcile_temporary',reconcile)
    monkeypatch.setattr(cycle,'verify_views',lambda root:[])
    cycle.run_cycle('2024-09',root,work,report,Mock())
    assert events==['ingestion','ETL','processed_publication','source_backup','temporary_reconciliation',
                    'cached_idempotence_check','integrity','complete']
    assert len(calls)==2 and json.loads((report/'result.json').read_text())['cached_rerun_verified_no_SQL']
    # Explicit resume of a published month does not republish or change receipt identity.
    monkeypatch.setattr(cycle,'publish_month',Mock(side_effect=AssertionError('redundant upload')))
    monkeypatch.setattr(cycle,'publish_sources',Mock(side_effect=AssertionError('redundant source upload')))
    cycle.run_cycle('2024-09',root,work,tmp_path/'resume',Mock())


def test_exact_source_candidates_are_reported_without_deletion(tmp_path,monkeypatch):
    folder=tmp_path/'raw';(folder/'archives').mkdir(parents=True)
    source=folder/'archives'/'source.zip';source.write_bytes(b'original zip fixture')
    csv=folder/'member.csv';csv.write_text('original CSV fixture')
    entry={'month':'2024-09','source':{'key':source.name,'sha256':fingerprint(source),
        'members':[{'name':csv.name}]},'csv_files':[{'path':str(csv),'member':csv.name,'sha256':fingerprint(csv)}]}
    receipts={source.name:{'cloud_path':'gs://existing/source.zip','generation':'123','sha256':fingerprint(source)},
              'processing_manifest.json':{'cloud_path':'gs://existing/manifest.json','generation':'456','sha256':'0'*64}}
    monkeypatch.setattr(cycle,'verify_publication',lambda *args:[])
    monkeypatch.setattr(cycle,'source_receipts',lambda *args:receipts)
    result=cycle.source_cleanup_candidates(entry,folder,Mock())
    assert [c['path'] for c in result['files']]==[str(source),str(csv)]
    assert all(c['approval_required'] for c in result['files']) and result['not_deleted']
    assert result['logical_bytes']==source.stat().st_size+csv.stat().st_size
    assert source.exists() and csv.exists()
