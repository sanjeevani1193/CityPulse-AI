"""Synthetic storage recovery only; every cloud operation is mocked."""

import csv
import json
from pathlib import Path
from unittest.mock import Mock
import zipfile

import pytest

pytest.importorskip('google.cloud.storage')
from citypulse_ai.cloud.gcs import GCSOperationError, GCSStorage, sha256_file
from citypulse_ai.cloud.historical import publish_month
from citypulse_ai.cloud.retention import publish_sources, restore_partition, verify_cloud_partition
from citypulse_ai.etl import historical, historical_cli, storage_cli
from citypulse_ai.etl.archives import write_json
from citypulse_ai.etl.history_catalog import build_catalog
from citypulse_ai.etl.pipeline import run_pipeline
from citypulse_ai.etl.retention import retention_plan
from citypulse_ai.etl.storage_state import output_receipts, source_receipts


@pytest.fixture
def setup(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    csv_path = raw / '202502-citibike-tripdata_1.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ride_id','started_at','ended_at','start_station_id','end_station_id'])
        writer.writerow(['boundary','2025-01-31 23:59:00.123456789','2025-02-01 00:01:00.123456789','6569.09_',''])
        writer.writerow(['feb','2025-02-01 01:00:00','2025-02-01 01:10:00','001.01',''])
    output = tmp_path / 'output'
    run_pipeline(csv_path, output, chunk_size=1)
    root = tmp_path / 'history'
    historical.register_existing('2025-02', output, root)
    store = Mock(spec=GCSStorage); store.bucket_name = 'bucket'
    objects = {}
    def upload(path, key, *, metadata):
        assert key not in objects
        objects[key] = {'bytes': Path(path).read_bytes(), 'metadata': dict(metadata),
                        'generation': str(1000 + len(objects))}
        return sha256_file(path)
    def info(key):
        f = objects[key]
        return {'generation': f['generation'], 'size_bytes': len(f['bytes']), 'metadata': dict(f['metadata'])}
    def download(key, path, *, expected_sha256, generation):
        assert str(generation) == objects[key]['generation']
        with Path(path).open('xb') as stream:stream.write(objects[key]['bytes'])
        return path
    store.upload_file.side_effect = upload
    store.object_exists.side_effect = lambda key: key in objects
    store.object_info.side_effect = info
    store.download_file.side_effect = download
    return root, raw, output, store, objects


def entry(root):
    return historical.load_manifest(root)['archives']['2025-02']


def sources(root, raw, store, **kwargs):
    return publish_sources('2025-02', root, raw_root=raw, project='project', bucket='bucket',
                           max_file_bytes=1024**2, store=store, **kwargs)


def outputs(root, store):
    return publish_month('2025-02', root, project='project', bucket='bucket', store=store)


def add_zip(root, raw):
    folder = raw / 'archives'; folder.mkdir()
    path = folder / '202502-citibike-tripdata.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for csv_path in raw.glob('*.csv'):archive.write(csv_path, arcname=csv_path.name)
    m = historical.load_manifest(root)
    m['archives']['2025-02']['source'].update({'key': path.name, 'sha256': sha256_file(path)})
    write_json(root / 'manifest.json', m)
    return path


@pytest.mark.parametrize('kind', ['zip', 'csv'])
def test_original_sources_and_snapshot_preserve_provenance(setup, kind):
    root, raw, output, store, objects = setup
    if kind == 'zip':add_zip(root, raw)
    before = {str(p): sha256_file(p) for folder in [raw, output] for p in folder.rglob('*') if p.is_file()}
    result = sources(root, raw, store)
    assert result['status'] == 'published'
    assert result['original_zip_available'] is (kind == 'zip')
    originals = [f['role'] for f in result['artifacts'].values() if f['role'].startswith('original_')]
    assert originals == ['original_' + kind]
    assert 'processing_manifest.json' in source_receipts(entry(root))
    snapshot_path = result['artifacts']['processing_manifest.json']['path']
    snapshot = json.loads(Path(snapshot_path).read_text())
    assert 'source_backup' not in snapshot['archives']['2025-02']
    assert snapshot['archives']['2025-02']['source'] == entry(root)['source']
    assert {p: sha256_file(Path(p)) for p in before} == before
    assert all(f['metadata']['original_zip_available'] == str(kind == 'zip').lower() for f in objects.values())
    uploads = store.upload_file.call_count
    sources(root, raw, store)
    assert store.upload_file.call_count == uploads


def test_backup_interruption_recovers_without_duplicate_upload(setup):
    root, raw, _, store, objects = setup
    original = store.upload_file.side_effect
    def lost(path, key, **kwargs):
        original(path, key, **kwargs)
        raise KeyboardInterrupt()
    store.upload_file.side_effect = lost
    with pytest.raises(KeyboardInterrupt):sources(root, raw, store)
    assert entry(root)['source_backup']['status'] == 'interrupted'
    assert entry(root)['status'] == 'processed' and len(objects) == 1
    store.upload_file.side_effect = original
    result = sources(root, raw, store)
    assert result['status'] == 'published'
    assert store.upload_file.call_count == len(objects)


def test_missing_original_zip_does_not_fall_back_to_csv(setup):
    root, raw, _, store, _ = setup
    path = add_zip(root, raw);path.rename(path.with_suffix('.preserved'))
    with pytest.raises(FileNotFoundError):sources(root, raw, store)
    store.object_exists.assert_not_called()
    assert 'source_backup' not in entry(root)


def test_changed_source_refuses_backup_before_cloud(setup):
    root, raw, _, store, _ = setup
    next(raw.glob('*.csv')).write_text('changed')
    with pytest.raises(ValueError, match='Source changed'):sources(root, raw, store)
    store.object_exists.assert_not_called()


def test_cloud_resume_skips_etl_with_missing_local_working_files(setup, monkeypatch):
    root, raw, output, store, _ = setup
    outputs(root, store)
    output.rename(output.with_name('preserved-output'))
    monkeypatch.setattr(historical, 'discover_archive', Mock(side_effect=AssertionError('download attempted')))
    monkeypatch.setattr(historical, 'run_pipeline', Mock(side_effect=AssertionError('ETL attempted')))
    result = historical.process_month('2025-02', raw, root, cloud_store=store)
    assert result['status'] == 'skipped_cloud_verified'
    store.upload_file.reset_mock();store.download_file.reset_mock()
    verify_cloud_partition(entry(root), store)
    store.upload_file.assert_not_called();store.download_file.assert_not_called()
    with pytest.raises(ValueError, match='Processed partition changed'):
        historical.process_month('2025-02', raw, root)


@pytest.mark.parametrize('bad', ['status', 'receipt', 'hash', 'generation', 'metadata', 'missing'])
def test_incomplete_or_changed_cloud_backups_never_resume(setup, bad):
    root, _, output, store, objects = setup
    outputs(root, store);e = entry(root)
    output.rename(output.with_name('preserved-output'))
    name = 'validated_trips.parquet';f = e['cloud_publication']['files'][name]
    key = f['cloud_path'].split('gs://bucket/',1)[1]
    if bad == 'status':e['cloud_publication']['status'] = 'failed'
    elif bad == 'receipt':f['verification_status'] = 'failed'
    elif bad == 'hash':f['sha256'] = '0'*64
    elif bad == 'generation':objects[key]['generation'] = '999'
    elif bad == 'metadata':objects[key]['metadata']['source_archive_month'] = '2024-01'
    else:store.object_info.side_effect = GCSOperationError('missing object')
    with pytest.raises((ValueError, GCSOperationError)):verify_cloud_partition(e, store)


def test_changed_local_survivor_is_not_hidden_by_cloud(setup):
    root, _, output, store, _ = setup
    outputs(root, store)
    (output / 'hourly_rides.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Local output changed'):
        verify_cloud_partition(entry(root), store)


@pytest.mark.parametrize('kind', ['outputs', 'sources'])
def test_restore_matches_bytes_and_reuses_only_matching_cache(setup, kind, tmp_path):
    root, raw, _, store, _ = setup
    outputs(root, store);sources(root, raw, store)
    e = entry(root);dest = tmp_path / 'restore'
    path = restore_partition(e, dest, store=store, kind=kind, reserve_bytes=0)
    receipts = output_receipts(e) if kind == 'outputs' else source_receipts(e)
    assert {p.name for p in path.iterdir()} == set(receipts)
    for name, f in receipts.items():assert sha256_file(path / name) == f['sha256']
    calls = store.download_file.call_count
    restore_partition(e, dest, store=store, kind=kind, reserve_bytes=0)
    assert store.download_file.call_count == calls
    (dest / next(iter(receipts))).write_bytes(b'collision')
    with pytest.raises(GCSOperationError, match='collision'):
        restore_partition(e, dest, store=store, kind=kind, reserve_bytes=0)
    assert (dest / next(iter(receipts))).read_bytes() == b'collision'


def test_partial_restore_is_retained_and_not_replaced(setup, tmp_path):
    root, _, _, store, _ = setup
    outputs(root, store)
    def partial(key, path, **kwargs):
        Path(path).write_bytes(b'partial')
        raise GCSOperationError('interrupted download')
    store.download_file.side_effect = partial
    with pytest.raises(GCSOperationError):
        restore_partition(entry(root), tmp_path / 'restore', store=store, reserve_bytes=0)
    assert next((tmp_path / 'restore').iterdir()).read_bytes() == b'partial'
    with pytest.raises(GCSOperationError, match='collision'):
        restore_partition(entry(root), tmp_path / 'restore', store=store, reserve_bytes=0)


def test_catalog_restores_required_inputs_without_etl(setup, tmp_path):
    root, _, output, store, _ = setup
    outputs(root, store)
    output.rename(output.with_name('preserved-output'))
    result = build_catalog(root, cloud_store=store, restore_root=tmp_path / 'cache', reserve_bytes=0)
    assert result['accepted_rows'] == 2
    assert result['duplicate_ride_ids'] == 0
    assert result['start_month_counts'] == {'2025-01': 1, '2025-02': 1}
    restored = [p.name for p in (tmp_path / 'cache').rglob('*') if p.is_file()]
    assert sorted(restored) == ['rejected_trips.csv','validated_trips.parquet']
    assert entry(root)['output_dir'] == str(output)


def test_retention_refuses_incomplete_and_metadata_only_evidence(setup):
    root, raw, _, store, _ = setup
    outputs(root, store)
    report = retention_plan(root, raw_root=raw, verify_local_hashes=True)
    assert report['recorded_backup_eligible_bytes'] == 0
    assert all(not f['cleanup_authorized'] for f in report['candidates'])
    sources(root, raw, store)
    report = retention_plan(root, raw_root=raw)
    assert report['recorded_backup_eligible_bytes'] == 0
    report = retention_plan(root, raw_root=raw, verify_local_hashes=True)
    assert report['recorded_backup_eligible_bytes'] > 0
    assert report['deletion_supported'] is False
    assert any(f['role'] == 'derived_database' and f['recorded_backup_eligible'] for f in report['candidates'])
    assert all(Path(f['path']).exists() for f in report['candidates'])


def test_retention_original_zip_covers_extracted_csvs(setup):
    root, raw, _, store, _ = setup
    add_zip(root, raw);outputs(root, store);sources(root, raw, store)
    report = retention_plan(root, raw_root=raw, verify_local_hashes=True)
    csv_files = [f for f in report['candidates'] if f['role']=='extracted_csv']
    assert csv_files and all(f['recorded_backup_eligible'] for f in csv_files)
    assert all('re-extract' in f['recovery_method'] for f in csv_files)


def test_cli_plan_does_not_import_cloud_or_modify_manifest(setup, monkeypatch, capsys):
    import builtins
    root, raw, _, _, _ = setup
    before = (root / 'manifest.json').read_bytes()
    original = builtins.__import__
    def blocked(name, *a, **k):
        if name.startswith(('google','citypulse_ai.cloud')):raise AssertionError('Plan imported cloud')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',blocked)
    storage_cli.main(['plan','--output-root',str(root),'--raw-root',str(raw)])
    assert json.loads(capsys.readouterr().out)['dry_run'] is True
    assert (root / 'manifest.json').read_bytes() == before


def test_cli_cloud_resume_uses_existing_partition_only(setup, monkeypatch):
    root, raw, output, store, _ = setup
    outputs(root, store);output.rename(output.with_name('preserved-output'))
    monkeypatch.setattr('citypulse_ai.cloud.gcs.GCSStorage',lambda *a,**k:store)
    monkeypatch.setattr(historical,'discover_archive',Mock(side_effect=AssertionError('download')))
    historical_cli.main(['--month','2025-02','--output-root',str(root),'--raw-root',str(raw),
                         '--cloud-resume','--cloud-project','project','--cloud-bucket','bucket'])
    assert entry(root)['status']=='processed'


def test_catalog_cloud_restore_preserves_cross_month_duplicate_audit(setup, tmp_path):
    root, _, output, store, _ = setup
    outputs(root, store)
    second_csv = tmp_path / 'march.csv'
    second_csv.write_text('ride_id,started_at,ended_at,start_station_id,end_station_id\n'
                          'feb,2025-03-01 01:00:00,2025-03-01 01:10:00,001.01,\n'
                          'unique,2025-03-02 01:00:00.123456789,2025-03-02 01:10:00.123456789,6569.09_,\n')
    second_output = tmp_path / 'march-output'
    run_pipeline(second_csv, second_output, chunk_size=1)
    historical.register_existing('2025-03', second_output, root)
    publish_month('2025-03', root, project='project', bucket='bucket', store=store)
    output.rename(output.with_name('preserved-output'))
    second_output.rename(second_output.with_name('preserved-march-output'))
    result = build_catalog(root, cloud_store=store, reserve_bytes=0)
    assert result['input_rows'] == 4
    assert result['accepted_rows'] == 2 and result['rejected_rows'] == 2
    assert result['duplicate_ride_ids'] == 1 and result['cross_archive_duplicate_ride_ids'] == 1
    assert result['start_month_counts'] == {'2025-01':1,'2025-03':1}
    assert result['duplicate_station_hour_keys'] == 0


@pytest.mark.parametrize('failure', ['permission', 'hash', 'collision'])
def test_source_backup_failure_never_marks_published_or_deletes(setup, failure):
    root, raw, output, store, objects = setup
    before = {str(p):p.read_bytes() for folder in [raw,output] for p in folder.rglob('*') if p.is_file()}
    if failure == 'permission':
        store.upload_file.side_effect = GCSOperationError('Forbidden')
    elif failure == 'hash':
        def corrupt(key,path,**kwargs):Path(path).write_bytes(b'corrupt');return path
        store.download_file.side_effect = corrupt
    else:
        original=store.upload_file.side_effect
        def collision(path,key,**kwargs):
            digest=original(path,key,**kwargs)
            objects[key]['metadata']['backup_role']='unrelated'
            return digest
        store.upload_file.side_effect=collision
    with pytest.raises(GCSOperationError):sources(root,raw,store)
    assert entry(root)['source_backup']['status']=='failed'
    assert entry(root)['status']=='processed'
    assert {p:Path(p).read_bytes() for p in before}==before


def test_retention_changed_source_is_never_eligible(setup):
    root,raw,_,store,_=setup
    outputs(root,store);sources(root,raw,store)
    path=next(raw.glob('*.csv'));path.write_text('changed')
    report=retention_plan(root,raw_root=raw,verify_local_hashes=True)
    candidate=next(f for f in report['candidates'] if Path(f['path'])==path)
    assert candidate['recorded_backup_eligible'] is False
    assert any('SHA-256 differs' in reason for reason in candidate['blockers'])


def test_source_restore_preserves_original_zip_bytes(setup,tmp_path):
    root,raw,_,store,_=setup
    original=add_zip(root,raw);sources(root,raw,store)
    restored=restore_partition(entry(root),tmp_path/'restore',store=store,kind='sources',reserve_bytes=0)
    assert (restored/original.name).read_bytes()==original.read_bytes()


def test_restore_checks_disk_before_downloading(setup,tmp_path,monkeypatch):
    root,_,_,store,_=setup
    outputs(root,store);store.download_file.reset_mock()
    def full(*a,**k):raise ValueError('Insufficient disk space')
    monkeypatch.setattr('citypulse_ai.cloud.retention.check_disk',full)
    with pytest.raises(ValueError,match='disk space'):
        restore_partition(entry(root),tmp_path/'restore',store=store)
    store.download_file.assert_not_called()


def test_all_original_csv_parts_are_required_and_backed_up(setup):
    root,raw,_,store,_=setup
    m=historical.load_manifest(root)
    for number in [2,3]:
        path=raw/f'202502-citibike-tripdata_{number}.csv'
        path.write_text('ride_id,started_at,ended_at,start_station_id,end_station_id\n')
        m['archives']['2025-02']['csv_files'].append({'input_file':str(path),'input_sha256':sha256_file(path),'input_rows':0})
    write_json(root/'manifest.json',m)
    result=sources(root,raw,store)
    originals=[name for name,f in result['artifacts'].items() if f['role']=='original_csv']
    assert len(originals)==3
    assert all(name in result['files'] for name in originals)
    incomplete=entry(root)
    incomplete['source_backup']['files'].pop(originals[-1])
    with pytest.raises(ValueError,match='Missing'):
        source_receipts(incomplete)


def test_storage_operations_do_not_delete_any_local_file(setup,tmp_path,monkeypatch):
    root,raw,output,store,_=setup
    before=[p for folder in [raw,output] for p in folder.rglob('*') if p.is_file()]
    def forbidden(*a,**k):raise AssertionError('Unexpected deletion')
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',forbidden)
        sources(root,raw,store);outputs(root,store)
        retention_plan(root,raw_root=raw,verify_local_hashes=True)
        restore_partition(entry(root),tmp_path/'cache',store=store,reserve_bytes=0)
    assert all(p.exists() for p in before)


def test_retention_reports_unprocessed_month_without_claiming_recoverability(setup):
    root,raw,_,_,_=setup
    m=historical.load_manifest(root)
    m['archives']['2025-03']={'month':'2025-03','status':'interrupted','attempts':[]}
    write_json(root/'manifest.json',m)
    plan=retention_plan(root,raw_root=raw)
    assert plan['month_blockers']['2025-03']
    assert plan['recorded_backup_eligible_bytes']==0


def test_standalone_hourly_receipt_does_not_enable_partition_cleanup(setup):
    root,raw,output,_,_=setup
    path=output/'hourly_rides.parquet'
    folder=root.parent/'gcs-publications';folder.mkdir()
    write_json(folder/'hourly.json',{'uploaded':True,'verified':True,'source_path':str(path),
              'sha256':sha256_file(path),'destination':'gs://bucket/hourly.parquet',
              'checks':{'sha256_match':True,'schema_match':True,'row_count_match':True}})
    report=retention_plan(root,raw_root=raw,verify_local_hashes=True)
    candidate=next(f for f in report['candidates'] if Path(f['path'])==path)
    assert candidate['role']=='standalone_published_output'
    assert candidate['cloud_path']=='gs://bucket/hourly.parquet'
    assert candidate['recorded_backup_eligible'] is False


def test_retention_includes_verified_download_copies_but_keeps_control_metadata(setup):
    root,raw,_,store,_=setup
    outputs(root,store);sources(root,raw,store)
    report=retention_plan(root,raw_root=raw,verify_local_hashes=True)
    copies=[f for f in report['candidates'] if f['role']=='verification_copy']
    assert copies and all(f['recorded_backup_eligible'] for f in copies)
    paths={f['path'] for f in report['candidates']}
    assert str(root/'manifest.json') not in paths
    snapshot=entry(root)['source_backup']['artifacts']['processing_manifest.json']['path']
    assert snapshot not in paths
