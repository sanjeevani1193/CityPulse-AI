"""Historical ingestion tests are synthetic and offline; no real downloads required."""

import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import duckdb
import pyarrow.parquet as pq
import pytest

from citypulse_ai.etl import archives, historical
from citypulse_ai.etl.history_catalog import build_catalog
from citypulse_ai.etl.pipeline import run_pipeline

HEADER = ['ride_id','started_at','ended_at','start_station_id','end_station_id']


def make_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def ride(identifier, start='2025-03-01 00:00:00.123456789', end='2025-03-01 00:01:00.123456789', station='001.01'):
    return [identifier,start,end,station,'']


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status=status
        self.headers=headers or {}


def test_calendar_month_range_is_exclusive():
    assert archives.month_range('2024-01-01','2026-01-01') == [f'{year}-{month:02d}' for year in [2024,2025] for month in range(1,13)]
    with pytest.raises(ValueError):
        archives.month_range('2024-01-02','2026-01-01')
    with pytest.raises(ValueError):
        archives.month_range('2026-01-01','2024-01-01')
    with pytest.raises(ValueError):
        archives.validate_month('2024-1')


def test_discovery_ignores_jersey_city_and_nonmatching_keys(monkeypatch):
    xml=b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Contents><Key>202401-citibike-tripdata.zip</Key><Size>123</Size><ETag>"version"</ETag><LastModified>2026-01-01</LastModified></Contents><Contents><Key>202401-other.zip</Key><Size>5</Size></Contents></ListBucketResult>'
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Response(xml))
    source=archives.discover_archive('2024-01')
    assert source['url']=='https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.zip'
    assert source['bytes']==123 and source['etag']=='"version"'


def test_download_resumes_without_appending_full_response(tmp_path,monkeypatch):
    data=b'0123456789'
    source={'key':'test.zip','url':'https://s3.amazonaws.com/tripdata/test.zip','bytes':10,'etag':'"x"'}
    directory=tmp_path/'raw'
    directory.mkdir()
    (directory/'test.zip.part').write_bytes(data[:3])
    archives.write_json(directory/'test.zip.part.json',{k:source[k] for k in ['url','etag','bytes']})
    requests=[]
    def open_range(request,**kwargs):
        requests.append(request)
        assert request.get_header('Range')=='bytes=3-'
        assert request.get_header('If-match')=='"x"'
        return Response(data[3:],206,{'ETag':'"x"','Content-Range':'bytes 3-9/10'})
    monkeypatch.setattr(urllib.request,'urlopen',open_range)
    path,digest=archives.download_archive(source,directory)
    assert path.read_bytes()==data and digest==hashlib.sha256(data).hexdigest()
    archives.download_archive(source,directory)
    assert len(requests)==1  # Locally verified completed download is reused.
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA-256'):
        archives.download_archive(source,directory)


def test_ignored_range_and_changed_identity_preserve_partial(tmp_path,monkeypatch):
    source={'key':'test.zip','url':'https://s3.amazonaws.com/tripdata/test.zip','bytes':10,'etag':'"x"'}
    partial=tmp_path/'test.zip.part'
    partial.write_bytes(b'012')
    archives.write_json(tmp_path/'test.zip.part.json',{k:source[k] for k in ['url','etag','bytes']})
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Response(b'0123456789',200,{'ETag':'"x"'}))
    with pytest.raises(ValueError,match='range'):
        archives.download_archive(source,tmp_path)
    assert partial.read_bytes()==b'012'
    with pytest.raises(ValueError,match='identity changed'):
        archives.download_archive({**source,'etag':'"new"'},tmp_path)


def test_safe_nested_csv_extraction_and_reuse(tmp_path):
    path=tmp_path/'archive.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('folder/part_1.csv','a,b\n1,2\n')
        archive.writestr('part_2.csv','a,b\n3,4\n')
        archive.writestr('__MACOSX/._part_2.csv','resource fork')
    files=archives.extract_archive(path,tmp_path/'extracted')
    assert len(files)==2
    assert files==archives.extract_archive(path,tmp_path/'extracted')
    Path(files[0]['path']).write_text('tampered')
    with pytest.raises(ValueError,match='Unverified'):
        archives.extract_archive(path,tmp_path/'extracted')


@pytest.mark.parametrize('member',['../escape.csv','/escape.csv','folder\\escape.csv'])
def test_zip_path_traversal_is_rejected(tmp_path,member):
    path=tmp_path/'archive.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr(member,'a,b\n1,2\n')
    with pytest.raises(ValueError,match='Unsafe'):
        archives.extract_archive(path,tmp_path/'extracted')


def test_zip_crc_failure_never_publishes_csv(tmp_path):
    path=tmp_path/'archive.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('data.csv','unique payload\n')
    data=path.read_bytes().replace(b'unique payload',b'broken payload')
    path.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile):
        archives.extract_archive(path,tmp_path/'extracted')
    assert not (tmp_path/'extracted/data.csv').exists()


def test_low_disk_fails_before_download(tmp_path,monkeypatch):
    monkeypatch.setattr(archives.shutil,'disk_usage',lambda p:type('Disk',(),{'free':10})())
    with pytest.raises(OSError,match='Insufficient disk'):
        archives.check_disk(tmp_path,9,2)


def fake_official_archive(tmp_path,monkeypatch):
    raw=make_csv(tmp_path/'source.csv',[ride('one'),ride('two')])
    zipped=tmp_path/'source.zip'
    with zipfile.ZipFile(zipped,'w') as z:
        z.write(raw,'nested/part.csv')
    source={'month':'2025-03','key':'202503-citibike-tripdata.zip','url':'https://s3.amazonaws.com/tripdata/202503-citibike-tripdata.zip','bytes':zipped.stat().st_size,'etag':'"fake"'}
    monkeypatch.setattr(historical,'discover_archive',lambda m:source)
    monkeypatch.setattr(historical,'inspect_archive',lambda s:{'csv_bytes':raw.stat().st_size,'members':[]})
    monkeypatch.setattr(historical,'download_archive',lambda s,d:(zipped,archives.fingerprint(zipped)))
    return source


def test_samples_separate_full_skip_and_atomic_resume(tmp_path,monkeypatch):
    fake_official_archive(tmp_path,monkeypatch)
    root=tmp_path/'outputs'
    sample=historical.process_month('2025-03',tmp_path/'raw',root,sample_rows=1,reserve_bytes=0)
    assert sample['input_rows']==1 and historical.load_manifest(root)['archives']['2025-03']['status']=='sample_processed'
    with pytest.raises(ValueError,match='No complete'):
        build_catalog(root,reserve_bytes=0)
    full=historical.process_month('2025-03',tmp_path/'raw',root,reserve_bytes=0)
    assert full['input_rows']==2 and full['verification']['ride_counts_equal_accepted']
    assert historical.process_month('2025-03',tmp_path/'raw',root,reserve_bytes=0)['status']=='skipped_verified'
    assert historical.process_month('2025-03',tmp_path/'raw',root,sample_rows=100,reserve_bytes=0)['status']=='skipped_verified'
    manifest=historical.load_manifest(root)
    entry=manifest['archives']['2025-03']
    entry['status']='running'  # Simulate interruption after atomic directory publication.
    archives.write_json(root/'manifest.json',manifest)
    assert historical.process_month('2025-03',tmp_path/'raw',root,reserve_bytes=0)['status']=='resumed_verified'
    assert (root/'samples/2025-03-n1/report.json').exists()


def test_failed_run_records_error_and_can_restart(tmp_path,monkeypatch):
    fake_official_archive(tmp_path,monkeypatch)
    root=tmp_path/'outputs'
    original=historical.run_pipeline
    def fail(*a,**k):
        raise RuntimeError('simulated failure')
    monkeypatch.setattr(historical,'run_pipeline',fail)
    with pytest.raises(RuntimeError):
        historical.process_month('2025-03',tmp_path/'raw',root,reserve_bytes=0)
    assert historical.load_manifest(root)['archives']['2025-03']['attempts'][-1]['error']=='RuntimeError: simulated failure'
    assert not (root/'source_month=2025-03').exists()
    monkeypatch.setattr(historical,'run_pipeline',original)
    assert historical.process_month('2025-03',tmp_path/'raw',root,reserve_bytes=0)['status']=='processed'


def test_global_duplicate_rejections_boundaries_and_nanoseconds(tmp_path):
    root=tmp_path/'outputs'
    jan=run_pipeline(make_csv(tmp_path/'jan.csv',[
        ride('boundary','2023-12-31 23:59:59.123456789','2024-01-01 00:01:00.123456789'),
        ride('duplicate'),ride('january','2024-01-01 00:00:00','2024-01-01 00:01:00'),
    ]),tmp_path/'jan')
    march=run_pipeline(make_csv(tmp_path/'march.csv',[
        ride('duplicate'),ride('february','2025-02-28 23:59:59.123456789','2025-03-01 00:01:00.123456789'),
        ride('march'),ride('bad-end','2025-03-01 00:00:00','bad'),
    ]),tmp_path/'march')
    historical.register_existing('2024-01',tmp_path/'jan',root)
    historical.register_existing('2025-03',tmp_path/'march',root)
    result=build_catalog(root,reserve_bytes=0,chunk_size=2)
    assert result['input_rows']==7 and result['accepted_rows']==4 and result['rejected_rows']==3
    assert result['cross_archive_duplicate_ride_ids']==1 and result['additional_cross_archive_rejections']==2
    assert result['hourly_ride_starts']==4 and result['duplicate_station_hour_keys']==0
    assert result['out_of_window_ride_starts']==1
    assert result['start_month_counts']=={'2023-12':1,'2024-01':1,'2025-02':1,'2025-03':1}
    assert not result['january_2026_boundary_input_present']
    path=Path(result['snapshot'])/'trips/start_month=2025-02/validated_trips.parquet'
    table=pq.ParquetFile(path).read()
    assert str(table.schema.field('started_at').type)=='timestamp[ns, tz=UTC]'
    assert table.column('started_at').cast('int64').to_pylist()==table.column('started_at_utc_ns').to_pylist()
    assert table.column('source_archive_month').to_pylist()==['2025-03']
    import pandas as pd
    assert pd.read_parquet(path).ride_id.tolist()==['february']
    with duckdb.connect(str(Path(result['snapshot'])/'catalog.duckdb'),read_only=True) as connection:
        assert connection.execute('SELECT count(*) FROM canonical_rejections').fetchone()[0]==3
    again=build_catalog(root,reserve_bytes=0)
    assert again['snapshot']!=result['snapshot'] and path.exists()
    assert json.loads((tmp_path/'jan/report.json').read_text())==jan
    assert json.loads((tmp_path/'march/report.json').read_text())==march


def test_valid_id_also_in_invalid_archive_row_is_globally_rejected(tmp_path):
    root=tmp_path/'outputs'
    run_pipeline(make_csv(tmp_path/'one.csv',[ride('shared')]),tmp_path/'one')
    run_pipeline(make_csv(tmp_path/'two.csv',[ride('shared',end='bad')]),tmp_path/'two')
    historical.register_existing('2025-02',tmp_path/'one',root)
    historical.register_existing('2025-03',tmp_path/'two',root)
    result=build_catalog(root,reserve_bytes=0)
    assert result['accepted_rows']==0 and result['rejected_rows']==2
    assert result['additional_cross_archive_rejections']==1


def test_single_writer_prevents_manifest_races(tmp_path):
    from citypulse_ai.etl.historical_cli import single_writer
    with single_writer(tmp_path):
        with pytest.raises(RuntimeError,match='Another ingestion'):
            with single_writer(tmp_path):
                pytest.fail('Second writer acquired the lock')
    with single_writer(tmp_path):
        pass  # Interrupted/released commands do not leave a stale lock.


def test_boundary_probe_uses_local_start_month_and_is_only_a_sample(tmp_path,monkeypatch):
    raw=make_csv(tmp_path/'boundary.csv',[
        ride('previous','2025-12-31 23:59:00','2026-01-01 00:01:00'),
        ride('offset','2026-01-01T04:59:00Z','2026-01-01T05:01:00Z'),
        ride('current','2026-01-01 00:01:00','2026-01-01 00:02:00'),
    ])
    zipped=tmp_path/'archive.zip'
    with zipfile.ZipFile(zipped,'w') as archive:
        archive.write(raw,'data.csv')
    monkeypatch.setattr(archives,'discover_archive',lambda month:{'month':month})
    monkeypatch.setattr(archives,'RemoteArchive',lambda source:io.BytesIO(zipped.read_bytes()))
    report=archives.probe_boundary_archive('2026-01')
    assert report['members'][0]['start_month_counts']=={'2025-12':2,'2026-01':1}
    assert len(report['members'][0]['boundary_examples'])==2
    assert 'Not a full archive count' in report['limitations']


def test_storage_plan_checks_existing_parent_of_configured_roots(tmp_path,monkeypatch):
    monkeypatch.setattr(historical,'discover_archive',lambda m:{'month':m,'bytes':100})
    monkeypatch.setattr(historical,'inspect_archive',lambda s:{'csv_bytes':200,'members':[]})
    plan=historical.plan_months(['2024-01'],tmp_path/'new-raw',tmp_path/'new-outputs',reserve_bytes=0)
    assert plan['raw_filesystem']['existing_parent']==str(tmp_path)
    assert plan['output_filesystem']['existing_parent']==str(tmp_path)
    assert plan['same_filesystem']
    assert plan['estimated_retained_and_staging_bytes']==1900


def test_january_boundary_input_does_not_extend_fixed_model_window(tmp_path):
    source=tmp_path/'jan2026'
    run_pipeline(make_csv(tmp_path/'jan2026.csv',[
        ride('december','2025-12-31 23:59:00.123456789','2026-01-01 00:01:00.123456789'),
        ride('january','2026-01-01 00:00:00.123456789','2026-01-01 00:01:00.123456789'),
    ]),source)
    root=tmp_path/'outputs'
    historical.register_existing('2026-01',source,root)
    result=build_catalog(root,reserve_bytes=0)
    assert result['january_2026_boundary_input_present']
    assert result['in_window_ride_starts']==1 and result['out_of_window_ride_starts']==1
    assert result['start_month_counts']=={'2025-12':1,'2026-01':1}
    with duckdb.connect(str(Path(result['snapshot'])/'catalog.duckdb'),read_only=True) as connection:
        assert connection.execute('SELECT sum(ride_count) FROM model_window_hourly_rides').fetchone()[0]==1


def test_dst_elapsed_hours_remain_distinct_and_nonexistent_times_rejected(tmp_path):
    source=tmp_path/'dst'
    report=run_pipeline(make_csv(tmp_path/'dst.csv',[
        ride('spring','2025-03-09 01:59:59','2025-03-09 03:00:59'),
        ride('nonexistent','2025-03-09 02:10:00','2025-03-09 03:20:00'),
        ride('fall-first','2025-11-02T01:10:00-04:00','2025-11-02T01:20:00-04:00'),
        ride('fall-second','2025-11-02T01:10:00-05:00','2025-11-02T01:20:00-05:00'),
    ]),source)
    assert report['accepted_rows']==3 and report['rejected_rows']==1
    root=tmp_path/'outputs'
    historical.register_existing('2025-03',source,root)
    result=build_catalog(root,reserve_bytes=0)
    assert result['accepted_rows']==3 and result['hourly_rows']==3
    with duckdb.connect(str(Path(result['snapshot'])/'catalog.duckdb'),read_only=True) as connection:
        local=connection.execute("SELECT hour_start AT TIME ZONE 'America/New_York' FROM hourly_rides WHERE start_month='2025-11'").fetchall()
        assert len(local)==2 and local[0][0]==local[1][0]  # Same clock label, distinct elapsed-hour keys.
        assert connection.execute("SELECT duration_seconds FROM canonical_trips WHERE ride_id='spring'").fetchone()[0]==60
