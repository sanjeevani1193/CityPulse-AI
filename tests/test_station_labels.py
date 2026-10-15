"""Label evidence, uncertainty and causality; tiny synthetic inputs only."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from citypulse_ai.features.station_dataset import build_station_panel, make_window
from citypulse_ai.features.station_labels import positive_decisions, rejection_hours, review_positive_labels
from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.trips import validate_trips


def counts(rows):
    frame = pd.DataFrame(rows, columns=['station_id','hour_start_utc','ride_count_accepted'])
    frame['hour_start_utc'] = pd.to_datetime(frame.hour_start_utc, utc=True)
    return frame


def decide(a, b=None, **kw):
    return positive_decisions(a, a.copy() if b is None else b, stations=['001_'],
        start=pd.Timestamp('2024-07-01T04:00Z'), end=pd.Timestamp('2024-07-01T07:00Z'),
        blocked_hours=kw.get('blocked', set()), unlocalizable=kw.get('unknown', 0), evidence_id='pinned')


def test_verified_positive_not_absent_zero_and_status_stays_unknown():
    a=counts([('001_','2024-07-01T04:00Z',7)])
    d=decide(a)
    p=build_station_panel(a, {'001_':1}, start=d.hour_start_utc.min(),
        end=d.hour_start_utc.max()+pd.Timedelta(hours=1), quality=d)
    assert p.label_observed_mask.tolist()==[True,False,False]
    assert p.ride_count.iloc[0]==7 and p.ride_count.iloc[1:].isna().all()
    assert not p.availability_known_mask.any()
    assert d.reason.tolist()==['verified_positive_archived_accepted']+['absent_row_no_zero_certificate']*2


def test_missing_station_rejection_blocks_every_selected_station_in_hour():
    a=counts([('001_','2024-07-01T04:00Z',7)])
    d=decide(a,blocked={pd.Timestamp('2024-07-01T04:00Z')})
    assert not d.count_complete.any()
    assert d.reason.iloc[0]=='rejected_start_hour_station_uncertain'


def test_unlocalizable_rejection_masks_whole_slice():
    d=decide(counts([('001_','2024-07-01T04:00Z',7)]),unknown=1)
    assert not d.count_complete.any()
    assert set(d.reason)=={'unlocalizable_rejection_scope'}


def test_empty_station_never_becomes_zero_and_timestamp_units_do_not_change_counts():
    d=decide(counts([]))
    assert len(d)==3 and not d.count_complete.any()
    a=counts([('001_','2024-07-01T04:00Z',7)])
    b=a.copy();b['hour_start_utc']=b.hour_start_utc.dt.as_unit('us')
    assert decide(a,b).count_complete.tolist()==[True,False,False]


@pytest.mark.parametrize('other', [[], [('001_','2024-07-01T04:00Z',6)], [('001_','2024-07-01T05:00Z',7)]])
def test_recount_mismatch_fails_instead_of_masking(other):
    with pytest.raises(ValueError,match='differs'):
        decide(counts([('001_','2024-07-01T04:00Z',7)]),counts(other))


def test_ambiguous_rejection_masks_both_dst_folds_without_guessing():
    hours,n=rejection_hours(['2024-11-03 01:30:00.123456789'])
    assert n==0 and hours=={pd.Timestamp('2024-11-03T05:00Z'),pd.Timestamp('2024-11-03T06:00Z')}
    h,n=rejection_hours(['2024-03-10 02:30:00','bad',''])
    assert n==3 and not h


def test_timestamp_precision_offsets_and_boundary_rejections():
    h,n=rejection_hours(['2024-06-30 23:59:59.999999999','2024-07-01T00:00:00-04:00'])
    assert n==0 and h=={pd.Timestamp('2024-07-01T03:00Z'),pd.Timestamp('2024-07-01T04:00Z')}


def test_decisions_are_deterministic_and_future_labels_do_not_change_encoder():
    o=pd.Timestamp('2024-07-01T04:00Z');start=o-pd.Timedelta(hours=336)
    hours=pd.date_range(start,periods=360,freq='h')
    a=counts([('001_',h,3) for h in hours])
    def panel(data):
        d=positive_decisions(data,data.copy(),stations=['001_'],start=start,
            end=o+pd.Timedelta(hours=24),blocked_hours=set(),unlocalizable=0,evidence_id='pinned')
        return build_station_panel(data,{'001_':1},start=start,end=o+pd.Timedelta(hours=24),quality=d)
    p=panel(a);q=panel(a.sample(frac=1,random_state=2));pd.testing.assert_frame_equal(p,q)
    def w(p):return make_window(p,'001_',o,split_start=o,split_end=o+pd.Timedelta(hours=24))
    first=w(p);a.loc[a.hour_start_utc.ge(o),'ride_count_accepted']=999
    changed=w(panel(a))
    for key in ['encoder_known','encoder_observed','decoder_known']:
        np.testing.assert_array_equal(getattr(first,key),getattr(changed,key))


@pytest.fixture
def history(tmp_path):
    """Pinned local input fixture; StationHistory integrity itself has separate tests."""
    raw=pd.DataFrame({'ride_id':['exact0001','exact0002','badstation'],
        'started_at':['2024-06-30 23:59:59.123456789','2024-07-01 00:05:00','2024-07-01 01:00:00'],
        'ended_at':['2024-07-01 00:10:00','2024-07-01 00:15:00','2024-07-01 01:10:00'],
        'start_station_id':['001_','001_',''],'end_station_id':['','','']})
    raw['start_station_id']=['6569.09_','6569.09_','']
    result=validate_trips(raw)
    valid=result.valid.copy();valid['source_file']='member.csv';valid['source_row']=[2,3]
    for col in ['started_at','ended_at']:
        valid[col+'_utc_ns']=valid[col].astype('int64')
    rejected=result.rejected.copy();rejected['source_file']='member.csv'
    valid.to_parquet(tmp_path/'validated_trips.parquet',index=False)
    rejected.to_csv(tmp_path/'rejected_trips.csv',index=False)
    entry={'status':'processed','month':'2024-07','output_dir':str(tmp_path),
        'input_rows':3,'accepted_rows':2,'rejected_rows':1,
        'output_sha256':{n:fingerprint(tmp_path/n) for n in ['validated_trips.parquet','rejected_trips.csv']},
        'source':{'members':[{'name':'member.csv'}]},
        'csv_files':[{'member':'member.csv','path':'member.csv','sha256':'source-hash'}],
        'schemas':[{'member':'member.csv'}],
        'verification':{'input_equals_accepted_plus_rejected':True,'ride_counts_equal_accepted':True,
                        'timestamp_epoch_mismatches':0},
        'source_backup':{'revision':'source-revision'},'cloud_publication':{'revision':'output-revision'}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'archives':{'2024-07':entry}}))
    def hourly(stations,start,end):
        return counts([('6569.09_',t,1) for t in pd.to_datetime(['2024-07-01T03:00Z','2024-07-01T04:00Z'])
                       if start<=t<end])
    return SimpleNamespace(manifest_path=path,audit_path=tmp_path/'audit.json',identity='verified',
        audit_sha256='audit-hash',manifest_sha256=fingerprint(path),
        _unchanged=lambda:None,verify_local_months=lambda months:{m:2 for m in months},
        report={'duplicate_ride_ids':0,'cross_archive_duplicate_ride_ids':0},hourly=hourly)


def review(h):
    return review_positive_labels(h,['6569.09_'],start=pd.Timestamp('2024-07-01T03:00Z'),
                                  end=pd.Timestamp('2024-07-01T06:00Z'))


def test_read_only_lineage_boundary_and_exact_counts(history):
    before={p:p.read_bytes() for p in history.manifest_path.parent.iterdir() if p.is_file()}
    r=review(history)
    assert r.decisions.count_complete.tolist()==[True,True,False]
    assert r.lineage.source_archive_month.tolist()==['2024-07','2024-07']
    assert r.evidence['zero_labels_certified']==0
    assert r.evidence['finalized_accepted_local_months']=={'2024-06':2,'2024-07':2}
    assert r.evidence==review(history).evidence
    assert all(p.read_bytes()==v for p,v in before.items())


def test_nonzero_duplicates_fail_closed(history):
    history.report['duplicate_ride_ids']=1
    with pytest.raises(ValueError,match='exclusion-aware'):review(history)


def test_changed_processed_input_fails_closed(history):
    p=history.manifest_path.parent/'validated_trips.parquet';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='changed'):review(history)


def test_missing_member_evidence_fails_closed(history):
    m=json.loads(history.manifest_path.read_text());m['archives']['2024-07']['schemas']=[]
    history.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='evidence incomplete'):review(history)


def test_legacy_noncontributing_source_not_mistaken_for_verified_archive_inventory(history):
    m=json.loads(history.manifest_path.read_text())
    entry=json.loads(json.dumps(m['archives']['2024-07']))
    entry['csv_files']=[{'input_file':'member.csv','input_sha256':'source-hash'}]
    entry.pop('schemas');entry['source']={}
    # Same tiny processed records are outside this explicitly different slice.
    m['archives']={'2025-02':entry};history.manifest_path.write_text(json.dumps(m))
    history.hourly=lambda *args:counts([])
    r=review_positive_labels(history,['6569.09_'],start=pd.Timestamp('2024-01-01T05:00Z'),
                              end=pd.Timestamp('2024-01-01T06:00Z'))
    assert not r.decisions.count_complete.any()
    assert r.evidence['sources']['2025-02']['archive_member_inventory_verified'] is False
    # An archive required by the actual reviewed interval cannot use that path.
    with pytest.raises(ValueError,match='evidence incomplete'):
        review_positive_labels(history,['6569.09_'],start=pd.Timestamp('2025-02-01T05:00Z'),
                               end=pd.Timestamp('2025-02-01T06:00Z'))


def test_explicit_review_row_budget(history):
    with pytest.raises(ValueError,match='budget'):
        review_positive_labels(history,['6569.09_'],start=pd.Timestamp('2024-07-01T03:00Z'),
            end=pd.Timestamp('2024-07-01T06:00Z'),max_rows=2)
