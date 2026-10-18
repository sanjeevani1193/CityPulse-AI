"""Small synthetic closed-source-set certificates; no real history or cloud."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_station_labels import history
from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.accepted_recorded_labels import AcceptedRecordedStarts, ACCEPTED_RECORDED_POLICY
from citypulse_ai.features.station_dataset import build_station_panel, make_window


START = pd.Timestamp('2024-07-01T04:00Z')
END = pd.Timestamp('2024-07-01T07:00Z')


@pytest.fixture
def complete(history):
    """Use the existing pinned-input fixture plus an empty adjacent archive.

    StationHistory's real checkpoint/hash verification has separate reader tests;
    this fixture substitutes that reader contract while exercising actual Parquet
    recounts, rejected CSVs, grid comparison and certification logic.
    """
    manifest = json.loads(history.manifest_path.read_text())
    july = manifest['archives']['2024-07']
    folder = history.manifest_path.parent/'august';folder.mkdir()
    pd.read_parquet(Path(july['output_dir'])/'validated_trips.parquet').iloc[:0].to_parquet(
        folder/'validated_trips.parquet', index=False)
    pd.read_csv(Path(july['output_dir'])/'rejected_trips.csv').iloc[:0].to_csv(folder/'rejected_trips.csv', index=False)
    august = json.loads(json.dumps(july));august.update(output_dir=str(folder), month='2024-08',
        input_rows=0, accepted_rows=0, rejected_rows=0)
    august['output_sha256'] = {n: fingerprint(folder/n) for n in ['validated_trips.parquet','rejected_trips.csv']}
    manifest['archives']['2024-08'] = august
    history.manifest_path.write_text(json.dumps(manifest))
    history._query = lambda sql, parameters: pd.DataFrame({'station_id':['6569.09_'], 'training_starts':[2]})
    return history


def policy(h):
    return AcceptedRecordedStarts(h, {'6569.09_':1}, cohort_start=pd.Timestamp('2024-01-01T05:00Z'),
                                  cohort_end=pd.Timestamp('2024-07-01T04:00Z'))


def test_positive_zero_and_rejected_unknown_station_do_not_change_accepted_counts(complete):
    p=policy(complete);r=p.review(start=START,end=END)
    assert r.decisions.ride_count.tolist()==[1,0,0]
    assert r.decisions.count_complete.all() and r.evidence['zero_keys']==2
    assert r.evidence['policy']['version']=='accepted_recorded_starts_v1'
    assert r.diagnostics.rejected_record_count.tolist()==[1]
    assert r.diagnostics.hour_start_utc.iloc[0]==START+pd.Timedelta(hours=1)
    panel=build_station_panel(r.observed,p.codes,start=START,end=END,quality=r.quality)
    assert panel.ride_count.tolist()==[1,0,0]
    assert panel.label_observed_mask.all() and not panel.availability_known_mask.any()
    assert panel.ride_count_accepted.isna().sum()==2  # Original sparse provenance remains distinct.
    assert not any('rejected' in c for c in panel.columns)
    assert ACCEPTED_RECORDED_POLICY['publisher_completeness']=='not_established'


def test_deterministic_ids_and_no_mutation_of_existing_inputs(complete):
    before={p:p.read_bytes() for p in complete.manifest_path.parent.rglob('*') if p.is_file()}
    p=policy(complete);a=p.review(start=START,end=END);b=p.review(start=START,end=END)
    pd.testing.assert_frame_equal(a.decisions,b.decisions)
    assert a.evidence==b.evidence and all(p.read_bytes()==v for p,v in before.items())


@pytest.mark.parametrize('problem',['unfinalized','missing_archive'])
def test_missing_boundary_or_period_never_creates_zeros(complete,problem):
    p=policy(complete)
    def unavailable(months):raise ValueError(problem)
    complete.verify_local_months=unavailable
    with pytest.raises(ValueError,match=problem):p.review(start=START,end=END)


def test_missing_required_inventory_fails_closed(complete):
    p=policy(complete);m=json.loads(complete.manifest_path.read_text())
    m['archives']['2024-08']['schemas']=[];complete.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='evidence incomplete'):p.review(start=START,end=END)


def test_missing_actual_adjacent_source_entry_fails_closed(complete):
    p=policy(complete);m=json.loads(complete.manifest_path.read_text())
    del m['archives']['2024-08'];complete.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='inventory coverage'):p.review(start=START,end=END)


def test_frozen_cohort_identity_cannot_change_between_reviews(complete):
    p=policy(complete)
    with pytest.raises(TypeError):p.codes['newstation']=2
    p.cohort['station_codes']['newstation']=2
    with pytest.raises(ValueError,match='identity changed'):p.review(start=START,end=END)


@pytest.mark.parametrize('problem',['provenance','checkpoint'])
def test_reader_integrity_errors_block_certification(complete,problem):
    p=policy(complete)
    def corrupt():raise ValueError(problem)
    complete._unchanged=corrupt
    with pytest.raises(ValueError,match=problem):p.review(start=START,end=END)


def test_recount_mismatch_does_not_silently_make_zero(complete):
    p=policy(complete)
    complete.hourly=lambda *args: pd.DataFrame({'station_id':pd.Series(dtype=str),
        'hour_start_utc':pd.Series(dtype='datetime64[ns, UTC]'), 'ride_count_accepted':pd.Series(dtype=int)})
    with pytest.raises(ValueError,match='differs'):p.review(start=START,end=END)


def test_nonzero_duplicate_audit_requires_exclusion_aware_recount(complete):
    p=policy(complete);complete.report['duplicate_ride_ids']=1
    with pytest.raises(ValueError,match='exclusion-aware'):p.review(start=START,end=END)


def test_invalid_or_unseen_station_cannot_receive_zero_labels(complete):
    with pytest.raises(ValueError,match='membership'):
        AcceptedRecordedStarts(complete, {'fictional':1}, cohort_start=pd.Timestamp('2024-01-01T05:00Z'),cohort_end=START)
    with pytest.raises(ValueError,match='exact station'):
        AcceptedRecordedStarts(complete, {'':1}, cohort_start=pd.Timestamp('2024-01-01T05:00Z'),cohort_end=START)


def test_changed_processed_input_and_explicit_row_budget_fail(complete):
    p=policy(complete)
    with pytest.raises(ValueError,match='budget'):p.review(start=START,end=END,max_rows=2)
    path=complete.manifest_path.parent/'validated_trips.parquet'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):p.review(start=START,end=END)


def test_unlocalizable_rejected_timestamp_is_diagnostic_not_a_label_mask(complete):
    folder=complete.manifest_path.parent
    path=folder/'rejected_trips.csv';rows=pd.read_csv(path,dtype=str,keep_default_na=False)
    rows['started_at']='bad';rows['rejection_reasons']='started_at:invalid_timestamp';rows.to_csv(path,index=False)
    m=json.loads(complete.manifest_path.read_text())
    m['archives']['2024-07']['output_sha256'][path.name]=fingerprint(path)
    complete.manifest_path.write_text(json.dumps(m))
    r=policy(complete).review(start=START,end=END)
    assert r.decisions.count_complete.all()
    assert r.evidence['rejection_diagnostics']['unlocalizable_rejected_records_all_sources']==1
    assert r.diagnostics.empty


@pytest.mark.parametrize('date',['2024-03-10','2024-11-03'])
def test_certified_grid_dst_and_future_labels_never_enter_inputs(date):
    # Exercise the same positive/absent-row grid mechanics across both clock changes.
    o=pd.Timestamp(date,tz='America/New_York').tz_convert('UTC')
    hours=pd.date_range(o-pd.Timedelta(hours=336),periods=360,freq='h')
    observed=pd.DataFrame({'station_id':'001_', 'hour_start_utc':hours[::3], 'ride_count_accepted':2})
    quality=pd.DataFrame({'station_id':'001_', 'hour_start_utc':hours,
                          'count_complete':True,'evidence_id':'synthetic-accepted-enumeration'})
    panel=build_station_panel(observed,{'001_':1},start=hours[0],end=hours[-1]+pd.Timedelta(hours=1),quality=quality)
    def window(p):return make_window(p,'001_',o,split_start=o,split_end=o+pd.Timedelta(hours=24))
    a=window(panel)
    assert a.encoder_observed.shape==(336,5) and a.target.shape==(24,1) and a.target_mask.all()
    assert ((a.target_hours_utc[1:]-a.target_hours_utc[:-1])==pd.Timedelta(hours=1)).all()
    panel.loc[panel.hour_start_utc.ge(o),'ride_count']=99999
    panel.loc[panel.hour_start_utc.ge(o),'label_observed_mask']=False
    b=window(panel)
    for name in ['encoder_observed','encoder_known','decoder_known']:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
