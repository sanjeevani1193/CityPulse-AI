"""Small artifact-free station selection, quality, DST and causal window tests."""

import numpy as np
import pandas as pd
import pytest

from citypulse_ai.features.station_dataset import (
    NYC, CountScaler, InsufficientStations, build_station_panel, fit_count_scalers,
    make_window, origin_manifest, select_candidate_stations,
)


def stamp(s): return pd.Timestamp(s, tz=NYC).tz_convert('UTC')


def sparse(*rows):
    frame = pd.DataFrame(rows, columns=['station_id', 'hour_start_utc', 'ride_count_accepted'])
    frame['hour_start_utc'] = pd.to_datetime(frame.hour_start_utc, utc=True)
    return frame


def sample(origin='2024-07-01', complete=True):
    o = stamp(origin);start = o - pd.Timedelta(hours=336);end = o + pd.Timedelta(hours=24)
    hours = pd.date_range(start, end, freq='h', inclusive='left')
    observed = sparse(*[('001.01_', h, (i % 5) + 1) for i, h in enumerate(hours)])
    q = pd.DataFrame({'station_id': '001.01_', 'hour_start_utc': hours,
                      'count_complete': complete, 'evidence_id': 'synthetic-reviewed-coverage'})
    return build_station_panel(observed, {'001.01_': 7}, start=start, end=end, quality=q), o


def window(panel, origin, **kw):
    return make_window(panel, '001.01_', origin, split_start=origin,
                       split_end=origin+pd.Timedelta(hours=24), **kw)


def activity(stations=('001', '001_')):
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    return pd.DataFrame([(s, d.date(), 3) for s in stations for d in dates],
                        columns=['station_id', 'local_date', 'ride_starts'])


def select(frame, **kw):
    return select_candidate_stations(frame, train_start=stamp('2024-01-01'),
        train_end=stamp('2024-07-01'), finalized_months=[f'2024-{m:02d}' for m in range(1,7)],
        size=2, **kw)


def test_training_only_station_ranking_exact_ids_and_future_invariance():
    a = activity();result = select(a)
    assert result.codes == {'001': 1, '001_': 2}
    future = pd.DataFrame([('future', pd.Timestamp('2024-07-02').date(), 10**9),
                           ('001_', pd.Timestamp('2024-07-01').date(), 10**12)], columns=a.columns)
    changed = select(pd.concat([a.sample(frac=1, random_state=7), future], ignore_index=True))
    pd.testing.assert_frame_equal(result.candidates, changed.candidates)
    assert result.codes == changed.codes
    assert result.metadata['scope'] == 'development_candidates_not_final_cohort'


def test_stability_checks_months_and_leading_trailing_gaps():
    a = activity(('stable', 'late', 'gap', 'sparse'))
    a = a.loc[~((a.station_id.eq('late') & a.local_date.lt(pd.Timestamp('2024-01-16').date()))
             | (a.station_id.eq('gap') & a.local_date.between(pd.Timestamp('2024-03-01').date(), pd.Timestamp('2024-03-15').date()))
             | (a.station_id.eq('sparse') & a.local_date.between(pd.Timestamp('2024-06-01').date(), pd.Timestamp('2024-06-20').date())))]
    with pytest.raises(InsufficientStations) as exc: select(a)
    table = exc.value.candidates.set_index('station_id')
    assert table.loc['stable', 'eligible']
    assert table.loc['late', 'longest_no_record_dates'] == 15
    assert 'long_no_record_gap' in table.loc['gap', 'exclusion_reason']
    assert 'insufficient_monthly_activity' in table.loc['sparse', 'exclusion_reason']
    assert len(table.loc[table.selected]) == 1  # Never silently fill cohort.


def test_missing_finalized_training_month_blocks_selection():
    with pytest.raises(ValueError, match='coverage'):
        select_candidate_stations(activity(), train_start=stamp('2024-01-01'),
            train_end=stamp('2024-07-01'), finalized_months=['2024-01'], size=1)


@pytest.mark.parametrize('bad', [123, '', None])
def test_numeric_or_blank_station_ids_fail(bad):
    a = activity(('good',));a['station_id'] = a.station_id.astype(object);a.loc[0, 'station_id'] = bad
    with pytest.raises(ValueError):select(a)


def test_quality_default_preserves_audit_positive_and_unknown_missing():
    start = stamp('2024-06-01');end = start+pd.Timedelta(hours=3)
    observed = sparse(('001', start, 8))
    p = build_station_panel(observed, {'001': 1}, start=start, end=end)
    assert p.ride_count_accepted.tolist()[0] == 8
    assert p.ride_count.isna().all() and not p.label_observed_mask.any()
    assert p.station_available.isna().all() and not p.availability_known_mask.any()
    assert p.label_quality_reason.tolist() == ['accepted_count_unverified','unknown_enumeration','unknown_enumeration']


def test_trusted_zero_requires_coverage_not_operating_status():
    start=stamp('2024-06-01');hours=pd.date_range(start,periods=3,freq='h')
    q=pd.DataFrame({'station_id':['001']*2,'hour_start_utc':hours[:2],
                    'count_complete':[True,True],'evidence_id':['review','review']})
    status=pd.DataFrame({'station_id':['001'],'hour_start_utc':hours[2:],
                        'station_available':[False],'status_available_at_utc':[start]})
    p=build_station_panel(sparse(('001',start,7)),{'001':1},start=start,
                          end=start+pd.Timedelta(hours=3),quality=q,status=status)
    assert p.ride_count.iloc[:2].tolist()==[7,0]
    assert p.label_observed_mask.tolist()==[True,True,False]
    assert pd.isna(p.ride_count.iloc[2])  # Closure alone does not imply count zero.
    assert pd.isna(p.station_available.iloc[1])
    assert p.label_quality_reason.iloc[1]=='trusted_zero_complete_enumeration'
    q.loc[0,'evidence_id']=''
    with pytest.raises(ValueError,match='evidence'):build_station_panel(
        sparse(('001',start,7)),{'001':1},start=start,end=start+pd.Timedelta(hours=3),quality=q)


@pytest.mark.parametrize('count', [-1,1.5,float('nan'),float('inf')])
def test_invalid_sparse_counts_rejected(count):
    start=stamp('2024-06-01')
    with pytest.raises(ValueError):build_station_panel(sparse(('A',start,count)),{'A':1},start=start,end=start+pd.Timedelta(hours=1))


def test_duplicate_hours_and_row_budget_fail_closed():
    start=stamp('2024-06-01');d=sparse(('A',start,1),('A',start,2))
    with pytest.raises(ValueError,match='duplicate'):build_station_panel(d,{'A':1},start=start,end=start+pd.Timedelta(hours=2))
    with pytest.raises(ValueError,match='budget'):build_station_panel(d.iloc[:1],{'A':1},start=start,end=start+pd.Timedelta(hours=2),max_rows=1)


def test_shapes_feature_order_and_repeated_windows():
    panel,o=sample();a=window(panel,o);b=window(panel.sample(frac=1,random_state=42),o)
    assert a.static_station_code.tolist()==[7]
    assert a.encoder_known.shape==(336,11) and a.encoder_observed.shape==(336,5)
    assert a.decoder_known.shape==(24,11) and a.target.shape==a.target_mask.shape==(24,1)
    assert a.primary_evaluation_eligible
    assert a.encoder_observed[-1,1]==1 and a.encoder_observed[-1,2]==0
    assert a.decoder_known[0,-1]==pytest.approx(336/359)
    for n in ['encoder_known','encoder_observed','decoder_known','target','target_mask']:
        np.testing.assert_array_equal(getattr(a,n),getattr(b,n))


def test_future_counts_masks_and_status_cannot_change_inputs():
    p,o=sample();a=window(p,o)
    p.loc[p.hour_start_utc.ge(o),'ride_count']=100000
    p.loc[p.hour_start_utc.ge(o),'label_observed_mask']=False
    p.loc[p.hour_start_utc.ge(o),'station_available']=True
    p.loc[p.hour_start_utc.ge(o),'availability_known_mask']=True
    p.loc[p.hour_start_utc.ge(o),'status_available_at_utc']=o-pd.Timedelta(hours=1)
    b=window(p,o)
    for n in ['encoder_known','encoder_observed','decoder_known','static_station_code']:
        np.testing.assert_array_equal(getattr(a,n),getattr(b,n))
    assert not b.target_mask.any()


def test_latency_masks_recent_hours_and_ages_without_filling():
    p,o=sample();w=window(p,o,latency_hours=3)
    assert w.encoder_observed[-4:,1].tolist()==[1,0,0,0]
    np.testing.assert_allclose(w.encoder_observed[-4:,2],[0,1/336,2/336,3/336])
    assert (w.encoder_observed[-3:,0]==0).all()
    assert p.observation_available_at_utc.isna().all()  # Replay assumption never writes fake timestamps.


def test_operational_inputs_require_real_publication_times_and_quality_dates():
    p,o=sample();assert not window(p,o,mode='operational').input_eligible
    p['observation_available_at_utc']=p.hour_start_utc+pd.Timedelta(hours=1)
    p['coverage_status_available_at_utc']=p.observation_available_at_utc
    p.loc[p.hour_start_utc.eq(o-pd.Timedelta(hours=1)),'observation_available_at_utc']=o+pd.Timedelta(hours=1)
    w=window(p,o,mode='operational')
    assert w.encoder_observed[-1,1]==0 and w.encoder_observed[-2,1]==1
    p.loc[p.hour_start_utc.eq(o-pd.Timedelta(hours=2)),'coverage_status_available_at_utc']=o+pd.Timedelta(hours=1)
    assert window(p,o,mode='operational').encoder_observed[-2,1]==0


def test_scaler_unique_training_only_constant_and_missing_failure():
    p,o=sample();end=o-pd.Timedelta(hours=100)
    scales=fit_count_scalers(p,train_start=p.hour_start_utc.min(),train_end=end)
    q=p.copy();q.loc[q.hour_start_utc.ge(end),'ride_count']=1000000
    assert scales==fit_count_scalers(q,train_start=p.hour_start_utc.min(),train_end=end)
    s=scales['001.01_'];assert s.sample_count==236
    p['ride_count']=4
    assert fit_count_scalers(p,train_start=p.hour_start_utc.min(),train_end=o)['001.01_'].scale==1
    p['label_observed_mask']=False
    with pytest.raises(ValueError,match='No trusted'):fit_count_scalers(p,train_start=p.hour_start_utc.min(),train_end=o)


def test_unknown_and_real_zero_tensor_masks_are_distinct_after_scaling():
    p,o=sample();p.loc[0,'ride_count']=0;p.loc[1,'ride_count']=pd.NA;p.loc[1,'label_observed_mask']=False
    w=window(p,o,scalers={'001.01_':CountScaler(3,2,100,p.hour_start_utc.min(),o)})
    assert w.encoder_observed[0,:2].tolist()==[-1.5,1]
    assert w.encoder_observed[1,:2].tolist()==[0,0]
    assert np.isfinite(w.encoder_observed).all()


def test_status_masks_respect_publication_time_even_in_retrospective_mode():
    p,o=sample();p.loc[:1,'station_available']=False;p.loc[:1,'availability_known_mask']=True
    p.loc[0,'status_available_at_utc']=o;p.loc[1,'status_available_at_utc']=o+pd.Timedelta(hours=1)
    w=window(p,o)
    assert w.encoder_observed[0,3:].tolist()==[0,1]
    assert w.encoder_observed[1,3:].tolist()==[0,0]


def test_all_missing_windows_remain_structural_diagnostics_not_trainable():
    p,o=sample(complete=False);w=window(p,o)
    assert not w.input_eligible and not w.primary_evaluation_eligible
    assert not w.target_mask.any() and (w.target==0).all()
    assert (w.encoder_observed[:,2]==1).all() and np.isfinite(w.encoder_observed).all()


@pytest.mark.parametrize('origin,last_hour,folds',[
    ('2024-03-10','2024-03-11T00:00:00-04:00',0),
    ('2024-11-03','2024-11-03T22:00:00-05:00',1)])
def test_dst_24_elapsed_buckets_and_local_fold(origin,last_hour,folds):
    p,o=sample(origin);w=window(p,o);local=w.target_hours_utc.tz_convert(NYC)
    assert len(w.target_hours_utc)==24 and local[-1]==pd.Timestamp(last_hour)
    assert int(w.decoder_known[:,9].sum())==folds
    if folds:
        assert list(local.hour).count(1)==2
        assert set(w.decoder_known[:,8])=={-4,-5}
    else:assert 2 not in local.hour


def test_midnight_warmup_and_target_split_boundary():
    p,o=sample()
    with pytest.raises(ValueError,match='midnight'):window(p,o+pd.Timedelta(hours=1))
    with pytest.raises(ValueError,match='Full encoder'):window(p.iloc[1:],o)
    with pytest.raises(ValueError,match='one split'):make_window(p,'001.01_',o,split_start=o,split_end=o+pd.Timedelta(hours=23))


def test_calendar_midnight_origins_dst_overlap_gap_and_initial_warmup():
    for day,nextday,hours in [('2024-03-10','2024-03-11',23),('2024-11-03','2024-11-04',25)]:
        origins=origin_manifest(data_start=stamp(day)-pd.Timedelta(hours=336),data_end=stamp(nextday)+pd.Timedelta(days=2),split_start=stamp(day),split_end=stamp(nextday)+pd.Timedelta(days=1))
        assert origins.origin_utc.iloc[1]-origins.origin_utc.iloc[0]==pd.Timedelta(hours=hours)
    warm=origin_manifest(data_start=stamp('2024-01-01'),data_end=stamp('2024-02-01'),split_start=stamp('2024-01-01'),split_end=stamp('2024-02-01'))
    assert warm.exclusion_reason.iloc[:14].eq('encoder_warmup').all()
    assert warm.structurally_eligible.iloc[14]


def test_explicit_zero_needs_certification_but_can_be_published_operationally():
    p,o=sample();p.loc[0,'ride_count']=0
    p['observation_available_at_utc']=p.hour_start_utc+pd.Timedelta(hours=1)
    p['coverage_status_available_at_utc']=p.observation_available_at_utc
    w=window(p,o,mode='operational')
    assert w.encoder_observed[0,:2].tolist()==[0,1]


def test_scaler_from_evaluation_labels_cannot_be_used_for_evaluation():
    p,o=sample()
    with pytest.raises(ValueError,match='Scaler fitting period'):
        window(p,o,scalers={'001.01_':CountScaler(1,1,100,p.hour_start_utc.min(),o+pd.Timedelta(hours=1))})


def test_out_of_scope_quality_and_naive_publication_instants_fail():
    start=stamp('2024-06-01');d=sparse(('A',start,0))
    q=pd.DataFrame({'station_id':['B'],'hour_start_utc':[start],'count_complete':[True],'evidence_id':['review']})
    with pytest.raises(ValueError,match='outside'):
        build_station_panel(d,{'A':1},start=start,end=start+pd.Timedelta(hours=1),quality=q)
    d['observation_available_at_utc']=pd.Timestamp('2024-06-01')
    with pytest.raises(ValueError,match='Publication'):
        build_station_panel(d,{'A':1},start=start,end=start+pd.Timedelta(hours=1))


def test_lazy_windows_and_paired_evaluation_keys():
    from citypulse_ai.features.station_dataset import iter_windows
    p,o=sample()
    stream=iter_windows(p,split_start=o,split_end=o+pd.Timedelta(hours=24))
    assert iter(stream) is stream
    w=next(stream)
    assert len(list(stream))==0
    keys=w.evaluation_keys()
    assert keys.horizon_step.tolist()==list(range(1,25))
    assert not keys.duplicated().any()
    assert keys.target_hour_start_utc.iloc[0]==o


def test_empty_midnight_interval_yields_no_windows():
    from citypulse_ai.features.station_dataset import iter_windows
    p,o=sample()
    assert list(iter_windows(p,split_start=o+pd.Timedelta(hours=1),split_end=o+pd.Timedelta(hours=2)))==[]


def test_corrupt_trusted_label_and_static_code_fail_closed():
    p,o=sample();p.loc[0,'ride_count']=pd.NA
    with pytest.raises(ValueError,match='Counts'):window(p,o)
    p,o=sample();p.loc[0,'station_code']=8
    with pytest.raises(ValueError,match='Station code'):window(p,o)


def test_fourteen_date_gap_is_eligible_and_fifteen_is_not():
    a=activity(('A','B'))
    remove=a.station_id.eq('B') & a.local_date.between(pd.Timestamp('2024-03-01').date(),pd.Timestamp('2024-03-14').date())
    assert select(a.loc[~remove]).codes=={'A':1,'B':2}
    remove=a.station_id.eq('B') & a.local_date.between(pd.Timestamp('2024-03-01').date(),pd.Timestamp('2024-03-15').date())
    with pytest.raises(InsufficientStations):select(a.loc[~remove])
