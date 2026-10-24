"""Small final-panel/lazy-reader fixtures; no real history or cloud access."""
import json

import numpy as np
import pandas as pd
import pytest

from citypulse_ai.features.final_station_dataset import (
    LazyStationWindows, file_receipt, fit_partition_scalers, index_station, load_preprocessing, write_json,
)
from citypulse_ai.features.station_dataset import NYC, build_station_panel, make_window, select_candidate_stations


@pytest.fixture
def saved(tmp_path):
    start=pd.Timestamp('2024-02-15',tz=NYC).tz_convert('UTC');end=start+pd.Timedelta(hours=60*24)
    a=start+pd.Timedelta(hours=40*24);b=a+pd.Timedelta(hours=10*24)
    # Boundaries must be local midnight even across spring DST.
    a=pd.Timestamp('2024-03-26',tz=NYC).tz_convert('UTC');b=pd.Timestamp('2024-04-05',tz=NYC).tz_convert('UTC')
    end=pd.Timestamp('2024-04-15',tz=NYC).tz_convert('UTC')
    splits={'train':(start.isoformat(),a.isoformat()),'validation':(a.isoformat(),b.isoformat()),'test':(b.isoformat(),end.isoformat())}
    codes={'exact.00_':1,'001':2};hours=pd.date_range(start,end,freq='h',inclusive='left')
    observed=pd.MultiIndex.from_product([list(codes),hours],names=['station_id','hour_start_utc']).to_frame(index=False)
    observed['ride_count_accepted']=np.arange(len(observed))%9
    # Dramatically different future counts must not influence fitted moments.
    observed.loc[observed.hour_start_utc.ge(a),'ride_count_accepted']=10000
    quality=observed[['station_id','hour_start_utc']].assign(count_complete=True,evidence_id='synthetic-certified')
    panel=build_station_panel(observed,codes,start=start,end=end,quality=quality)
    paths=[]
    for i,(left,right) in enumerate([(start,a),(a,end)]):
        p=tmp_path/f'panel-{i}.parquet';panel.loc[panel.hour_start_utc.ge(left)&panel.hour_start_utc.lt(right)].to_parquet(p,index=False);paths.append(p)
    scalers=fit_partition_scalers(paths,codes,train_start=start,train_end=a,destination=tmp_path/'preprocessing.json')
    indices=[]
    for station,code in codes.items():
        idx,exc=index_station(panel.loc[panel.station_id.eq(station)],station,code,data_start=start,data_end=end,splits=splits)
        indices.append(idx)
    index=pd.concat(indices,ignore_index=True);index.to_parquet(tmp_path/'windows.parquet',index=False)
    manifest={'status':'complete','panels':[file_receipt(p) for p in paths],'splits':splits,
        'artifacts':[file_receipt(p) for p in sorted(tmp_path.iterdir())]}
    write_json(tmp_path/'dataset.json',manifest)
    return tmp_path,panel,scalers,index,splits,start,end


def test_lazy_matches_original_windows_and_scalers(saved):
    root,panel,scalers,index,splits,start,end=saved
    ds=LazyStationWindows(root)
    assert len(ds)==len(index)
    for i in [0,len(ds)//2,len(ds)-1]:
        r=index.iloc[i];a,b=splits[r.split]
        expected=make_window(panel,r.station_id,r.origin_utc,split_start=a,split_end=b,scalers=scalers,
                             split_role='training' if r.split=='train' else 'evaluation')
        actual=ds[i]
        for name in ['encoder_known','encoder_observed','decoder_known','target','target_mask','static_station_code']:
            np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
        assert actual.target_mask.all() and actual.encoder_observed[:,1].all()
    for s,scaler in scalers.items():
        truth=panel.loc[panel.station_id.eq(s)&panel.hour_start_utc.lt(pd.Timestamp(splits['train'][1])),'ride_count'].astype(float)
        assert scaler.mean==truth.mean() and scaler.scale==truth.std(ddof=0) and scaler.mean<9


def test_dst_elapsed_horizons_and_repeated_reads(saved):
    root,_,_,index,_,_,_=saved;ds=LazyStationWindows(root)
    origin=pd.Timestamp('2024-03-10',tz=NYC).tz_convert('UTC')
    pos=np.flatnonzero(index.origin_utc.eq(origin).to_numpy())[0]
    a,b=ds[int(pos)],ds[int(pos)]
    assert len(a.target_hours_utc)==24 and 2 not in a.target_hours_utc.tz_convert(NYC).hour
    assert a.target_hours_utc.tz_convert(NYC)[-1].hour==0
    np.testing.assert_array_equal(a.target,b.target)
    assert a.encoder_known.shape==(336,11) and a.encoder_observed.shape==(336,5)


def test_unknown_masks_and_incomplete_grid(saved):
    _,panel,_,_,splits,start,end=saved
    frame=panel.loc[panel.station_id.eq('001')].copy()
    origin=pd.Timestamp('2024-03-10',tz=NYC).tz_convert('UTC')
    frame.loc[frame.hour_start_utc.eq(origin),'label_observed_mask']=False
    idx,_=index_station(frame,'001',2,data_start=start,data_end=end,splits=splits)
    row=idx.loc[idx.origin_utc.eq(origin)].iloc[0]
    assert row.trusted_targets==23 and not row.primary_evaluation_eligible and row.training_eligible
    with pytest.raises(ValueError,match='grid'):
        index_station(frame.iloc[1:],'001',2,data_start=start,data_end=end,splits=splits)


def test_lazy_hash_failure_and_unknown_split(saved):
    root,*_=saved
    with pytest.raises(ValueError,match='Unknown split'):LazyStationWindows(root,'future')
    (root/'panel-0.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact changed'):LazyStationWindows(root)


def test_future_counts_never_enter_encoder(saved):
    root,_,_,index,_,_,_=saved;ds=LazyStationWindows(root,'validation');w=ds[0]
    assert w.target.min()==10000
    assert w.encoder_observed[:,0].max()<10
    assert not w.encoder_observed[:,4].any()  # operating status remains unknown
    assert w.metadata['replay']=='retrospective'


def test_training_only_selection_ignores_future_activity():
    days=pd.date_range('2024-01-01','2024-01-31',freq='D')
    data=pd.DataFrame({'station_id':['A']*31+['B']*31,'local_date':list(days)*2,'ride_starts':[2]*31+[1]*31})
    future=pd.DataFrame({'station_id':['B','future-only'],'local_date':['2024-02-01']*2,'ride_starts':[10**9,10**9]})
    result=select_candidate_stations(pd.concat([data,future]),train_start='2024-01-01T00:00:00-05:00',
        train_end='2024-02-01T00:00:00-05:00',finalized_months=['2024-01'],size=1)
    assert result.codes=={'A':1}


def test_torch_preprocessing_compatibility(saved):
    root,*_=saved
    from citypulse_ai.forecasting.tft_preparation import load_preprocessing as original_load
    assert original_load(root/"preprocessing.json")==load_preprocessing(root/"preprocessing.json")


def test_build_split_identity_survives_json_checkpoint_roundtrip():
    from citypulse_ai.features.final_station_dataset import SPLITS
    plan={"identity":"synthetic","splits":SPLITS}
    assert json.loads(json.dumps(plan))==plan
