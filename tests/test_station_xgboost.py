"""Small synthetic leakage, DST, missingness and native XGBoost regressions."""
import json
import numpy as np
import pandas as pd
import pytest

from citypulse_ai.features.station_dataset import _calendar
from citypulse_ai.forecasting.station_xgboost import FEATURES,LAGS,feature_block,materialize,paired_metrics,train
from citypulse_ai.forecasting.xgboost_model import forecast_features
from tests.test_final_station_dataset import saved


def arrays(origin):
    hours=pd.date_range(pd.Timestamp(origin)-pd.Timedelta(hours=336),periods=360,freq='h')
    y=np.arange(360,dtype=float);mask=np.ones(360,dtype=bool)
    from citypulse_ai.features.station_dataset import KNOWN_FEATURES
    known=_calendar(hours)[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=float)
    return hours,y,mask,known,np.zeros(360),np.zeros(360,dtype=bool)


@pytest.mark.parametrize('origin',['2024-03-10T05:00:00Z','2024-11-03T04:00:00Z'])
def test_exact_lags_dst_and_future_demand_is_not_an_input(origin):
    hours,y,mask,known,status,sm=arrays(origin)
    a=feature_block(y,mask,known,status,sm,336,1)
    expected=forecast_features(pd.Series(y,index=hours),pd.Timestamp(origin))
    for lag in LAGS:np.testing.assert_array_equal(a[:,FEATURES.index(f'lag_{lag}')],y[336+np.arange(24)-lag])
    for lag in (24,168):np.testing.assert_array_equal(a[:,FEATURES.index(f'lag_{lag}')],expected[f'lag_{lag}'])
    y[336:]=999999;mask[336:]=False;status[336:]=1;sm[336:]=True
    b=feature_block(y,mask,known,status,sm,336,1);np.testing.assert_array_equal(a,b)
    assert a.shape==(24,48) and len(hours[336:])==24
    assert a[:,FEATURES.index('availability_known_fraction')].tolist()==[0]*24
    assert np.isnan(a[:,FEATURES.index('last_station_available')]).all()


def test_observed_zero_unknown_counts_and_unknown_station_operation():
    _,y,mask,known,status,sm=arrays('2024-02-01T05:00:00Z')
    y[312]=0;mask[313]=False
    a=feature_block(y,mask,known,status,sm,336,100)
    assert a[0,FEATURES.index('lag_24')]==0 and a[0,FEATURES.index('lag_24_observed')]==1
    assert np.isnan(a[1,FEATURES.index('lag_24')]) and a[1,FEATURES.index('lag_24_observed')]==0
    mask[:336]=False;b=feature_block(y,mask,known,status,sm,336,1)
    assert np.isnan(b[:,FEATURES.index('past_336_mean')]).all()
    assert (b[:,FEATURES.index('past_336_observed_fraction')]==0).all()
    mask[0]=True;y[0]=-1
    with pytest.raises(ValueError,match='Invalid trusted'):feature_block(y,mask,known,status,sm,336,1)
    with pytest.raises(ValueError,match='Full'):feature_block(y,mask,known,status,sm,335,1)


def rename_panels(root):
    data=json.loads((root/'dataset.json').read_text());data['identity']='synthetic'
    for old,new in [('panel-0.parquet','panel-2024-01.parquet'),('panel-1.parquet','panel-2024-02.parquet')]:
        (root/old).rename(root/new)
        for receipt in data['artifacts']+data['panels']:
            if receipt['name']==old:receipt['name']=new
    (root/'dataset.json').write_text(json.dumps(data))


def test_materialization_preserves_targets_and_exact_station_keys(saved,tmp_path):
    root,panel,scalers,index,splits,*_=saved;rename_panels(root)
    out=tmp_path/'validation';meta=materialize(root,'validation',out)
    keys=pd.read_parquet(out/'keys.parquet');features=np.load(out/'features.npy');labels=np.load(out/'labels.npy')
    assert len(keys)==len(index.loc[index.split.eq('validation')])*24==meta['trusted_targets']
    assert set(keys.station_id)=={'001','exact.00_'} and not keys.duplicated(['station_id','origin_utc','target_hour_utc','horizon_step']).any()
    np.testing.assert_array_equal(labels,keys.actual)
    for lag in (24,168):np.testing.assert_array_equal(features[:,FEATURES.index(f'lag_{lag}')],keys[f'lag{lag}'])
    with pytest.raises(ValueError,match='Test split'):materialize(root,'test',tmp_path/'never')
    # Explicit panel filters must exclude the test-period body, even in a shared
    # quarter file: every returned key ends before the validation boundary.
    assert keys.target_hour_utc.max()<pd.Timestamp(splits['validation'][1])


def test_native_tiny_training_reload_and_search_are_validation_only(saved,tmp_path):
    import xgboost as xgb
    root,*_=saved;rename_panels(root)
    out=tmp_path/'model';result=train(root,out,max_seconds=1200,rounds=4,patience=2,
        search=({'max_depth':2,'objective':'reg:squarederror'},{'max_depth':2,'objective':'reg:absoluteerror'}))
    assert len(result['search'])==2 and not result['test_evaluated'] and not result['refit_on_validation']
    assert result['selected']['metrics']['clipped']['MAE']==min(s['metrics']['clipped']['MAE'] for s in result['search'])
    data=pd.read_parquet(out/'validation-predictions.parquet')
    assert len(data)==result['validation']['rows'] and (data.prediction_clipped>=0).all()
    model=xgb.Booster();model.load_model(out/'model.ubj');assert model.feature_names==list(FEATURES)
    np.testing.assert_array_equal(model.inplace_predict(np.load(out/'reload-probe.npy')),np.load(out/'reload-expected.npy'))
    assert paired_metrics(data.actual,data.prediction_raw)==result['selected']['metrics']['raw']


def test_raw_negative_predictions_are_scored_without_silent_clipping():
    raw=paired_metrics([0,2],[-1,3]);clipped=paired_metrics([0,2],[0,3])
    assert raw['MAE']==raw['RMSE']==1 and raw['WAPE_percent']==100
    assert clipped['MAE']==.5 and clipped['WAPE_percent']==50
    assert paired_metrics([0],[0])['WAPE_percent'] is None


def test_mismatched_tft_reference_stops_before_any_fit(saved,tmp_path,monkeypatch):
    import xgboost as xgb
    root,*_=saved;rename_panels(root)
    from citypulse_ai.etl.archives import fingerprint
    ref=tmp_path/'reference.json'
    ref.write_text(json.dumps({'dataset_manifest_sha256':fingerprint(root/'dataset.json'),
        'train_windows':-1,'validation':{'forecast_windows':1,'targets':24,'paired_key_sha256':'different'}}))
    def forbidden(*args,**kwargs):raise AssertionError('Fit called despite mismatched reference')
    monkeypatch.setattr(xgb,'train',forbidden)
    with pytest.raises(ValueError,match='fitting prohibited'):train(root,tmp_path/'mismatch',reference=ref)
