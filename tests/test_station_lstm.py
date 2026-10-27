"""Synthetic station LSTM contracts; never real datasets or saved model weights."""
import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.station_dataset import make_window
from citypulse_ai.forecasting.station_lstm import (
    Config, DevelopmentData, build_model, evaluate, masked_mse, optimizer_step, train, verify_reference,
)
from tests.test_final_station_dataset import saved
from tests.test_station_xgboost import rename_panels


def reference(root, destination):
    data={s:DevelopmentData(root,s) for s in ('train','validation')}
    import hashlib
    digest=hashlib.sha256()
    # Use the original canonical row-key convention, not model predictions.
    records=[]
    for _,y,m,meta in data['validation'].batches(64):
        for i,origin in enumerate(meta['origins']):
            for lead,hour in enumerate(pd.DatetimeIndex(meta['target_hours'][i])):
                digest.update(f"{meta['station_id']}|{origin.isoformat()}|{hour.isoformat()}|{lead+1}\n".encode())
            records.append((meta['actual'][i],meta['lag24'][i],meta['lag168'][i]))
    from citypulse_ai.forecasting.station_xgboost import paired_metrics
    a,b,c=[np.concatenate([r[i] for r in records]) for i in range(3)]
    result={'dataset_manifest_sha256':fingerprint(root/'dataset.json'),'test_evaluated':False,
        'train_windows':len(data['train']),'validation':{'forecast_windows':len(data['validation']),
        'targets':len(a),'paired_key_sha256':digest.hexdigest(),
        'baselines':{'lag_24':paired_metrics(a,b),'lag_168':paired_metrics(a,c)}}}
    destination.write_text(json.dumps(result));return destination


def test_100_station_architecture_deterministic_forward_and_optimizer(tmp_path):
    model=build_model();other=build_model()
    assert model.count_params()==8953
    x={'history':np.zeros((2,336,16),np.float32),'station':np.array([[1],[100]],np.int32),
       'calendar':np.zeros((2,24,11),np.float32)}
    before=model(x,training=False).numpy()
    assert before.shape==(2,24)
    np.testing.assert_array_equal(before,other(x,training=False))
    assert not np.array_equal(before[0],before[1])
    loss=optimizer_step(model)(x,np.ones((2,24),np.float32),np.ones((2,24),np.float32))
    assert np.isfinite(loss) and int(model.optimizer.iterations)==1
    model.save(tmp_path/'model.keras');restored=tf.keras.models.load_model(tmp_path/'model.keras')
    np.testing.assert_array_equal(model(x,training=False),restored(x,training=False))
    assert int(restored.optimizer.iterations)==1
    optimizer_step(restored)(x,np.ones((2,24),np.float32),np.ones((2,24),np.float32))
    optimizer_step(model)(x,np.ones((2,24),np.float32),np.ones((2,24),np.float32))
    for a,b in zip(model.get_weights(),restored.get_weights()):np.testing.assert_array_equal(a,b)


def test_masked_loss_ignores_unknown_targets_and_empty_batches():
    y=tf.constant([[1.,np.nan,3.]]);p=tf.Variable([[2.,999.,1.]])
    with tf.GradientTape() as tape:loss=masked_mse(y,p,tf.constant([[1.,0.,1.]]))
    assert float(loss)==2.5
    np.testing.assert_array_equal(tape.gradient(loss,p),[[1.,0.,-2.]])
    assert float(masked_mse(y,p,tf.zeros((1,3))))==0
    model=build_model(Config(stations=2));x={'history':np.zeros((1,336,16),np.float32),
        'station':np.ones((1,1),np.int32),'calendar':np.zeros((1,24,11),np.float32)}
    optimizer_step(model)(x,np.full((1,24),np.nan,np.float32),np.zeros((1,24),np.float32))
    assert int(model.optimizer.iterations)==0


def test_batch_matches_original_window_and_training_scalers(saved):
    root,panel,scalers,_,splits,*_=saved;rename_panels(root)
    data=DevelopmentData(root,'validation')
    x,y,m,meta=next(data.batches(3));station=meta['station_id'];origin=meta['origins'][0]
    expected=make_window(panel,station,origin,split_start=splits['validation'][0],
        split_end=splits['validation'][1],scalers=scalers)
    np.testing.assert_array_equal(x['history'][0,:,:11],expected.encoder_known)
    np.testing.assert_array_equal(x['history'][0,:,11:],expected.encoder_observed)
    np.testing.assert_array_equal(x['calendar'][0],expected.decoder_known)
    np.testing.assert_allclose(y[0],(expected.target[:,0]-scalers[station].mean)/scalers[station].scale,rtol=1e-6)
    assert m.all() and max(s.mean for s in data.scalers.values())<9
    assert not x['history'][...,15].any()
    with pytest.raises(ValueError,match='Test split'):DevelopmentData(root,'test')


def test_no_future_target_inputs_and_dst_grid(saved,monkeypatch):
    root,panel,_,_,_,*_=saved;rename_panels(root)
    data=DevelopmentData(root,'train')
    origin=pd.Timestamp('2024-03-10',tz='America/New_York').tz_convert('UTC')
    data.index=data.index.loc[data.index.origin_utc.eq(origin)].reset_index(drop=True)
    x,y,m,meta=next(data.batches(2));hours=pd.DatetimeIndex(meta['target_hours'][0])
    assert len(hours)==24 and 2 not in hours.tz_convert('America/New_York').hour
    original=pd.read_parquet
    def changed(*a,**kw):
        frame=original(*a,**kw)
        frame.loc[frame.hour_start_utc.ge(origin),'ride_count']=123456
        return frame
    monkeypatch.setattr(pd,'read_parquet',changed)
    xx,yy,_,_=next(data.batches(2))
    for k in x:np.testing.assert_array_equal(x[k],xx[k])
    assert not np.array_equal(y,yy)


def test_test_quarter_never_opened_and_bad_receipt_fails(saved):
    root,*_=saved;rename_panels(root)
    p=root/'dataset.json';d=json.loads(p.read_text());r={'name':'panel-2025-10.parquet','sha256':'absent','size_bytes':123}
    d['panels'].append(r);d['artifacts'].append(r);p.write_text(json.dumps(d))
    assert len(DevelopmentData(root,'train'))>0
    (root/'panel-2024-01.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):DevelopmentData(root,'validation')


def test_complete_epochs_resume_with_optimizer_and_reference_fail_closed(saved,tmp_path):
    root,*_=saved;rename_panels(root);ref=reference(root,tmp_path/'reference.json')
    config=Config(stations=2,max_epochs=2,patience=2,batch_size=64)
    out=tmp_path/'interrupted'
    def interrupt(path):
        if path.name=='epoch-001.json':raise RuntimeError('simulated interruption after checkpoint')
    with pytest.raises(RuntimeError,match='simulated'):
        train(root,out,ref,config=config,publish=interrupt)
    state=json.loads((out/'epoch-001.json').read_text())
    report=train(root,out,ref,config=config,resume='epoch-001.keras')
    assert report['epochs_completed']==2 and report['training_complete'] and not report['test_evaluated']
    assert report['history'][0]==state['history'][0]
    assert report['optimizer_iterations']>=state['optimizer_iterations']
    assert report['best']['validation_mse_scaled']==min(h['validation']['mse_scaled'] for h in report['history'])
    pred=pd.read_parquet(out/'validation-predictions.parquet')
    assert len(pred)==report['validation']['targets'] and pred.groupby(['station_id','origin_utc']).size().eq(24).all()
    data={s:DevelopmentData(root,s) for s in ('train','validation')}
    value=json.loads(ref.read_text());value['validation']['paired_key_sha256']='wrong';ref.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='keys differ'):verify_reference(root,data,ref)
    with pytest.raises(ValueError,match='keys differ'):
        train(root,out,ref,config=config,resume='epoch-001.keras')
