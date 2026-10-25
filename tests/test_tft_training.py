"""Training control and metric tests use tiny synthetic windows only."""
import numpy as np
import pytest
import torch

from citypulse_ai.forecasting.tft_training import Metrics, StationStream, restore
from citypulse_ai.forecasting.tft import TFTConfig, TemporalFusionTransformer


def test_masked_original_unit_metrics():
    m=Metrics();m.update(np.array([[0.,4.,999.]]),np.array([[[-1.,1.,3.],[2.,3.,5.],[0.,0.,0.]]]),np.array([[True,True,False]]))
    r=m.result();assert r['targets']==2 and r['MAE']==1 and r['RMSE']==1 and r['WAPE_percent']==50
    assert r['interval_coverage']==1 and r['mean_interval_width']==3.5
    np.testing.assert_allclose(list(r['pinball_by_quantile'].values()),[.15,.5,.2])


def test_empty_metrics_and_crossings_fail():
    with pytest.raises(ValueError,match='No eligible'):Metrics().result()
    with pytest.raises(ValueError,match='crossings'):
        Metrics().update(np.array([1.]),np.array([[2.,1.,3.]]),np.array([True]))


def test_test_split_forbidden(tmp_path):
    with pytest.raises(ValueError,match='test targets'):StationStream(tmp_path,'test')


def test_checkpoint_optimizer_rng_and_model_restore(tmp_path):
    torch.manual_seed(42);model=TemporalFusionTransformer(TFTConfig(hidden=4,heads=1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    path=tmp_path/'epoch-001.pt';rng=torch.get_rng_state()
    torch.save({'dataset_manifest_sha256':'frozen','model_config':model.artifact_config(),
        'state_dict':model.state_dict(),'optimizer_state':optimizer.state_dict(),
        'training_config':{'learning_rate':.001},'torch_rng_state':rng},path)
    _,other,opt=restore(path,'frozen')
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters()))
    assert torch.equal(torch.get_rng_state(),rng) and opt.param_groups[0]['lr']==.001
    with pytest.raises(ValueError,match='dataset differs'):restore(path,'other')

# Reuse the tiny, synthetic on-disk panel fixture, never real historical data.
from tests.test_final_station_dataset import saved


def test_complete_epoch_budget_stop_and_resume(saved,tmp_path,monkeypatch):
    import json,time
    from citypulse_ai.forecasting import tft_training as t
    root,*_=saved
    manifest=json.loads((root/'dataset.json').read_text());manifest['identity']='synthetic-final'
    # Match the final builder's dated-panel naming contract.
    for old,new in [('panel-0.parquet','panel-2024-01.parquet'),('panel-1.parquet','panel-2024-02.parquet')]:
        (root/old).rename(root/new)
        for receipt in manifest['artifacts']+manifest['panels']:
            if receipt['name']==old:receipt['name']=new
    (root/'dataset.json').write_text(json.dumps(manifest))
    output=tmp_path/'training';clock=[0.];actual=time.monotonic
    monkeypatch.setattr(t.time,'monotonic',lambda:actual()+clock[0])
    def published(path):
        clock[0]=1000.
        return {'local_sha256':t.fingerprint(path)}
    first=t.train(root,output,max_seconds=1000,batch_size=8,max_epochs=2,publish=published)
    assert first['epochs_completed']==1 and not first['training_complete']
    assert first['stop_reason']=='runtime_budget_before_next_epoch' and not first['test_evaluated']
    assert first['history'][0]['training_windows']==len(t.StationStream(root,'train').index)
    clock[0]=0.
    second=t.train(root,output,max_seconds=1000,batch_size=8,max_epochs=2,resume=output/'epoch-001.pt')
    assert second['epochs_completed']==2 and second['training_complete']
    assert second['history'][0]==first['history'][0]
    scores=second['validation'];assert scores['targets']==len(t.StationStream(root,'validation').index)*24
    assert all(b['targets']==scores['targets'] for b in scores['baselines'].values())
    assert scores['nonnegative_clipped']['targets']==scores['targets']
    assert scores['nonnegative_clipped']['paired_key_sha256']==scores['paired_key_sha256']
    assert not scores['nonnegative_clipped']['selection_used']


def test_resume_driver_requires_reviewed_hash():
    from citypulse_ai.forecasting.tft_training import validate_resume_driver, COMPATIBLE_RESUME_DRIVERS
    validate_resume_driver({'training_code_sha256':next(iter(COMPATIBLE_RESUME_DRIVERS))})
    with pytest.raises(ValueError,match='unreviewed'):
        validate_resume_driver({'training_code_sha256':'unknown'})


def test_session_cap_preserves_global_schedule():
    from citypulse_ai.forecasting.tft_training import session_last_epoch
    assert session_last_epoch(4,10,3)==6
    assert session_last_epoch(9,10,3)==10
    assert session_last_epoch(4,10,None)==10
    with pytest.raises(ValueError,match='positive'):session_last_epoch(4,10,0)


def test_nonnegative_clipping_is_ordered_and_masks_unchanged():
    actual=np.array([0.,4.,999.]);raw=np.array([[-3.,-1.,2.],[-1.,3.,5.],[-9.,-8.,-7.]])
    mask=np.array([True,True,False]);a=Metrics();b=Metrics()
    a.update(actual,raw,mask);b.update(actual,np.maximum(raw,0.),mask)
    original=a.result();clipped=b.result()
    assert original['targets']==clipped['targets']==2
    assert original['negative_median_fraction']==.5 and clipped['negative_median_fraction']==0
    assert clipped['MAE']==.5 and original['MAE']==1
    assert clipped['interval_coverage']==original['interval_coverage']==1
    assert all(clipped['pinball_by_quantile'][q]<=original['pinball_by_quantile'][q] for q in ['q10','q50','q90'])


def test_complete_epoch_session_cap_and_reviewed_predecessor_resume(saved,tmp_path):
    import json
    from citypulse_ai.forecasting import tft_training as t
    root,*_=saved
    manifest=json.loads((root/'dataset.json').read_text());manifest['identity']='synthetic-session-cap'
    for old,new in [('panel-0.parquet','panel-2024-01.parquet'),('panel-1.parquet','panel-2024-02.parquet')]:
        (root/old).rename(root/new)
        for receipt in manifest['artifacts']+manifest['panels']:
            if receipt['name']==old:receipt['name']=new
    (root/'dataset.json').write_text(json.dumps(manifest))
    out=tmp_path/'capped'
    first=t.train(root,out,max_seconds=1000,batch_size=8,max_epochs=3,additional_epochs=1)
    assert first['epochs_completed']==1 and first['stop_reason']=='session_epoch_cap'
    assert not first['training_complete'] and first['training_config']['max_epochs']==3
    cp=torch.load(out/'epoch-001.pt',weights_only=True)
    cp['training_code_sha256']=next(iter(t.COMPATIBLE_RESUME_DRIVERS))
    predecessor=tmp_path/'reviewed-predecessor.pt';torch.save(cp,predecessor)
    second=t.train(root,out,max_seconds=1000,batch_size=8,max_epochs=3,
        additional_epochs=1,resume=predecessor)
    assert second['epochs_completed']==2 and not (out/'epoch-003.pt').exists()
    assert second['history'][0]==first['history'][0]
    assert second['resumed_driver_sha256']==cp['training_code_sha256']
    assert second['best']['validation_loss']==min(x['validation']['standardized_masked_pinball'] for x in second['history'])
