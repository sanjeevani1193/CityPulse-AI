"""Native TFT architecture, causal inputs, masking and frozen preprocessing."""

import io

import numpy as np
import pandas as pd
import pytest
import torch

from citypulse_ai.forecasting.tft import (
    TFTConfig, TemporalFusionTransformer, inverse_quantiles, masked_pinball_loss, training_step,
)
from citypulse_ai.forecasting.tft_preparation import (
    fit_training_scalers, load_preprocessing, save_preprocessing, windows_to_batch,
)
from citypulse_ai.features.station_dataset import build_station_panel, make_window


@pytest.fixture(autouse=True)
def reproducible_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def model():
    return TemporalFusionTransformer(TFTConfig(hidden=8, heads=2, dropout=0.))


def inputs():
    observed = torch.randn(2,336,5)
    observed[...,1] = 1
    observed[...,4] = 0
    return (torch.tensor([[1],[100]]), torch.randn(2,336,11), observed, torch.randn(2,24,11))


def test_shapes_station_context_selection_and_interpretable_attention():
    net=model().eval();x=inputs()
    with torch.inference_mode():pred,d=net(*x,return_diagnostics=True)
    assert pred.shape==(2,24,3) and torch.isfinite(pred).all()
    assert (pred[...,0]<=pred[...,1]).all() and (pred[...,1]<=pred[...,2]).all()
    assert d['encoder_weights'].shape==(2,336,16)
    assert d['decoder_weights'].shape==(2,24,11)
    assert d['static_weights'].shape==(2,1)
    for key in ['encoder_weights','decoder_weights','attention_weights','static_weights']:
        torch.testing.assert_close(d[key].sum(-1),torch.ones_like(d[key].sum(-1)))
    assert not torch.equal(d['static_representation'][0],d['static_representation'][1])
    assert net.attention.value.out_features==4  # One value projection shared by both heads.


def test_future_known_causality_and_actual_demand_is_not_a_decoder_argument():
    net=model().eval();x=inputs()
    with torch.inference_mode():
        before,d=net(*x,return_diagnostics=True)
        changed=[v.clone() for v in x];changed[3][:,12:]+=100
        after=net(*changed)
    torch.testing.assert_close(before[:,:12],after[:,:12],rtol=0,atol=0)
    assert not torch.equal(before[:,12:],after[:,12:])
    for i in range(24):assert (d['attention_weights'][:,i,337+i:]==0).all()
    with pytest.raises(TypeError):net(*x,target=torch.zeros(2,24,1))


def test_masked_encoder_payloads_cannot_affect_predictions():
    net=model().eval();x=inputs();x[2][:,::2,1]=0
    with torch.inference_mode():
        a=net(*x)
        changed=[v.clone() for v in x]
        changed[2][:,::2,0]=99999
        changed[2][...,3]=-99999  # All operating-status masks are unknown.
        b=net(*changed)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_forward_backward_updates_all_main_components():
    net=model();x=inputs();optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    before={n:p.detach().clone() for n,p in net.named_parameters()}
    loss=masked_pinball_loss(net(*x),torch.ones(2,24,1),torch.ones(2,24,1,dtype=torch.bool))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    optimizer.step()
    for prefix in ['station.','contexts.','encoder_selection.','decoder_selection.','encoder.',
                   'decoder.','enrichment.','attention.','quantile_head.']:
        assert any(not torch.equal(p,before[n]) for n,p in net.named_parameters() if n.startswith(prefix)),prefix


def test_pinball_manual_value_denominator_and_masked_nan_gradients():
    prediction=torch.tensor([[[1.,2.,3.],[100.,200.,300.]]],requires_grad=True)
    target=torch.tensor([[[4.],[float('nan')]]]);mask=torch.tensor([[[True],[False]]])
    loss=masked_pinball_loss(prediction,target,mask)
    assert loss.item()==pytest.approx(2.2/3)
    loss.backward();assert (prediction.grad[:,1]==0).all()
    assert torch.isfinite(prediction.grad).all()
    doubled=masked_pinball_loss(prediction.detach().repeat(2,1,1),target.repeat(2,1,1),mask.repeat(2,1,1))
    assert doubled==loss.detach()  # Valid hours, not batch/grid size, set denominator.


def test_all_missing_step_skips_optimizer_and_weight_decay():
    from citypulse_ai.forecasting.tft_preparation import TFTBatch
    net=model();x=inputs()
    batch=TFTBatch(*x,torch.full((2,24,1),float('nan')),torch.zeros(2,24,1,dtype=torch.bool),
                   torch.zeros(2),torch.ones(2))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1)
    before={k:v.clone() for k,v in net.state_dict().items()}
    assert training_step(net,batch,opt) is None
    assert not opt.state
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())


@pytest.mark.parametrize('problem',['prediction_nan','trusted_nan','wrong_mask'])
def test_invalid_loss_inputs_fail(problem):
    p=torch.zeros(1,24,3);y=torch.ones(1,24,1);m=torch.ones_like(y,dtype=torch.bool)
    if problem=='prediction_nan':p[0,0,0]=float('nan')
    if problem=='trusted_nan':y[0,0,0]=float('nan')
    if problem=='wrong_mask':m=m.float()
    with pytest.raises(ValueError):masked_pinball_loss(p,y,m)


def test_deterministic_eval_serialization_and_inverse_count_units():
    net=model().eval();x=inputs()
    with torch.inference_mode():a=net(*x);b=net(*x)
    assert torch.equal(a,b)
    artifact=io.BytesIO();torch.save({'config':net.artifact_config(),'state_dict':net.state_dict()},artifact)
    artifact.seek(0);saved=torch.load(artifact,map_location='cpu',weights_only=True)
    restored=TemporalFusionTransformer(TFTConfig(**saved['config']));restored.load_state_dict(saved['state_dict']);restored.eval()
    with torch.inference_mode():assert torch.equal(restored(*x),a)
    counts=inverse_quantiles(a,torch.tensor([10.,20.]),torch.tensor([2.,3.]))
    torch.testing.assert_close(counts[0],a[0]*2+10)
    assert (counts[...,0]<=counts[...,1]).all()


@pytest.mark.parametrize('bad',[0,101,-1])
def test_station_code_contract(bad):
    x=list(inputs());x[0][0,0]=bad
    with pytest.raises(ValueError,match='vocabulary'):model()(*x)


def panel_for(origin):
    o=pd.Timestamp(origin,tz='America/New_York').tz_convert('UTC')
    hours=pd.date_range(o-pd.Timedelta(hours=336),periods=360,freq='h')
    obs=pd.DataFrame({'station_id':'001_', 'hour_start_utc':hours,'ride_count_accepted':np.arange(360)%10+1})
    q=obs[['station_id','hour_start_utc']].assign(count_complete=True,evidence_id='synthetic')
    panel=build_station_panel(obs,{'001_':100},start=hours[0],end=hours[-1]+pd.Timedelta(hours=1),quality=q)
    return panel,o


@pytest.mark.parametrize('origin',['2024-03-10','2024-11-03'])
def test_dst_window_preprocessing_and_tft_forward(origin,tmp_path):
    panel,o=panel_for(origin);start=panel.hour_start_utc.min()
    scalers=fit_training_scalers(panel,train_start=start,train_end=o)
    w=make_window(panel,'001_',o,split_start=o,split_end=o+pd.Timedelta(hours=24),scalers=scalers)
    batch=windows_to_batch([w],scalers)
    assert len(w.target_hours_utc)==24
    assert ((w.target_hours_utc[1:]-w.target_hours_utc[:-1])==pd.Timedelta(hours=1)).all()
    pred=model()(*batch.inputs());assert pred.shape==(1,24,3)
    path=tmp_path/'preprocessing.json';save_preprocessing(path,scalers,{'001_':100})
    restored,codes=load_preprocessing(path)
    assert restored==scalers and codes=={'001_':100}
    raw=batch.target*batch.scales[:,None,None]+batch.means[:,None,None]
    torch.testing.assert_close(raw,torch.tensor(w.target[None],dtype=torch.float32))
    with pytest.raises(FileExistsError):save_preprocessing(path,scalers,codes)


def test_scaler_fit_ignores_future_values_and_rejects_corrupt_trusted_labels():
    p,o=panel_for('2024-07-01');start=p.hour_start_utc.min()
    a=fit_training_scalers(p,train_start=start,train_end=o)
    p.loc[p.hour_start_utc.ge(o),'ride_count']=999999
    assert a==fit_training_scalers(p,train_start=start,train_end=o)
    p.loc[p.index[0],'ride_count']=pd.NA
    with pytest.raises(ValueError,match='Trusted training'):fit_training_scalers(p,train_start=start,train_end=o)


def test_batch_adapter_ignores_masked_nan_without_creating_labels():
    p,o=panel_for('2024-07-01')
    scales=fit_training_scalers(p,train_start=p.hour_start_utc.min(),train_end=o)
    w=make_window(p,'001_',o,split_start=o,split_end=o+pd.Timedelta(hours=24),scalers=scales)
    w.target=w.target.copy();w.target_mask=w.target_mask.copy()
    w.target[::2]=np.nan;w.target_mask[::2]=False
    b=windows_to_batch([w],scales)
    assert b.target_mask.sum()==12 and torch.isfinite(b.target).all()
    assert (b.target[~b.target_mask]==0).all()  # Tensor placeholders only; mask remains false.
    assert torch.isfinite(masked_pinball_loss(model()(*b.inputs()),b.target,b.target_mask))


def test_preprocessing_rejects_schema_changes_and_invalid_scale(tmp_path):
    import json
    p,o=panel_for('2024-07-01')
    scales=fit_training_scalers(p,train_start=p.hour_start_utc.min(),train_end=o)
    path=tmp_path/'prep.json';save_preprocessing(path,scales,{'001_':100})
    original=json.loads(path.read_text())
    changed=json.loads(path.read_text());changed['known_features'].reverse();path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='schema'):load_preprocessing(path)
    original['scalers']['001_']['scale']=0;path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='scaler'):load_preprocessing(path)
