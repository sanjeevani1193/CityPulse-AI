"""CPU full-epoch TFT training, validation-only selection and bounded recovery.

No test split is accepted here. Architecture and preprocessing are reused as-is.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.final_station_dataset import LazyStationWindows
from citypulse_ai.features.station_dataset import make_window, utc
from citypulse_ai.forecasting.tft import (
    TFTConfig, TemporalFusionTransformer, training_step, masked_pinball_loss, inverse_quantiles,
)
from citypulse_ai.forecasting.tft_preparation import windows_to_batch


def write_json(path, value):
    """Atomic update of this run's own metadata; completed epoch files never change."""
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    temp.replace(path)


# Reviewed predecessor: same architecture, optimizer, sampler, loss and state.
# Only session stopping and additional validation reporting changed. Unknown
# driver hashes still fail closed; configuration and dataset checks are separate.
COMPATIBLE_RESUME_DRIVERS = frozenset({
    '2697566a11e6610a2f6256212e4a4cced778e0515e491eb35efd5ea7b910ecc4',
})


def validate_resume_driver(checkpoint):
    digest = checkpoint['training_code_sha256']
    if digest != fingerprint(Path(__file__)) and digest not in COMPATIBLE_RESUME_DRIVERS:
        raise ValueError('Resume code differs: unreviewed training driver')


def session_last_epoch(first_epoch, max_epochs, additional_epochs):
    if additional_epochs is not None and additional_epochs < 1:
        raise ValueError('Additional epoch cap must be positive')
    return min(max_epochs, first_epoch + additional_epochs - 1) if additional_epochs is not None else max_epochs


class StationStream:
    """Lazy station-block loading; one unique-hour station frame cached at a time.

    Shuffle station order and origins within station each epoch. This bounds I/O
    without caching repeated training tensors. The sampler's station-block order
    is an explicit optimization setting, not an architecture change.
    """
    def __init__(self,root,split):
        if split not in ('train','validation'):raise ValueError('Training never opens test targets')
        self.data=LazyStationWindows(root,split);self.split=split
        self.index=self.data.index;self.scalers=self.data.scalers
        self.cached_station=None;self.frame=None
        self.left,self.right=map(utc,self.data.manifest['splits'][split])
        # Exclude test-quarter files even from data reads. Constructor hash checks
        # validate the complete frozen bundle, but expose no test observations.
        self.paths=[p for p in self.data.paths if
            pd.Timestamp(p.stem.removeprefix('panel-')+'-01',tz='America/New_York').tz_convert('UTC')<self.right]

    def groups(self,batch_size,epoch=None,seed=42):
        if batch_size<1:raise ValueError('Positive batch size required')
        rng=np.random.default_rng(seed+(epoch or 0));stations=sorted(self.data.codes)
        if epoch is not None:rng.shuffle(stations)
        for station in stations:
            indices=np.flatnonzero(self.index.station_id.eq(station).to_numpy())
            if epoch is not None:rng.shuffle(indices)
            for start in range(0,len(indices),batch_size):yield indices[start:start+batch_size]

    def steps(self,batch_size):
        return sum(math.ceil(n/batch_size) for n in self.index.groupby('station_id').size())

    def window(self,i):
        row=self.index.iloc[int(i)];station=row.station_id
        if station!=self.cached_station:
            subset=self.index.loc[self.index.station_id.eq(station)]
            left=subset.origin_utc.min()-pd.Timedelta(hours=336)
            right=subset.origin_utc.max()+pd.Timedelta(hours=24)
            if right>self.right:raise ValueError('Read would cross split target boundary')
            self.frame=pd.read_parquet(self.paths,filters=[('station_id','==',station),
                ('hour_start_utc','>=',left),('hour_start_utc','<',right)]).sort_values('hour_start_utc').reset_index(drop=True)
            self.cached_station=station
        origin=utc(row.origin_utc)
        window=make_window(self.frame,station,origin,split_start=self.left,split_end=self.right,
            scalers=self.scalers,split_role='training' if self.split=='train' else 'evaluation')
        position=int((origin-self.frame.hour_start_utc.iloc[0])/pd.Timedelta(hours=1))
        baselines={}
        for lag in (24,168):
            raw=self.frame.iloc[position-lag:position-lag+24]
            if len(raw)!=24 or not raw.label_observed_mask.all():raise ValueError('Unavailable paired baseline')
            if not (raw.hour_start_utc+pd.Timedelta(hours=1)).le(origin).all():raise ValueError('Baseline leakage')
            baselines[lag]=raw.ride_count.to_numpy(dtype=float)
        return window,baselines

    def batch(self,indices):
        items=[self.window(i) for i in indices]
        windows=[v[0] for v in items]
        return windows_to_batch(windows,self.scalers),windows,{lag:np.stack([v[1][lag] for v in items]) for lag in (24,168)}


class Metrics:
    def __init__(self):
        self.n=0;self.absolute=self.squared=self.truth=self.covered=self.width=0.;self.pinball=np.zeros(3);self.negative=0

    def update(self,actual,quantiles,mask):
        y=np.asarray(actual,dtype=float)[np.asarray(mask,dtype=bool)];p=np.asarray(quantiles,dtype=float)[np.asarray(mask,dtype=bool)]
        if p.shape!=(len(y),3) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid paired values')
        if (np.diff(p,axis=1)<0).any():raise ValueError('Quantile crossings')
        error=y-p[:,1];q=np.array([.1,.5,.9]);residual=y[:,None]-p
        self.n+=len(y);self.absolute+=float(np.abs(error).sum());self.squared+=float((error**2).sum());self.truth+=float(y.sum())
        self.pinball+=np.maximum(q*residual,(q-1)*residual).sum(axis=0)
        self.covered+=int(((p[:,0]<=y)&(y<=p[:,2])).sum());self.width+=float((p[:,2]-p[:,0]).sum())
        self.negative+=int((p[:,1]<0).sum())

    def result(self):
        if not self.n:raise ValueError('No eligible validation targets')
        return {'targets':self.n,'MAE':self.absolute/self.n,'RMSE':math.sqrt(self.squared/self.n),
            'WAPE_percent':100*self.absolute/self.truth if self.truth else None,
            'pinball_by_quantile':dict(zip(['q10','q50','q90'],(self.pinball/self.n).tolist())),
            'mean_pinball':float(self.pinball.mean()/self.n),'interval_coverage':self.covered/self.n,
            'nominal_interval_coverage':.8,'mean_interval_width':self.width/self.n,
            'negative_median_fraction':self.negative/self.n}


def evaluate(model,stream,batch_size,*,prediction_path=None):
    if stream.split!='validation':raise ValueError('Only validation evaluation is authorized')
    model.eval();metrics=Metrics();clipped=Metrics();baselines={lag:Metrics() for lag in (24,168)}
    loss_sum=0.;valid_total=0;keys=hashlib.sha256();writer=None
    if prediction_path is not None and Path(prediction_path).exists():raise ValueError('Predictions already exist')
    started=time.monotonic()
    try:
        with torch.inference_mode():
            for indices in stream.groups(batch_size):
                batch,windows,naive=stream.batch(indices)
                prediction=model(*batch.inputs());loss=masked_pinball_loss(prediction,batch.target,batch.target_mask)
                mask=batch.target_mask.squeeze(-1).numpy();valid=int(mask.sum())
                if loss is None:continue
                loss_sum+=float(loss)*valid;valid_total+=valid
                original=inverse_quantiles(prediction,batch.means,batch.scales).numpy()
                truth=np.stack([w.target[:,0] for w in windows])
                metrics.update(truth,original,mask)
                # Predeclared validation-only rule, after inverse scaling. A
                # common nondecreasing transform preserves quantile ordering.
                clipped.update(truth,np.maximum(original,0.),mask)
                for lag in (24,168):baselines[lag].update(truth,np.repeat(naive[lag][...,None],3,axis=-1),mask)
                frames=[]
                for j,w in enumerate(windows):
                    for lead,hour in enumerate(w.target_hours_utc):
                        if mask[j,lead]:keys.update(f'{w.station_id}|{w.origin_utc.isoformat()}|{hour.isoformat()}|{lead+1}\n'.encode())
                    if prediction_path is not None:
                        frames.append(pd.DataFrame({'station_id':w.station_id,'origin_utc':w.origin_utc,
                            'target_hour_utc':w.target_hours_utc,'horizon_step':np.arange(1,25),'trusted':mask[j],
                            'actual':truth[j],'q10':original[j,:,0],'q50':original[j,:,1],'q90':original[j,:,2],
                            'lag24':naive[24][j],'lag168':naive[168][j]}))
                if frames:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    table=pa.Table.from_pandas(pd.concat(frames,ignore_index=True),preserve_index=False)
                    if writer is None:writer=pq.ParquetWriter(prediction_path,table.schema,compression='zstd')
                    writer.write_table(table)
    finally:
        if writer is not None:writer.close()
    scores=metrics.result();scores.update(standardized_masked_pinball=loss_sum/valid_total,
        forecast_windows=len(stream.index),paired_key_sha256=keys.hexdigest(),elapsed_seconds=time.monotonic()-started)
    # Point baselines have no fitted probabilistic intervals; do not present their
    # degenerate repeated quantiles as an 80% prediction interval.
    scores['nonnegative_clipped']={**clipped.result(), 'rule':'max(0, inverse_scaled_quantile)',
        'paired_key_sha256':keys.hexdigest(), 'selection_used':False}
    scores['baselines']={f'lag_{lag}':{k:v for k,v in m.result().items() if k in ('targets','MAE','RMSE','WAPE_percent')}
                        for lag,m in baselines.items()}
    return scores


def restore(path,dataset_sha):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp['dataset_manifest_sha256']!=dataset_sha:raise ValueError('Checkpoint dataset differs')
    model=TemporalFusionTransformer(TFTConfig(**cp['model_config'])).cpu();model.load_state_dict(cp['state_dict'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cp['training_config']['learning_rate']);optimizer.load_state_dict(cp['optimizer_state'])
    torch.set_rng_state(cp['torch_rng_state'])
    return cp,model,optimizer


def train(root,out,*,max_seconds=10800,batch_size=32,max_epochs=10,patience=3,resume=None,publish=None,additional_epochs=None,progress=lambda s:None):
    """Complete epochs only. Time-limited runs preserve a completed checkpoint.

    ``publish(path)`` must return a verified immutable receipt or raise. Failure
    halts the run before another epoch. No automatic resource/integrity retries.
    """
    started=time.monotonic();deadline=started+max_seconds;root=Path(root);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.use_deterministic_algorithms(True)
    import pyarrow as pa
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    progress('verify_dataset_and_load_train_validation_indices')
    training=StationStream(root,'train');validation=StationStream(root,'validation')
    sha=fingerprint(root/'dataset.json')
    config={'seed':42,'batch_size':batch_size,'max_epochs':max_epochs,'patience':patience,
        'learning_rate':.001,'min_delta':.0001,'cpu_threads':2,'sampler':'seeded_station_blocks_and_within_station_origin_shuffle',
        'gradient_clip_norm':1.,'selection_metric':'validation_standardized_masked_pinball',
        'quantiles':[.1,.5,.9],'point_output':'raw_inverse_scaled_q50_no_clipping'}
    model_config=TFTConfig();model=TemporalFusionTransformer(model_config).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];receipts=[];best=None;stale=0;first_epoch=1
    if resume:
        cp,model,optimizer=restore(resume,sha)
        if cp['training_config']!=config:raise ValueError('Resume training settings differ')
        history=cp['history'];best=cp['best'];stale=cp['stale_epochs'];first_epoch=cp['epoch']+1
        receipts=json.loads((out/'checkpoint-receipts.json').read_text())
        validate_resume_driver(cp)
    elif (out/'run.json').exists():raise ValueError('Existing run requires explicit completed-epoch resume')
    last_epoch=session_last_epoch(first_epoch,max_epochs,additional_epochs)
    run={'session_additional_epoch_cap':additional_epochs,
        'resumed_driver_sha256':cp['training_code_sha256'] if resume else None,
        'dataset_manifest_sha256':sha,'dataset_identity':training.data.manifest['identity'],
        'training_config':config,'model_config':asdict(model_config),'parameters':sum(p.numel() for p in model.parameters()),
        'train_windows':len(training.index),'validation_windows':len(validation.index),'test_evaluated':False,
        'torch':torch.__version__,'training_code_sha256':fingerprint(Path(__file__))}
    write_json(out/'run.json',run)
    progress('benchmark_real_training_batches')
    # Feasibility batches are discarded; actual optimization starts from the
    # fixed seed below. Include cold station reads across the full training set.
    benchmark=[]
    bench_model=TemporalFusionTransformer(model_config).cpu();bench_optimizer=torch.optim.Adam(bench_model.parameters(),lr=.001)
    station_groups=training.index.groupby('station_id',sort=True)
    station_names=list(station_groups.groups)
    for j in np.linspace(0,len(station_names)-1,8,dtype=int):
        indices=station_groups.groups[station_names[j]][:batch_size]
        t=time.monotonic();batch,_,_=training.batch(indices);loaded=time.monotonic()
        result=training_step(bench_model,batch,bench_optimizer)
        if result is None:raise ValueError('Benchmark has no trusted targets')
        benchmark.append({'load_seconds':loaded-t,'step_seconds':time.monotonic()-loaded,'total_seconds':time.monotonic()-t})
    del bench_model,bench_optimizer,batch
    estimate=float(np.mean([b['total_seconds'] for b in benchmark[2:]]))*training.steps(batch_size)
    run['benchmark']={'batches':benchmark,'train_steps_per_epoch':training.steps(batch_size),
        'estimated_training_epoch_seconds_conservative_cold_reads':estimate,
        'estimated_training_plus_validation_seconds':estimate*(1+len(validation.index)/len(training.index))}
    write_json(out/'run.json',run)
    # Reset all model initialization/dropout state after the disposable benchmark.
    if resume:cp,model,optimizer=restore(resume,sha)
    else:
        torch.manual_seed(42);model=TemporalFusionTransformer(model_config).cpu();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    stop_reason='session_epoch_cap' if last_epoch < max_epochs else 'max_epochs'
    for epoch in range(first_epoch,last_epoch+1):
        expected=run['benchmark']['estimated_training_plus_validation_seconds'] if not history else history[-1]['epoch_seconds']
        if time.monotonic()+expected*1.15+300>deadline:
            stop_reason='runtime_budget_before_next_epoch';break
        progress(f'train_epoch_{epoch}')
        epoch_start=time.monotonic();weighted=0.;targets=windows_seen=0
        for step,indices in enumerate(training.groups(batch_size,epoch=epoch)):
            if time.monotonic()>deadline-300:
                stop_reason='runtime_budget_partial_epoch_discarded';break
            batch,_,_=training.batch(indices);result=training_step(model,batch,optimizer)
            if result is None:raise ValueError('No valid targets in a required training batch')
            weighted+=result['loss']*result['valid_targets'];targets+=result['valid_targets'];windows_seen+=len(indices)
            if step%100==0:write_json(out/'progress.json',{'stage':f'train_epoch_{epoch}',
                'step':step,'windows_seen':windows_seen,'elapsed_epoch_seconds':time.monotonic()-epoch_start})
        if stop_reason=='runtime_budget_partial_epoch_discarded':break
        if windows_seen!=len(training.index):raise ValueError('Epoch omitted training windows')
        train_seconds=time.monotonic()-epoch_start;progress(f'validate_epoch_{epoch}')
        scores=evaluate(model,validation,batch_size)
        record={'epoch':epoch,'training_standardized_pinball':weighted/targets,'training_windows':windows_seen,
            'training_targets':targets,'training_seconds':train_seconds,'validation':scores,
            'epoch_seconds':time.monotonic()-epoch_start}
        history.append(record)
        improved=best is None or scores['standardized_masked_pinball']<best['validation_loss']-config['min_delta']
        path=out/f'epoch-{epoch:03d}.pt'
        if improved:best={'epoch':epoch,'checkpoint':path.name,'validation_loss':scores['standardized_masked_pinball']};stale=0
        else:stale+=1
        cp={'format_version':1,'epoch':epoch,'dataset_manifest_sha256':sha,'model_config':asdict(model_config),
            'training_config':config,'training_code_sha256':fingerprint(Path(__file__)),'state_dict':model.state_dict(),'optimizer_state':optimizer.state_dict(),
            'torch_rng_state':torch.get_rng_state(),'history':history,'best':best,'stale_epochs':stale}
        if path.exists():raise ValueError('Completed checkpoint already exists')
        temp=path.with_suffix('.tmp');torch.save(cp,temp);temp.rename(path)
        restored,other,_=restore(path,sha);model.eval();other.eval()
        with torch.inference_mode():
            if not torch.equal(model(*batch.inputs()),other(*batch.inputs())):raise ValueError('Checkpoint reload parity differs')
        torch.set_rng_state(restored['torch_rng_state']);del other
        receipt=publish(path) if publish else {'local_sha256':fingerprint(path)}
        receipts.append({'epoch':epoch,'checkpoint':path.name,'receipt':receipt})
        write_json(out/'history.json',history);write_json(out/'checkpoint-receipts.json',receipts)
        progress(f'epoch_{epoch}_checkpoint_verified')
        if stale>=patience:stop_reason='validation_early_stopping';break
    if not history:
        report={**run,'status':'no_complete_epoch','stop_reason':stop_reason,'history':history,'training_complete':False}
    else:
        progress('evaluate_best_completed_checkpoint')
        best_path=out/best['checkpoint'];_,model,_=restore(best_path,sha)
        predictions=out/f'best-validation-epoch-{best["epoch"]:03d}-through-{history[-1]["epoch"]:03d}-raw-clipped-v2.parquet'
        if predictions.exists():raise ValueError('Prior validation export retained; choose a new resume output export')
        scores=evaluate(model,validation,batch_size,prediction_path=predictions)
        if scores['paired_key_sha256']!=history[0]['validation']['paired_key_sha256']:raise ValueError('Paired keys changed')
        report={**run,'status':'trained_completed_epochs','history':history,'epochs_completed':len(history),
            'best':best,'validation':scores,'stop_reason':stop_reason,
            'training_complete':stop_reason in ('max_epochs','validation_early_stopping'),
            'checkpoint_receipts':receipts,'validation_predictions':predictions.name,'test_evaluated':False}
    report['elapsed_seconds']=time.monotonic()-started
    write_json(out/'experiment.json',report)
    return report
