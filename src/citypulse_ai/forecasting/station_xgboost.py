"""Bounded pooled station XGBoost; validation-only selection, no test reader."""
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.final_station_dataset import LazyStationWindows
from citypulse_ai.features.station_dataset import KNOWN_FEATURES, utc
from citypulse_ai.forecasting.baselines import error_metrics

LAGS=(24,48,72,168,192,336)
ROLLING=(6,24,168,336)
FEATURES=('station_code','horizon_step',*KNOWN_FEATURES,
          *(name for lag in LAGS for name in (f'lag_{lag}',f'lag_{lag}_observed')),
          *(f'past_{hours}_{stat}' for hours in ROLLING for stat in ('mean','std','min','max','observed_fraction')),
          'availability_known_fraction','last_availability_known','last_station_available')
FEATURE_TYPES=('c',)+('q',)*(len(FEATURES)-1)
SEARCH=(
    {'max_depth':4,'objective':'reg:squarederror'},
    {'max_depth':6,'objective':'reg:squarederror'},
    {'max_depth':6,'objective':'reg:absoluteerror'},
)


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    temp.replace(path)


def feature_block(counts,observed,known,available,status_known,position,station_code):
    """Only [origin-336h, origin) counts; decoder calendar contains no demand.

    Status masks are already limited to evidence published by issuance. Native
    missing values stay NaN; certified observed zeros stay actual zeros.
    """
    if position<336 or position+24>len(counts):raise ValueError('Full 336/24 grid required')
    if not isinstance(station_code,(int,np.integer)) or not 1<=station_code<=100:raise ValueError('Frozen positive station code required')
    history=np.asarray(counts[position-336:position],dtype=float)
    mask=np.asarray(observed[position-336:position],dtype=bool)
    if not np.isfinite(history[mask]).all() or (history[mask]<0).any():raise ValueError('Invalid trusted history')
    history=np.where(mask,history,np.nan)
    future=np.asarray(known[position:position+24],dtype=np.float32)
    if future.shape!=(24,10) or not np.isfinite(future).all():raise ValueError('Known calendar schema differs')
    result=np.empty((24,len(FEATURES)),dtype=np.float32)
    result[:,0]=station_code;result[:,1]=np.arange(1,25)
    result[:,2:12]=future;result[:,12]=np.arange(336,360)/359
    column=13
    for lag in LAGS:
        positions=336+np.arange(24)-lag
        if (positions<0).any() or (positions>=336).any():raise ValueError('Unavailable lag at issuance')
        result[:,column]=history[positions];result[:,column+1]=mask[positions];column+=2
    for hours in ROLLING:
        values=history[-hours:];valid=values[np.isfinite(values)]
        stats=[valid.mean(),valid.std(),valid.min(),valid.max()] if len(valid) else [np.nan]*4
        result[:,column:column+5]=[*stats,len(valid)/hours];column+=5
    sm=np.asarray(status_known[position-336:position],dtype=bool)
    status=np.asarray(available[position-336:position],dtype=float)
    if not np.isfinite(status[sm]).all() or not np.isin(status[sm],[0,1]).all():raise ValueError('Invalid station status')
    result[:,column:column+3]=[sm.mean(),float(sm[-1]),status[-1] if sm[-1] else np.nan]
    assert column+3==len(FEATURES)
    return result


def paired_metrics(actual,prediction):
    actual=np.asarray(actual,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if len(actual)!=len(prediction) or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(prediction).all() or (actual<0).any():raise ValueError('Invalid paired observations')
    # Legacy forecasts are already clipped, so that helper rejects negatives.
    # Keep its exact definitions while permitting raw negative forecasts here.
    if (prediction>=0).all():scores=error_metrics(actual,prediction)
    else:
        residual=actual-prediction;denominator=float(actual.sum())
        scores={'MAE':float(np.abs(residual).mean()),'RMSE':float(np.sqrt(np.square(residual).mean())),
                'WAPE_percent':100*float(np.abs(residual).sum())/denominator if denominator else None}
    if not actual.sum():scores['WAPE_percent']=None
    return {'targets':len(actual),**scores}


def materialize(root,split,out,*,progress=lambda s:None):
    """One station frame in RAM, float32 feature memmaps; never test targets."""
    if split not in ('train','validation'):raise ValueError('Test split is not authorized')
    root=Path(root);out=Path(out);data=LazyStationWindows(root,split)
    left,right=map(utc,data.manifest['splits'][split])
    paths=[p for p in data.paths if pd.Timestamp(p.stem.removeprefix('panel-')+'-01',tz='America/New_York').tz_convert('UTC')<right]
    index=data.index.sort_values(['station_id','origin_utc']).reset_index(drop=True)
    n=len(index)*24;out.mkdir(parents=True,exist_ok=False)
    x=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float32',shape=(n,len(FEATURES)))
    y=np.lib.format.open_memmap(out/'labels.npy',mode='w+',dtype='float32',shape=(n,))
    writer=None;offset=0;digest=hashlib.sha256();trusted=0;origins=0
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        for station,group in index.groupby('station_id',sort=True):
            progress(f'features_{split}_station_{station}')
            begin=group.origin_utc.min()-pd.Timedelta(hours=336);end=group.origin_utc.max()+pd.Timedelta(hours=24)
            if begin<utc(data.manifest['splits']['train'][0]) or end>right:raise ValueError('Cross-split target boundary')
            frame=pd.read_parquet(paths,filters=[('station_id','==',station),('hour_start_utc','>=',begin),('hour_start_utc','<',end)]).sort_values('hour_start_utc').reset_index(drop=True)
            hours=pd.DatetimeIndex(frame.hour_start_utc)
            if not hours.equals(pd.date_range(begin,end,freq='h',inclusive='left')):raise ValueError('Incomplete UTC grid')
            if not frame.station_id.eq(station).all() or not frame.station_code.eq(data.codes[station]).all():raise ValueError('Station identity differs')
            counts=frame.ride_count.to_numpy(dtype=float,na_value=np.nan);mask=frame.label_observed_mask.to_numpy(dtype=bool)
            calendar=frame[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=np.float32)
            status=frame.station_available.to_numpy(dtype=float,na_value=np.nan)
            status_base=frame.availability_known_mask.to_numpy(dtype=bool)
            published=pd.to_datetime(frame.status_available_at_utc,utc=True)
            records=[]
            for row in group.itertuples(index=False):
                origin=utc(row.origin_utc)
                if origin.tz_convert('America/New_York')!=origin.tz_convert('America/New_York').normalize():raise ValueError('Midnight issuance required')
                if origin<left or origin+pd.Timedelta(hours=24)>right:raise ValueError('Targets cross split')
                position=int((origin-begin)/pd.Timedelta(hours=1))
                status_mask=status_base & published.notna().to_numpy() & published.le(origin).to_numpy()
                x[offset:offset+24]=feature_block(counts,mask,calendar,status,status_mask,position,data.codes[station])
                targets=counts[position:position+24];target_mask=mask[position:position+24]
                # Final benchmark must use exactly the TFT's complete targets.
                if not target_mask.all() or not np.isfinite(targets).all() or (targets<0).any():raise ValueError('Incomplete benchmark target labels')
                y[offset:offset+24]=targets;trusted+=int(target_mask.sum());origins+=1
                if split=='validation':
                    target_hours=hours[position:position+24]
                    for lead,hour in enumerate(target_hours):digest.update(f'{station}|{origin.isoformat()}|{hour.isoformat()}|{lead+1}\n'.encode())
                    records.append(pd.DataFrame({'station_id':station,'origin_utc':origin,'target_hour_utc':target_hours,
                        'horizon_step':np.arange(1,25),'actual':targets,
                        'lag24':counts[position-24:position],'lag168':counts[position-168:position-144]}))
                offset+=24
            if records:
                table=pa.Table.from_pandas(pd.concat(records,ignore_index=True),preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(out/'keys.parquet',table.schema,compression='zstd')
                writer.write_table(table)
            del frame
    finally:
        x.flush();y.flush()
        if writer is not None:writer.close()
    if offset!=n:raise ValueError('Window omission')
    result={'split':split,'rows':n,'forecast_windows':origins,'trusted_targets':trusted,'features':list(FEATURES),
            'dataset_manifest_sha256':fingerprint(root/'dataset.json'),'paired_key_sha256':digest.hexdigest() if split=='validation' else None}
    write_json(out/'manifest.json',result);return result


def clipped_mae(prediction,matrix):
    return 'clipped_mae',float(np.abs(matrix.get_label().astype(float)-np.maximum(prediction.astype(float),0)).mean())


def train(root,out,*,max_seconds=9000,progress=lambda s:None,search=SEARCH,rounds=300,patience=30,reference=None):
    import xgboost as xgb
    import pyarrow as pa
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    started=time.monotonic();deadline=started+max_seconds;out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def stage(name):
        if time.monotonic()>deadline-600:raise RuntimeError('Feature/search time reserve crossed')
        progress(name)
    training=materialize(root,'train',out/'scratch-train',progress=stage)
    validation=materialize(root,'validation',out/'scratch-validation',progress=stage)
    reference_verified=False
    if reference is not None:
        expected=json.loads(Path(reference).read_text())
        if (expected['dataset_manifest_sha256']!=training['dataset_manifest_sha256']
                or expected['train_windows']!=training['forecast_windows']
                or expected['validation']['forecast_windows']!=validation['forecast_windows']
                or expected['validation']['targets']!=validation['rows']
                or expected['validation']['paired_key_sha256']!=validation['paired_key_sha256']):
            raise ValueError('TFT paired dataset/target identity differs; fitting prohibited')
        keys=pd.read_parquet(out/'scratch-validation/keys.parquet')
        for lag in (24,168):
            measured=paired_metrics(keys.actual,keys[f'lag{lag}'])
            recorded=expected['validation']['baselines'][f'lag_{lag}']
            for name in ('targets','MAE','RMSE','WAPE_percent'):
                if not np.isclose(measured[name],recorded[name],rtol=0,atol=1e-9):raise ValueError('Paired baseline differs')
        del keys
        reference_verified=True
    dataset=json.loads((Path(root)/'dataset.json').read_text())
    write_json(out/'preprocessing.json',{'feature_names':list(FEATURES),'feature_types':list(FEATURE_TYPES),
        'station_codes':json.loads((Path(root)/'preprocessing.json').read_text())['station_codes'],
        'scalers':'none; raw accepted recorded counts','missing':'NaN plus explicit observation/status masks; observed zero remains zero',
        'lags':list(LAGS),'rolling_hours':list(ROLLING),'encoder_hours':336,'horizon_hours':24,
        'label_policy':'accepted_recorded_starts_v1','replay':'counts available at bucket end, matching TFT',
        'dataset_manifest_sha256':training['dataset_manifest_sha256'],'splits':dataset['splits']})
    matrices=[];raw=[]
    for split,meta in [('train',training),('validation',validation)]:
        folder=out/('scratch-'+split);features=np.load(folder/'features.npy',mmap_mode='r');labels=np.load(folder/'labels.npy',mmap_mode='r')
        kwargs={'label':labels,'feature_names':list(FEATURES),'feature_types':list(FEATURE_TYPES),'enable_categorical':True,'nthread':2,'max_bin':64}
        if matrices:kwargs['ref']=matrices[0]
        progress('quantile_matrix_'+split);matrices.append(xgb.QuantileDMatrix(features,**kwargs));raw.append((features,labels))
    class Budget(xgb.callback.TrainingCallback):
        def after_iteration(self,model,epoch,evals_log):
            if time.monotonic()>deadline-300:raise RuntimeError('Training time reserve crossed; no automatic retry')
            return False
    results=[];best=None
    for number,configuration in enumerate(search):
        if time.monotonic()>deadline-600:raise RuntimeError('Insufficient search/publication reserve')
        params={'tree_method':'hist','device':'cpu','nthread':2,'seed':42,'eta':.05,'max_bin':64,
                'subsample':1.,'colsample_bytree':1.,'disable_default_eval_metric':1,**configuration}
        progress(f'train_configuration_{number}');began=time.monotonic();log={}
        model=xgb.train(params,matrices[0],num_boost_round=rounds,evals=[(matrices[1],'validation')],
            custom_metric=clipped_mae,evals_result=log,verbose_eval=25,
            callbacks=[Budget(),xgb.callback.EarlyStopping(rounds=patience,metric_name='clipped_mae',data_name='validation',maximize=False,save_best=True)])
        prediction=model.predict(matrices[1]);actual=np.asarray(raw[1][1])
        scores={'raw':paired_metrics(actual,prediction),'clipped':paired_metrics(actual,np.maximum(prediction,0))}
        path=out/f'candidate-{number:02d}.ubj';model.save_model(path)
        record={'configuration':number,'parameters':params,'max_rounds':rounds,'early_stopping_rounds':patience,
                'selected_rounds':model.num_boosted_rounds(),'training_seconds':time.monotonic()-began,'metrics':scores,
                'model_sha256':fingerprint(path),'validation_history':log}
        results.append(record);write_json(out/'search-results.json',results)
        if best is None or scores['clipped']['MAE']<best['metrics']['clipped']['MAE']:
            best=record;model.save_model(out/'model.ubj')
        del model,prediction;gc.collect()
    progress('verify_selected_model_and_export_validation')
    model=xgb.Booster();model.load_model(out/'model.ubj')
    if model.feature_names!=list(FEATURES) or model.feature_types!=list(FEATURE_TYPES):raise ValueError('Serialized schema differs')
    prediction=model.predict(matrices[1]);again=model.predict(matrices[1]);assert np.array_equal(prediction,again)
    assert paired_metrics(raw[1][1],prediction)==best['metrics']['raw']
    frame=pd.read_parquet(out/'scratch-validation/keys.parquet');assert len(frame)==validation['rows']
    frame['prediction_raw']=prediction;frame['prediction_clipped']=np.maximum(prediction,0);frame.to_parquet(out/'validation-predictions.parquet',index=False)
    baselines={lag:paired_metrics(frame.actual,frame[lag]) for lag in ('lag24','lag168')}
    # Include complete first/last horizons and representatives across stations.
    probe_indices=np.unique(np.r_[np.arange(24),np.arange(0,len(prediction),max(1,len(prediction)//100)),np.arange(len(prediction)-24,len(prediction))])
    np.save(out/'reload-probe.npy',np.asarray(raw[1][0][probe_indices]));np.save(out/'reload-expected.npy',prediction[probe_indices])
    report={'status':'station_xgboost_validation_complete','model':'station-xgboost','framework_version':xgb.__version__,
        'dataset_manifest_sha256':training['dataset_manifest_sha256'],'dataset_identity':dataset.get('identity'),
        'train':training,'validation':validation,'search':results,'selected':best,'baselines':baselines,
        'selection_metric':'validation clipped MAE; exact ties keep earlier configuration','test_evaluated':False,'TFT_reference_verified_before_fit':reference_verified,
        'training_period':dataset['splits']['train'],'validation_period':dataset['splits']['validation'],
        'model_sha256':fingerprint(out/'model.ubj'),'model_bytes':(out/'model.ubj').stat().st_size,
        'elapsed_seconds':time.monotonic()-started,'point_clip_rule':'max(0, raw prediction)','refit_on_validation':False}
    write_json(out/'experiment.json',report);return report
