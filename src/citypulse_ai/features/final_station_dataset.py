"""Bounded final station panels and lazy windows; no model training or cloud I/O."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from citypulse_ai.etl.archives import fingerprint, month_range
from citypulse_ai.etl.storage_state import revision_for
from citypulse_ai.features.accepted_recorded_labels import AcceptedRecordedStarts, ACCEPTED_RECORDED_POLICY
from citypulse_ai.features.station_dataset import (
    NYC, KNOWN_FEATURES, OBSERVED_FEATURES, CountScaler, build_station_panel,
    fit_count_scalers, make_window, origin_manifest, select_candidate_stations, utc,
)

SPLITS = {
    'train': ['2024-01-01T00:00:00-05:00', '2025-07-01T00:00:00-04:00'],
    'validation': ['2025-07-01T00:00:00-04:00', '2025-10-01T00:00:00-04:00'],
    'test': ['2025-10-01T00:00:00-04:00', '2026-01-01T00:00:00-05:00'],
}


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def file_receipt(path):
    path = Path(path)
    return {'name': path.name, 'size_bytes': path.stat().st_size, 'sha256': fingerprint(path)}


def load_preprocessing(path):
    """Same saved schema as tft_preparation, without importing a training framework."""
    d = json.loads(Path(path).read_text())
    if (d['format_version'] != 1 or d['known_features'] != list(KNOWN_FEATURES)
            or d['observed_features'] != list(OBSERVED_FEATURES)):
        raise ValueError('Preprocessing schema differs')
    codes = d['station_codes']
    if (not codes or set(codes) != set(d['scalers']) or len(set(codes.values())) != len(codes)
            or any(not isinstance(s, str) or not s.strip() or type(c) is not int or c < 1
                   for s,c in codes.items())):
        raise ValueError('Invalid station vocabulary')
    scalers = {}
    for station, v in d['scalers'].items():
        s = CountScaler(v['mean'], v['scale'], v['sample_count'],
                        utc(v['train_start_utc']), utc(v['train_end_utc']))
        if (not np.isfinite([s.mean,s.scale]).all() or s.scale <= 0
                or type(s.sample_count) is not int or s.sample_count < 1
                or s.train_start_utc >= s.train_end_utc):
            raise ValueError('Invalid scaler')
        scalers[station] = s
    return scalers, codes


def fit_partition_scalers(paths, codes, *, train_start, train_end, destination):
    """Read one station's unique training hours at a time, never window-weighted."""
    start,end = utc(train_start),utc(train_end)
    scalers = {}
    for station in codes:
        frame = pd.read_parquet(paths, columns=['station_id','hour_start_utc','ride_count',
            'label_observed_mask'], filters=[('station_id','==',station),
            ('hour_start_utc','>=',start),('hour_start_utc','<',end)])
        scalers.update(fit_count_scalers(frame, train_start=start, train_end=end))
    data = {'format_version':1,'station_codes':codes,'known_features':list(KNOWN_FEATURES),
        'observed_features':list(OBSERVED_FEATURES),'scalers':{s:{**asdict(v),
            'train_start_utc':v.train_start_utc.isoformat(),
            'train_end_utc':v.train_end_utc.isoformat()} for s,v in scalers.items()}}
    write_json(destination,data)
    loaded,loaded_codes = load_preprocessing(destination)
    assert loaded == scalers and loaded_codes == codes
    return scalers


def index_station(frame, station, code, *, data_start, data_end, splits):
    """Index every structurally eligible issue; retain input/target eligibility flags."""
    start,end = utc(data_start),utc(data_end)
    frame=frame.sort_values('hour_start_utc')
    hours=pd.date_range(start,end,freq='h',inclusive='left')
    if (set(frame.station_id) != {station} or len(frame)!=len(hours)
            or not pd.DatetimeIndex(frame.hour_start_utc).equals(hours)):
        raise ValueError('Station panel grid missing, duplicated or out of bounds')
    if not frame.label_observed_mask.map(lambda x:isinstance(x,(bool,np.bool_))).all():
        raise ValueError('Explicit observation masks required')
    mask=frame.label_observed_mask.to_numpy(dtype=bool)
    prefix=np.r_[0,np.cumsum(mask)]
    rows=[];exclusions=[]
    for split,(left,right) in splits.items():
        origins=origin_manifest(data_start=start,data_end=end,split_start=left,split_end=right)
        for r in origins.itertuples(index=False):
            if not r.structurally_eligible:
                exclusions.append({'split':split,'origin_utc':r.origin_utc,
                    'reason':r.exclusion_reason});continue
            i=int((r.origin_utc-start)/pd.Timedelta(hours=1))
            encoder=int(prefix[i]-prefix[i-336]);targets=int(prefix[i+24]-prefix[i])
            rows.append({'station_id':station,'station_code':code,'split':split,
                'origin_utc':r.origin_utc,'encoder_start_idx':i-336,'target_start_idx':i,
                'target_stop_idx':i+24,'encoder_observed_hours':encoder,'trusted_targets':targets,
                'input_eligible':encoder>0,'training_eligible':encoder>0 and targets>0,
                'primary_evaluation_eligible':encoder>0 and targets==24})
    return pd.DataFrame(rows),pd.DataFrame(exclusions)


class LazyStationWindows:
    """Random access to existing ForecastWindow objects, at most 360 rows per read.

    Hash all artifacts once on open. Treat the published dataset directory as
    immutable afterwards. Call verify() again after transfer or suspected changes.
    """
    def __init__(self, root, split=None):
        self.root=Path(root);self.manifest=json.loads((self.root/'dataset.json').read_text())
        if self.manifest['status']!='complete':raise ValueError('Incomplete dataset')
        self.verify()
        self.scalers,self.codes=load_preprocessing(self.root/'preprocessing.json')
        self.paths=[self.root/r['name'] for r in self.manifest['panels']]
        self.index=pd.read_parquet(self.root/'windows.parquet')
        if split is not None:
            if split not in self.manifest['splits']:raise ValueError('Unknown split')
            self.index=self.index.loc[self.index.split.eq(split)].reset_index(drop=True)

    def verify(self):
        for r in self.manifest['artifacts']:
            p=self.root/r['name']
            if (Path(r['name']).name!=r['name'] or p.is_symlink()
                    or p.stat().st_size!=r['size_bytes'] or fingerprint(p)!=r['sha256']):
                raise ValueError('Dataset artifact changed: '+r['name'])

    def __len__(self):return len(self.index)

    def __getitem__(self,i):
        r=self.index.iloc[i];origin=utc(r.origin_utc)
        panel=pd.read_parquet(self.paths,filters=[('station_id','==',r.station_id),
            ('hour_start_utc','>=',origin-pd.Timedelta(hours=336)),
            ('hour_start_utc','<',origin+pd.Timedelta(hours=24))])
        left,right=self.manifest['splits'][r.split]
        return make_window(panel,r.station_id,origin,split_start=left,split_end=right,
            scalers=self.scalers,split_role='training' if r.split=='train' else 'evaluation')


def build_final_dataset(history, destination, *, progress=lambda stage:None):
    """Final fixed contract. Resume only complete, hash-verified quarter checkpoints."""
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    start,end=utc(SPLITS['train'][0]),utc(SPLITS['test'][1]);train_end=utc(SPLITS['train'][1])
    progress('verify_complete_history')
    coverage=history.verify_source_inventories(month_range('2024-01-01','2026-01-01'))
    identity=revision_for({'source_identity':history.identity,'manifest_sha256':history.manifest_sha256,
        'csv_inventories':{m:sha for m,(_,sha) in history.csv_inventories.items()},
        'splits':SPLITS,'policy':ACCEPTED_RECORDED_POLICY,'encoder':336,'horizon':24,
        'selection':{'size':100,'min_active_dates':15,'max_inactive_dates':14},
        'builder_sha256':fingerprint(Path(__file__))})
    plan={'identity':identity,'source_identity':history.identity,'splits':SPLITS}
    plan_path=out/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing build identity differs')
    else:write_json(plan_path,plan)
    if (out/'dataset.json').exists():
        dataset=LazyStationWindows(out)
        if dataset.manifest['identity']!=identity:raise ValueError('Completed build identity differs')
        return dataset.manifest
    progress('select_training_only_stations')
    activity=history.activity(start,train_end)
    if int(activity.ride_starts.sum()) != sum(history.report['start_month_counts'][m] for m in
            month_range('2024-01-01','2025-07-01')):
        raise ValueError('Training activity totals differ from finalized history')
    selection=select_candidate_stations(activity,train_start=start,train_end=train_end,
        finalized_months=month_range('2024-01-01','2025-07-01'))
    del activity
    candidates=out/'station_candidates.csv';cohort=out/'cohort.json'
    cohort_data={'station_codes':selection.codes,'metadata':{**selection.metadata,
        'scope':'final_training_only_100_station_cohort'},'source_identity':history.identity,
        'stations_examined':len(selection.candidates),'eligible_stations':int(selection.candidates.eligible.sum())}
    if cohort.exists():
        if json.loads(cohort.read_text())!=cohort_data or candidates.read_text()!=selection.candidates.to_csv(index=False):
            raise ValueError('Training cohort changed on resume')
    else:
        selection.candidates.to_csv(candidates,index=False);write_json(cohort,cohort_data)
    policy=AcceptedRecordedStarts(history,selection.codes,cohort_start=start,cohort_end=train_end)
    panels=[];statistics=[]
    boundaries=pd.date_range('2024-01-01','2026-01-01',freq='QS',tz=NYC)
    for left,right in zip(boundaries,boundaries[1:]):
        tag=left.strftime('%Y-%m');progress('certify_panel_'+tag)
        checkpoint=out/f'quarter-{tag}.json';panel_path=out/f'panel-{tag}.parquet';evidence_path=out/f'evidence-{tag}.json'
        if checkpoint.exists():
            cp=json.loads(checkpoint.read_text())
            if cp['identity']!=identity:raise ValueError('Quarter checkpoint identity differs')
            for r in cp['files']:
                if file_receipt(out/r['name'])!=r:raise ValueError('Quarter checkpoint artifact changed')
            panels.append(cp['files'][0]);statistics.append(cp['statistics']);continue
        if panel_path.exists() or evidence_path.exists():
            raise ValueError('Incomplete uncheckpointed quarter retained; inspect before restart')
        review=policy.review(start=left,end=right,max_rows=225_000)
        panel=build_station_panel(review.observed,selection.codes,start=left,end=right,
            quality=review.quality,dataset_start=start,max_rows=225_000)
        if (not panel.label_observed_mask.all() or panel.ride_count.isna().any()
                or int(panel.ride_count.sum())!=review.evidence['accepted_starts']):
            raise ValueError('Certified panel conservation failed')
        # Keep rejected diagnostics in a separate audit artifact, never model inputs.
        evidence={**review.evidence,'diagnostic_rows':json.loads(review.diagnostics.to_json(orient='records',date_format='iso'))}
        stats={'rows':len(panel),'positive':int(panel.ride_count.gt(0).sum()),
            'zero':int(panel.ride_count.eq(0).sum()),'unknown':int((~panel.label_observed_mask).sum()),
            'availability_known':int(panel.availability_known_mask.sum()),'accepted_starts':int(panel.ride_count.sum())}
        panel.to_parquet(panel_path,index=False,row_group_size=744)
        write_json(evidence_path,evidence)
        cp={'identity':identity,'files':[file_receipt(panel_path),file_receipt(evidence_path)],'statistics':stats}
        write_json(checkpoint,cp);panels.append(cp['files'][0]);statistics.append(stats)
        del review,panel,evidence
    paths=[out/r['name'] for r in panels]
    progress('fit_training_only_preprocessing')
    preprocessing=out/'preprocessing.json'
    if not preprocessing.exists():
        fit_partition_scalers(paths,selection.codes,train_start=start,train_end=train_end,destination=preprocessing)
    scalers,codes=load_preprocessing(preprocessing)
    expected_train_hours=int((train_end-start)/pd.Timedelta(hours=1))
    if set(codes)!=set(selection.codes) or any(s.sample_count!=expected_train_hours or
            s.train_start_utc!=start or s.train_end_utc!=train_end for s in scalers.values()):
        raise ValueError('Training scaler coverage/boundaries differ')
    progress('index_lazy_forecast_windows')
    indices=[];excluded=None;split_stats={k:{'rows':0,'positive':0,'zero':0,'unknown':0,'accepted_starts':0} for k in SPLITS}
    for station,code in selection.codes.items():
        frame=pd.read_parquet(paths,columns=['station_id','hour_start_utc','ride_count','label_observed_mask'],
                              filters=[('station_id','==',station)])
        index,exc=index_station(frame,station,code,data_start=start,data_end=end,splits=SPLITS)
        indices.append(index);excluded=exc
        for split,(a,b) in SPLITS.items():
            f=frame.loc[frame.hour_start_utc.ge(utc(a)) & frame.hour_start_utc.lt(utc(b))]
            stats=split_stats[split];stats['rows']+=len(f);stats['positive']+=int(f.ride_count.gt(0).sum())
            stats['zero']+=int(f.ride_count.eq(0).sum());stats['unknown']+=int((~f.label_observed_mask).sum())
            stats['accepted_starts']+=int(f.ride_count.sum())
    index=pd.concat(indices,ignore_index=True).sort_values(['origin_utc','station_code']).reset_index(drop=True)
    if index.duplicated(['station_id','origin_utc']).any():raise ValueError('Duplicate forecast keys')
    for split in SPLITS:
        f=index.loc[index.split.eq(split)];split_stats[split].update(windows=len(f),
            usable_training_windows=int(f.training_eligible.sum()),paired_evaluation_windows=int(f.primary_evaluation_eligible.sum()),
            trusted_target_positions=int(f.trusted_targets.sum()))
    for name,frame in [('windows.parquet',index),('excluded-origins.parquet',excluded)]:
        if (out/name).exists():
            if not pd.read_parquet(out/name).equals(frame):raise ValueError('Index changed on resume')
        else:frame.to_parquet(out/name,index=False)
    history._unchanged();progress('verify_dataset_artifacts')
    artifacts=[file_receipt(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='dataset.json']
    manifest={'format_version':1,'status':'complete','identity':identity,'source_identity':history.identity,
        'source_manifest_sha256':history.manifest_sha256,'source_audit_sha256':history.audit_sha256,
        'source_inventories':coverage,'policy':ACCEPTED_RECORDED_POLICY,'cohort':cohort_data,
        'splits':SPLITS,'split_statistics':split_stats,'panels':panels,'quarter_statistics':statistics,
        'total_rows':sum(s['rows'] for s in statistics),'encoder_hours':336,'horizon_hours':24,
        'known_features':list(KNOWN_FEATURES),'observed_features':list(OBSERVED_FEATURES),
        'replay':'retrospective_eventual_counts_at_bucket_end','availability':'unknown_without_independent_status',
        'artifacts':artifacts,'training_performed':False,'tensors_materialized':False}
    write_json(out/'dataset.json',manifest)
    dataset=LazyStationWindows(out)
    for split in SPLITS:
        positions=np.flatnonzero(dataset.index.split.eq(split).to_numpy())
        for i in [positions[0],positions[-1]]:
            w=dataset[int(i)]
            if not w.input_eligible or not w.target_mask.all():raise ValueError('Lazy boundary window invalid')
            assert w.encoder_known.shape==(336,11) and w.encoder_observed.shape==(336,5)
            assert w.decoder_known.shape==(24,11) and w.target.shape==(24,1)
    return manifest
