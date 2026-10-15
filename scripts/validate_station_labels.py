"""Small, offline real-label review; existing immutable inputs only, no training."""

import argparse
import gzip
import json
from pathlib import Path
import platform
import resource
import shutil

import numpy as np
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.station_history import StationHistory
from citypulse_ai.features.station_labels import POLICY, review_positive_labels
from citypulse_ai.features.station_dataset import NYC, build_station_panel, make_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('outputs/historical/manifest.json'))
    parser.add_argument('--cohort-report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise ValueError('Use a new report directory')
    code_paths=[Path(__file__),Path('src/citypulse_ai/features/station_labels.py'),
        Path('src/citypulse_ai/features/station_history.py'),Path('src/citypulse_ai/features/station_dataset.py')]
    code_hashes={str(p):fingerprint(p) for p in code_paths}
    free=shutil.disk_usage('.').free
    history=StationHistory(args.audit,args.manifest)
    cohort=json.loads(args.cohort_report.read_text())
    if (cohort['source_revision']!=history.identity
            or cohort['selection']['train_end_utc_exclusive']!='2024-07-01T04:00:00+00:00'):
        raise ValueError('Frozen Jan–Jun development cohort provenance differs')
    codes=dict(list(cohort['selected_codes'].items())[:2])
    finalized=history.verify_local_months([f'2024-{i:02d}' for i in range(1,8)])
    summaries=[];reviews=[];decisions=[];lineages=[];max_panel=0
    for date in ['2024-01-15','2024-02-01','2024-03-10','2024-04-01','2024-05-01','2024-06-01','2024-07-01']:
        origin=pd.Timestamp(date,tz=NYC).tz_convert('UTC')
        start,stop=origin-pd.Timedelta(hours=336),origin+pd.Timedelta(hours=24)
        result=review_positive_labels(history,codes,start=start,end=stop)
        panel=build_station_panel(result.observed,codes,start=start,end=stop,quality=result.quality)
        max_panel=max(max_panel,int(panel.memory_usage(deep=True).sum()))
        assert panel.label_observed_mask.sum()==result.decisions.count_complete.sum()
        assert panel.loc[panel.label_observed_mask,'ride_count'].gt(0).all()
        assert not panel.availability_known_mask.any()
        assert int(panel.ride_count_accepted.sum())==int(result.lineage.ride_count_accepted.sum())
        for station in codes:
            a=make_window(panel,station,origin,split_start=origin,split_end=stop)
            b=make_window(panel.sample(frac=1,random_state=42),station,origin,split_start=origin,split_end=stop)
            for name in ['encoder_known','encoder_observed','decoder_known','target','target_mask']:
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            # Eventual target labels and their evidence cannot enter model inputs.
            changed=panel.copy();future=changed.hour_start_utc.ge(origin)
            changed.loc[future,'ride_count']=999999
            changed.loc[future,'label_observed_mask']=False
            c=make_window(changed,station,origin,split_start=origin,split_end=stop)
            for name in ['encoder_known','encoder_observed','decoder_known']:
                np.testing.assert_array_equal(getattr(a,name),getattr(c,name))
            if date=='2024-03-10':
                local=a.target_hours_utc.tz_convert(NYC)
                assert 2 not in local.hour and local[-1].hour==0
            subset=result.decisions.loc[result.decisions.station_id.eq(station)]
            trusted=int(subset.count_complete.sum())
            summaries.append({'station_id':station,'origin_utc':origin.isoformat(),
                'role':'July_development_holdout' if date=='2024-07-01' else 'retrospective_label_diagnostic_not_causal_backtest',
                'hours':len(subset),'positive_labels':trusted,'zero_labels':0,
                'masked_hours':len(subset)-trusted,'coverage_percent':100*trusted/len(subset),
                'mask_reasons':{k:int(v) for k,v in subset.loc[~subset.count_complete,'reason'].value_counts().items()},
                'encoder_observed_hours':int(a.encoder_observed[:,1].sum()),
                'target_observed_hours':int(a.target_mask.sum()),
                'input_eligible':a.input_eligible,'primary_evaluation_eligible':a.primary_evaluation_eligible,
                'accepted_starts_recounted':int(subset.ride_count_accepted.sum()),
                'shapes':{n:list(getattr(a,n).shape) for n in ['encoder_known','encoder_observed','decoder_known','target','target_mask']}})
        reviews.append(result.evidence)
        decisions.append(result.decisions.assign(probe_origin_utc=origin))
        lineages.append(result.lineage.assign(probe_origin_utc=origin))
        print(date, 'trusted:',int(panel.label_observed_mask.sum()),'/',len(panel),flush=True)
    d=pd.concat(decisions,ignore_index=True);l=pd.concat(lineages,ignore_index=True)
    assert not d.duplicated(['station_id','hour_start_utc']).any()
    history._unchanged()
    report={'policy':POLICY,'source_identity':history.identity,
        'source_audit_sha256':history.audit_sha256,'manifest_sha256':history.manifest_sha256,
        'frozen_cohort_report_sha256':fingerprint(args.cohort_report),'station_codes':codes,
        'finalized_accepted_start_months':finalized,'stations_examined':len(codes),
        'hours_reviewed':len(d),'positive_labels':int(d.count_complete.sum()),'zero_labels':0,
        'masked_hours':int((~d.count_complete).sum()),'coverage_percent':100*float(d.count_complete.mean()),
        'masked_reasons':{k:int(v) for k,v in d.loc[~d.count_complete,'reason'].value_counts().items()},
        'verified_positive_canonical_hours':int(d.ride_count_accepted.notna().sum()),
        'absent_sparse_hours':int(d.ride_count_accepted.isna().sum()),
        'known_operating_status_hours':0,'probes':summaries,'evidence_reviews':reviews,
        'count_comparison':'exact_equal_for_all_sparse_positive_hours_in_all_probes',
        'repeated_window_and_future_target_invariance':'passed',
        'normalization':'raw-unit development arrays; no scaler fitted or model trained',
        'all_hour_training_ready':False,'model_training':False,'physical_completeness_proven':False,
        'query_memory_limit':'512MB','query_spill':'disabled',
        'largest_panel_bytes':max_panel,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
            (1 if platform.system()=='Darwin' else 1024),
        'free_before_bytes':free,'free_after_bytes':shutil.disk_usage('.').free,
        'code_sha256':code_hashes}
    if any(fingerprint(p)!=code_hashes[str(p)] for p in code_paths):
        raise ValueError('Preparation code changed during the probe')
    files={'report.json':(json.dumps(report,indent=2)+'\n').encode(),
        'hourly_decisions.csv.gz':gzip.compress(d.to_csv(index=False).encode(),mtime=0),
        'positive_lineage.csv.gz':gzip.compress(l.to_csv(index=False).encode(),mtime=0)}
    if sum(map(len,files.values()))>2*1024**2:raise ValueError('Two-MiB report budget exceeded')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,content in files.items():(args.output_dir/name).write_bytes(content)
    print(json.dumps({k:report[k] for k in ['hours_reviewed','positive_labels','zero_labels','masked_hours',
        'coverage_percent','masked_reasons','process_peak_rss_bytes','free_before_bytes','free_after_bytes']},indent=2))
    print('Report bytes:',sum(map(len,files.values())))


if __name__=='__main__':main()
