"""Reproduce the small existing label probe under accepted_recorded_starts_v1."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import platform
import resource
import shutil
import time

import numpy as np
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.accepted_recorded_labels import ACCEPTED_RECORDED_POLICY, AcceptedRecordedStarts
from citypulse_ai.features.station_dataset import build_station_panel, make_window
from citypulse_ai.features.station_history import StationHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-review', type=Path, required=True)
    parser.add_argument('--cohort-report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/historical/manifest.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new report directory; preserve previous policies and artifacts')
    started, free = time.perf_counter(), shutil.disk_usage('.').free
    inputs = {str(p): fingerprint(p) for p in (args.previous_review, args.cohort_report)}
    old = json.loads(args.previous_review.read_text())
    cohort = json.loads(args.cohort_report.read_text())
    if (old['source_identity'] != cohort['source_revision']
            or old['frozen_cohort_report_sha256'] != inputs[str(args.cohort_report)]):
        raise ValueError('Frozen development cohort differs from original probe')
    history = StationHistory(Path(old['evidence_reviews'][0]['audit_path']), args.manifest)
    if (history.identity != old['source_identity'] or history.manifest_sha256 != old['manifest_sha256']
            or history.audit_sha256 != old['source_audit_sha256']):
        raise ValueError('Source checkpoint differs from original probe')
    codes = old['station_codes']
    if codes != dict(list(cohort['selected_codes'].items())[:2]):
        raise ValueError('Require the original two training-selected stations')
    policy = AcceptedRecordedStarts(history, codes,
        cohort_start=cohort['selection']['train_start_utc'],
        cohort_end=cohort['selection']['train_end_utc_exclusive'])
    code_paths = [Path(__file__),Path('src/citypulse_ai/features/accepted_recorded_labels.py'),
        Path('src/citypulse_ai/features/station_labels.py'),Path('src/citypulse_ai/features/station_history.py'),
        Path('src/citypulse_ai/features/station_dataset.py')]
    code_hashes = {str(p): fingerprint(p) for p in code_paths}
    probes, certificates, decisions, lineages, diagnostics = [], [], [], [], []
    max_panel = 0
    for origin_string in sorted({p['origin_utc'] for p in old['probes']}):
        origin = pd.Timestamp(origin_string)
        start, stop = origin-pd.Timedelta(hours=336), origin+pd.Timedelta(hours=24)
        review = policy.review(start=start, end=stop)
        panel = build_station_panel(review.observed, codes, start=start, end=stop, quality=review.quality)
        assert panel.label_observed_mask.all() and not panel.availability_known_mask.any()
        assert int(panel.ride_count.sum()) == int(review.lineage.ride_count_accepted.sum())
        keys = ['station_id','hour_start_utc']
        actual, expected = panel[keys+['ride_count']].copy(), review.decisions[keys+['ride_count']].copy()
        for frame in (actual, expected):
            frame['hour_start_utc'] = frame.hour_start_utc.dt.as_unit('ns')
        pd.testing.assert_series_equal(actual.set_index(keys).ride_count.sort_index(),
            expected.set_index(keys).ride_count.sort_index(),check_dtype=False)
        max_panel = max(max_panel, int(panel.memory_usage(deep=True).sum()))
        for station in codes:
            def window(data):
                return make_window(data,station,origin,split_start=origin,split_end=stop)
            a, b = window(panel), window(panel.sample(frac=1,random_state=42))
            for name in ('encoder_known','encoder_observed','decoder_known','target','target_mask'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            changed = panel.copy()
            changed.loc[changed.hour_start_utc.ge(origin),'ride_count'] = 99999
            changed.loc[changed.hour_start_utc.ge(origin),'label_observed_mask'] = False
            changed['future_rejected_record_count'] = 99999  # Not a model feature.
            c = window(changed)
            for name in ('encoder_known','encoder_observed','decoder_known'):
                np.testing.assert_array_equal(getattr(a,name),getattr(c,name))
            single = panel.loc[panel.station_id.eq(station)]
            probes.append({'station_id':station,'origin_utc':origin.isoformat(),
                'role':'July_development_holdout_structure' if origin.month == 7
                       else 'retrospective_development_diagnostic_not_causal_backtest',
                'positive_labels':int(single.ride_count.gt(0).sum()), 'zero_labels':int(single.ride_count.eq(0).sum()),
                'masked_hours':int((~single.label_observed_mask).sum()),
                'encoder_observed_hours':int(a.encoder_observed[:,1].sum()),
                'target_observed_hours':int(a.target_mask.sum()), 'input_eligible':a.input_eligible,
                'primary_evaluation_eligible':a.primary_evaluation_eligible,
                'target_zero_count':int((a.target==0).sum()),
                'shapes':{n:list(getattr(a,n).shape) for n in
                    ('encoder_known','encoder_observed','decoder_known','target','target_mask')}})
        certificates.append(review.evidence)
        decisions.append(review.decisions.assign(probe_origin_utc=origin))
        lineages.append(review.lineage.assign(probe_origin_utc=origin))
        diagnostics.append(review.diagnostics.assign(probe_origin_utc=origin))
        print(origin.isoformat(), 'labels',int(panel.label_observed_mask.sum()),'zeros',int(panel.ride_count.eq(0).sum()),flush=True)
    d, l, q = (pd.concat(frames,ignore_index=True) for frames in (decisions,lineages,diagnostics))
    if d.duplicated(['station_id','hour_start_utc']).any():
        raise ValueError('Probe slices overlap; do not double-count coverage')
    reason_counts = Counter()
    for certificate in certificates:
        reason_counts.update(certificate['rejection_diagnostics']['overlapping_rejection_reasons'])
    history._unchanged()
    for path, digest in {**inputs,**code_hashes}.items():
        if fingerprint(Path(path)) != digest:
            raise ValueError('Preparation code or input report changed')
    report = {'policy':ACCEPTED_RECORDED_POLICY,'source_identity':history.identity,
        'audit_sha256':history.audit_sha256,'manifest_sha256':history.manifest_sha256,
        'input_report_sha256':inputs,'code_sha256':code_hashes,'station_codes':codes,
        'hours_reviewed':len(d),'positive_labels':int(d.ride_count.gt(0).sum()),
        'zero_labels':int(d.ride_count.eq(0).sum()),'masked_hours':int((~d.count_complete).sum()),
        'coverage_percent':100*float(d.count_complete.mean()),
        'old_policy_labels':old['positive_labels']+old['zero_labels'],
        'certification_change':'versioned_accepted_only_target_not_reclassification_of_raw_records',
        'forecast_target_positions':sum(p['target_observed_hours'] for p in probes),
        'forecast_target_grid':24*len(probes),'complete_24_target_windows':sum(p['primary_evaluation_eligible'] for p in probes),
        'accepted_starts':int(d.ride_count.sum()),'positive_canonical_keys_verified':len(d.loc[d.ride_count.gt(0)]),
        'zero_keys_verified_by_complete_validated_row_enumeration':int(d.ride_count.eq(0).sum()),
        'overlapping_rejected_records':sum(c['rejection_diagnostics']['unique_rejected_records_overlapping_slice'] for c in certificates),
        'overlapping_rejection_reasons':dict(reason_counts),'availability_known_hours':0,
        'diagnostics_are_model_inputs':False,'probes':probes,'certificates':certificates,
        'deterministic_windows_and_future_input_invariance':'passed',
        'queries':{'memory_limit':'512MB','spill':'disabled'},'model_training':False,
        'normalization':'raw-unit preparation diagnostics; no new preprocessing fit',
        'largest_panel_bytes':max_panel,'runtime_seconds':time.perf_counter()-started,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        'free_before_bytes':free,'free_after_bytes':shutil.disk_usage('.').free}
    def compressed(frame, keys):
        return gzip.compress(frame.sort_values(keys).to_csv(index=False).encode(),mtime=0)
    files = {'report.json':(json.dumps(report,indent=2)+'\n').encode(),
        'hourly_certificates.csv.gz':compressed(d,['station_id','hour_start_utc']),
        'positive_lineage.csv.gz':compressed(l,['station_id','hour_start_utc','source_archive_month','source_file']),
        'rejection_diagnostics.csv.gz':compressed(q,['hour_start_utc'])}
    if sum(map(len,files.values())) > 2*1024**2:
        raise ValueError('Two-MiB small diagnostic report budget exceeded')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name, content in files.items():
        with (args.output_dir/name).open('xb') as stream:
            stream.write(content)
    print(json.dumps({k:report[k] for k in ['hours_reviewed','positive_labels','zero_labels','masked_hours',
        'coverage_percent','forecast_target_positions','complete_24_target_windows','overlapping_rejected_records',
        'runtime_seconds','process_peak_rss_bytes']},indent=2))
    print('New diagnostic bytes:',sum(map(len,files.values())))


if __name__ == '__main__':
    main()
