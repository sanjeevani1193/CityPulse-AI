"""Small offline development probe. Writes reports/cohort statistics, never a full panel."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil

import duckdb
import numpy as np
import pandas as pd

from citypulse_ai.etl.pipeline import _timezone_data
from citypulse_ai.features.station_history import StationHistory
from citypulse_ai.features.station_dataset import (
    NYC, KNOWN_FEATURES, OBSERVED_FEATURES, build_station_panel, make_window,
    select_candidate_stations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/historical/manifest.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise ValueError('Use a new report directory; never overwrite prior results')
    free_before = shutil.disk_usage('.').free
    history = StationHistory(args.audit, args.manifest)
    months = [f'2024-{i:02d}' for i in range(1, 8)]
    finalized = history.verify_local_months(months)
    train_start = pd.Timestamp('2024-01-01', tz=NYC).tz_convert('UTC')
    train_end = pd.Timestamp('2024-07-01', tz=NYC).tz_convert('UTC')
    activity = history.activity(train_start, train_end)
    activity['local_date'] = pd.to_datetime(activity.local_date).dt.date
    assert int(activity.ride_starts.sum()) == sum(finalized[m] for m in months[:6])
    assert all(train_start.tz_convert(NYC).date() <= d < train_end.tz_convert(NYC).date()
               for d in activity.local_date)
    selection = select_candidate_stations(activity, train_start=train_start, train_end=train_end,
                                          finalized_months=months[:6])
    subset = dict(list(selection.codes.items())[:2])
    probes = []
    peak_panel_bytes = total_panel_rows = 0
    for date in ['2024-03-10', '2024-07-01']:
        origin = pd.Timestamp(date, tz=NYC).tz_convert('UTC')
        start, stop = origin-pd.Timedelta(hours=336), origin+pd.Timedelta(hours=24)
        observed = history.hourly(subset, start, stop)
        panel = build_station_panel(observed, subset, start=start, end=stop, dataset_start=train_start)
        assert int(panel.ride_count_accepted.sum()) == int(observed.ride_count_accepted.sum())
        assert set(panel.station_id) == set(subset)
        total_panel_rows += len(panel)
        peak_panel_bytes = max(peak_panel_bytes, int(panel.memory_usage(deep=True).sum()))
        for station in subset:
            a = make_window(panel, station, origin, split_start=origin, split_end=stop)
            b = make_window(panel, station, origin, split_start=origin, split_end=stop)
            for field in ['encoder_known', 'encoder_observed', 'decoder_known', 'target', 'target_mask']:
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            assert np.isfinite(a.encoder_observed).all()
            assert a.target_hours_utc[0] == origin and len(a.target_hours_utc) == 24
            if date == '2024-03-10':
                local = a.target_hours_utc.tz_convert(NYC)
                assert 2 not in local.hour and local[-1].hour == 0
            single = panel.loc[panel.station_id.eq(station)]
            probes.append({'station_id': station, 'origin_utc': origin.isoformat(),
                'role': 'DST_calendar_diagnostic_not_historical_backtest' if date.startswith('2024-03')
                        else 'development_July_holdout_structure',
                'observed_positive_hours': int(single.ride_count_accepted.notna().sum()),
                'unknown_sparse_hours': int(single.ride_count_accepted.isna().sum()),
                'accepted_starts': int(single.ride_count_accepted.sum()),
                'trusted_label_hours': int(single.label_observed_mask.sum()),
                'availability_known_hours': int(single.availability_known_mask.sum()),
                'input_eligible': a.input_eligible,
                'primary_evaluation_eligible': a.primary_evaluation_eligible,
                'shapes': {n: list(getattr(a, n).shape) for n in
                    ['static_station_code', 'encoder_known', 'encoder_observed', 'decoder_known', 'target', 'target_mask']}})
    history._unchanged()
    report = {'scope': 'development_only_no_model_training', 'source_audit': str(args.audit.resolve()),
        'audit_sha256': history.audit_sha256,
        'timezone_rule_provenance': _timezone_data(),
        'runtime_versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'duckdb': duckdb.__version__},
        'preparation_code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('src/citypulse_ai/features/station_dataset.py'),
             Path('src/citypulse_ai/features/station_history.py')]},
        'training_accepted_starts': int(activity.ride_starts.sum()),
        'training_activity_local_dates': [str(activity.local_date.min()), str(activity.local_date.max())], 'source_manifest_sha256': history.manifest_sha256,
        'source_revision': history.identity, 'source_partitions': len(history.paths),
        'local_processed_inputs_verified': len(history.input_paths), 'finalized_accepted_start_counts': finalized,
        'coverage_note': 'Accepted-record month finalization is not a trusted-label or operating-status certificate.',
        'selection': selection.metadata, 'stations_examined': len(selection.candidates),
        'eligible_stations': int(selection.candidates.eligible.sum()), 'selected_candidate_stations': len(selection.codes),
        'selected_codes': selection.codes, 'small_probe_station_codes': subset,
        'total_small_panel_rows_examined': total_panel_rows, 'windows_examined': len(probes),
        'features': {'known': KNOWN_FEATURES, 'encoder_observed': OBSERVED_FEATURES}, 'probes': probes,
        'quality_policy': 'No real station-hour quality certificate supplied: all real labels remain unknown. '
                          'Accepted positive counts are retained for audit, not silently promoted. '
                          'No zeros or station status inferred. Synthetic tests exercise certified labels.',
        'normalization': 'No real scaler fitted: no trusted training labels. Raw-unit diagnostic arrays only.',
        'replay_assumption': 'Retrospective finalized quality-approved counts available at bucket end; no measured publication times.',
        'selection_timing': 'Jan-Jun cohort is known at June end. March probes are calendar diagnostics, not causal March forecasts.',
        'full_station_dataset_materialized': False, 'model_training': False,
        'query_duckdb_memory_limit': '512MB', 'query_spill': 'disabled',
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if platform.system() == 'Darwin' else 1024),
        'activity_dataframe_bytes': int(activity.memory_usage(deep=True).sum()),
        'largest_small_panel_bytes': peak_panel_bytes, 'free_before_bytes': free_before,
        'free_after_probe_bytes': shutil.disk_usage('.').free}
    content = selection.candidates.to_csv(index=False).encode()
    payload = (json.dumps(report, indent=2) + '\n').encode()
    if len(content) + len(payload) > 1024**2:
        raise ValueError('Prototype reports exceed the one-MiB artifact budget')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'station_candidates.csv').write_bytes(content)
    (args.output_dir/'report.json').write_bytes(payload)
    print(json.dumps({k: report[k] for k in ['stations_examined', 'eligible_stations',
        'selected_candidate_stations', 'small_probe_station_codes', 'total_small_panel_rows_examined',
        'windows_examined', 'process_peak_rss_bytes', 'largest_small_panel_bytes', 'free_before_bytes',
        'free_after_probe_bytes']}, indent=2))
    print('New report bytes:', len(content)+len(payload))
    print('Report directory:', args.output_dir)


if __name__ == '__main__':
    main()
