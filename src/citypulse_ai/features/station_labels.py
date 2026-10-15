"""Conservative positive-label adjudication for an immutable archived source set.

This verifies enumeration *within that source set*, not completeness of the
publisher's trip feed. It never certifies absent rows or station operation.
"""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import duckdb
import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.history_catalog import aligned_trip_query
from citypulse_ai.etl.storage_state import revision_for
from citypulse_ai.etl.trips import _parse_timestamp
from citypulse_ai.features.station_dataset import NYC, _keys, _counts, utc


POLICY = {
    'version': 'archived-accepted-positive-v1',
    'target': 'accepted_starts_in_the_pinned_reconciled_source_set',
    'source_completeness': 'conditional_on_archived_members_not_physical_start_coverage',
    'positive': 'exact_validated_row_recount_equals_canonical_and_no_rejection_in_hour',
    'rejection_scope': 'any_rejected_start_masks_all_stations_in_its_possible_UTC_hours',
    'unlocalizable_rejection': 'mask_entire_requested_slice',
    'absent_row': 'unknown_never_zero',
    'operating_status': 'independent_unknown_without_external_evidence',
}


def rejection_hours(values):
    """Reuse ETL parsing; ambiguous fall-back starts affect both possible folds.

    Invalid/nonexistent/unlocalizable starts conservatively block the whole
    slice. No inference from end times, station names, or neighboring trips.
    """
    hours, unlocalizable = set(), 0
    for value in values:
        stamp, error = _parse_timestamp(str(value).strip())
        if error is None:
            hours.add(stamp.floor('h'))
            continue
        if error == 'ambiguous_or_nonexistent_local_time':
            naive = pd.Timestamp(str(value).strip())
            candidates = [naive.tz_localize(NYC, ambiguous=fold, nonexistent='NaT')
                          for fold in (True, False)]
            if all(pd.notna(t) for t in candidates):
                hours.update(t.tz_convert('UTC').floor('h') for t in candidates)
                continue
        unlocalizable += 1
    return hours, unlocalizable


@dataclass
class LabelReview:
    observed: pd.DataFrame
    quality: pd.DataFrame
    decisions: pd.DataFrame
    lineage: pd.DataFrame
    evidence: dict


def positive_decisions(observed, recounted, *, stations, start, end, blocked_hours,
                       unlocalizable, evidence_id):
    """Pure decision table; a positive mismatch fails rather than hiding a bug."""
    start, end = utc(start), utc(end)
    a, b = _keys(observed), _keys(recounted)
    for frame in (a, b):
        frame['hour_start_utc'] = frame.hour_start_utc.dt.as_unit('ns')
        frame['ride_count_accepted'] = _counts(frame.ride_count_accepted, positive=True)
        if (not frame.station_id.isin(stations).all()
                or not frame.hour_start_utc.ge(start).all()
                or not frame.hour_start_utc.lt(end).all()):
            raise ValueError('Counts outside reviewed station/hour bounds')
    keys = ['station_id', 'hour_start_utc']
    a = a[keys + ['ride_count_accepted']].sort_values(keys).reset_index(drop=True)
    b = b[keys + ['ride_count_accepted']].sort_values(keys).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError('Validated recount differs from canonical station-hour counts')
    hours = pd.date_range(start, end, freq='h', inclusive='left')
    grid = pd.MultiIndex.from_product([stations, hours], names=keys).to_frame(index=False)
    grid = grid.merge(a, on=keys, how='left', validate='one_to_one')
    grid['reason'] = 'absent_row_no_zero_certificate'
    positive = grid.ride_count_accepted.notna()
    grid.loc[positive, 'reason'] = 'verified_positive_archived_accepted'
    grid.loc[grid.hour_start_utc.isin(blocked_hours), 'reason'] = 'rejected_start_hour_station_uncertain'
    if unlocalizable:
        grid['reason'] = 'unlocalizable_rejection_scope'
    grid['count_complete'] = grid.reason.eq('verified_positive_archived_accepted')
    grid['evidence_id'] = [revision_for({'review': evidence_id, 'station_id': s,
        'hour_start_utc': h.isoformat(), 'count': int(c)}) if ok else None
        for s, h, c, ok in grid[keys + ['ride_count_accepted', 'count_complete']].itertuples(index=False, name=None)]
    return grid


def review_positive_labels(history, stations, *, start, end, max_rows=10_000):
    """Read only small filtered counts; no reconciliation, cloud calls or writes.

    Nonzero duplicate audits fail closed here: this adapter deliberately cannot
    bypass the canonical exclusion algorithm for those future source sets.
    """
    start, end = utc(start), utc(end)
    stations = list(stations)
    if (start >= end or not stations or len(set(stations)) != len(stations)
            or any(not isinstance(s, str) or not s.strip() for s in stations)):
        raise ValueError('Ordered bounds and unique exact station IDs required')
    if (end-start)/pd.Timedelta(hours=1)*len(stations) > max_rows:
        raise ValueError('Review exceeds explicit small-panel row budget')
    history._unchanged()
    if any(history.report[k] != 0 for k in ('duplicate_ride_ids', 'cross_archive_duplicate_ride_ids')):
        raise ValueError('Nonzero duplicate source set requires canonical exclusion-aware recount')
    months = list(pd.period_range(start.tz_convert(NYC).strftime('%Y-%m'),
        (end-pd.Timedelta(nanoseconds=1)).tz_convert(NYC).strftime('%Y-%m'), freq='M').astype(str))
    required_sources = set(months) | {str(pd.Period(m, freq='M')+1) for m in months}
    finalized = history.verify_local_months(months)
    manifest = json.loads(history.manifest_path.read_text())
    entries = {m: e for m, e in manifest['archives'].items() if e['status'] == 'processed'}
    receipts, lineage, blocked, unlocalizable, reasons = {}, [], set(), 0, Counter()

    def check_inputs():
        for entry in entries.values():
            for name in ('validated_trips.parquet', 'rejected_trips.csv'):
                if fingerprint(Path(entry['output_dir']) / name) != entry['output_sha256'][name]:
                    raise ValueError('Processed label input changed')

    check_inputs()
    with duckdb.connect(':memory:', config={'memory_limit': '512MB', 'threads': 2,
                                           'max_temp_directory_size': '0B'}) as con:
        con.execute("SET TimeZone='UTC'")
        for month, entry in entries.items():
            members = entry['source'].get('members', [])
            files, schemas = entry.get('csv_files', []), entry.get('schemas', [])
            expected = {m['name'] for m in members}
            inventory_complete = bool(expected) and (
                len(expected) == len(members) == len(files) == len(schemas)
                and {f.get('member') for f in files} == expected
                and {s.get('member') for s in schemas} == expected
                and all(f.get('sha256') for f in files))
            if (entry['input_rows'] != entry['accepted_rows'] + entry['rejected_rows']
                    or not entry['verification']['input_equals_accepted_plus_rejected']
                    or not entry['verification']['ride_counts_equal_accepted']
                    or entry['verification']['timestamp_epoch_mismatches'] != 0):
                raise ValueError('Source members/schema/conservation evidence incomplete')
            query, _ = aligned_trip_query(Path(entry['output_dir'])/'validated_trips.parquet', month)
            part = con.execute("SELECT start_station_id AS station_id, "
                "date_trunc('hour', started_at) AS hour_start_utc, source_file, "
                "count(*)::BIGINT AS ride_count_accepted FROM (" + query + ") "
                "WHERE started_at >= ? AND started_at < ? "
                "AND start_station_id IN (SELECT unnest(?)) GROUP BY 1,2,3",
                [start, end, stations]).df()
            source_paths = [f.get('path') or f['input_file'] for f in files]
            if not part.source_file.isin(source_paths).all():
                raise ValueError('Validated row has unrecognized source-member provenance')
            part['source_archive_month'] = month
            lineage.append(part)
            rejected = pd.read_csv(Path(entry['output_dir'])/'rejected_trips.csv',
                                   dtype=str, keep_default_na=False)
            if (len(rejected) != entry['rejected_rows']
                    or not rejected.source_file.isin(source_paths).all()
                    or rejected.rejection_reasons.eq('').any()):
                raise ValueError('Rejected-row audit/provenance incomplete')
            h, n = rejection_hours(rejected.started_at)
            relevant_rejections = n or any(start <= t < end for t in h)
            if not inventory_complete and (month in required_sources or len(part) or relevant_rejections):
                raise ValueError('Relevant source members/schema/conservation evidence incomplete')
            blocked.update(h);unlocalizable += n
            reasons.update(rejected.rejection_reasons)
            receipts[month] = {'source': entry['source'], 'csv_files': files,
                'input_rows': entry['input_rows'], 'accepted_rows': entry['accepted_rows'],
                'rejected_rows': entry['rejected_rows'], 'processed_sha256': entry['output_sha256'],
                'archive_member_inventory_verified': inventory_complete,
                'selected_positive_starts_in_slice': int(part.ride_count_accepted.sum()),
                'rejected_starts_affect_slice': bool(relevant_rejections),
                'source_backup_revision': entry['source_backup']['revision'],
                'output_revision': entry['cloud_publication']['revision']}
    lineage = pd.concat(lineage, ignore_index=True)
    recount = lineage.groupby(['station_id', 'hour_start_utc'], as_index=False).ride_count_accepted.sum()
    observed = history.hourly(stations, start, end)
    evidence = {'policy': POLICY, 'source_identity': history.identity,
        'audit_path': str(history.audit_path), 'audit_sha256': history.audit_sha256,
        'manifest_sha256': history.manifest_sha256, 'sources': receipts,
        'stations': stations, 'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
        'finalized_accepted_local_months': finalized, 'duplicate_ride_ids': 0,
        'lineage_sha256': revision_for(json.loads(lineage.sort_values(
            ['station_id','hour_start_utc','source_archive_month','source_file']).to_json(orient='records', date_format='iso'))),
        'rejection_reasons_all_sources': dict(reasons), 'unlocalizable_rejected_starts': unlocalizable,
        'blocked_UTC_hours_in_slice': sorted(h.isoformat() for h in blocked if start <= h < end),
        'zero_labels_certified': 0, 'physical_source_completeness_proven': False,
        'count_comparison': 'exact_equal_to_canonical',
        'replay': 'retrospective_only_eventual_approved_counts_at_bucket_end_assumption'}
    evidence['identity'] = revision_for(evidence)
    decisions = positive_decisions(observed, recount, stations=stations, start=start, end=end,
        blocked_hours=blocked, unlocalizable=unlocalizable, evidence_id=evidence['identity'])
    quality = decisions[['station_id','hour_start_utc','count_complete','evidence_id']].copy()
    check_inputs();history._unchanged()
    return LabelReview(observed, quality, decisions, lineage, evidence)
