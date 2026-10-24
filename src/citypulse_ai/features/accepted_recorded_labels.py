"""Versioned closed-source-set labels; no assertion about physical demand."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.storage_state import revision_for
from citypulse_ai.etl.trips import _parse_timestamp
from citypulse_ai.features.station_dataset import NYC, utc
from citypulse_ai.features.station_labels import LabelReview, positive_decisions, rejection_hours, review_positive_labels


ACCEPTED_RECORDED_POLICY = {
    'version': 'accepted_recorded_starts_v1',
    'target': 'number_of_canonical_accepted_recorded_starts_per_station_elapsed_UTC_hour',
    'scope': 'immutable_verified_source_set_and_finalized_local_months',
    'positive': 'exact_canonical_and_validated_row_enumeration_agree',
    'zero': 'no_accepted_rows_in_verified_enumeration_for_a_valid_grid_key',
    'rejected_records': 'excluded_by_unchanged_ETL_rules_diagnostics_only',
    'station_operation': 'independent_unknown_without_status_evidence',
    'publisher_completeness': 'not_established',
    'physical_or_lost_demand': 'not_measured',
    'replay': 'retrospective_eventual_counts_available_at_bucket_end_assumption',
}


def local_months(start, end):
    return list(pd.period_range(start.tz_convert(NYC).strftime('%Y-%m'),
        (end-pd.Timedelta(nanoseconds=1)).tz_convert(NYC).strftime('%Y-%m'), freq='M').astype(str))


@dataclass
class AcceptedRecordedReview(LabelReview):
    diagnostics: pd.DataFrame


def _rejection_diagnostics(history, start, end):
    """Citywide diagnostics, never station labels or future-known features.

    Exact-hour counts and possible fold assignments are kept separate. Starts
    that cannot be localized remain explicit in the summary, not imputed.
    """
    manifest = json.loads(history.manifest_path.read_text())
    buckets = {}
    unlocalized, unlocalized_reasons = 0, Counter()
    unique_overlapping = 0
    overlapping_reasons = Counter()
    for month, entry in manifest['archives'].items():
        if entry['status'] != 'processed':
            continue
        path = Path(entry['output_dir'])/'rejected_trips.csv'
        digest = entry['output_sha256'][path.name]
        if fingerprint(path) != digest:
            raise ValueError('Rejected audit changed')
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
        for row in rows.itertuples(index=False):
            stamp, error = _parse_timestamp(row.started_at.strip())
            hours = {stamp.floor('h')} if error is None else rejection_hours([row.started_at])[0]
            reasons = row.rejection_reasons.split(';')
            if not hours:
                unlocalized += 1
                unlocalized_reasons.update(reasons)
            overlap = sorted(h for h in hours if start <= h < end)
            if overlap:
                unique_overlapping += 1
                overlapping_reasons.update(reasons)
            for hour in overlap:
                bucket = buckets.setdefault(hour, {'exact': 0, 'possible': 0,
                    'reasons_exact': Counter(), 'reasons_possible': Counter(), 'sources': set()})
                kind = 'exact' if error is None else 'possible'
                bucket[kind] += 1
                bucket['reasons_'+kind].update(reasons)
                bucket['sources'].add(month)
        if fingerprint(path) != digest:
            raise ValueError('Rejected audit changed during diagnostics')
    columns = ['hour_start_utc', 'rejected_record_count', 'possible_rejected_record_count',
               'rejection_reasons_json', 'possible_rejection_reasons_json', 'source_archives_json']
    records = [(h, b['exact'], b['possible'], json.dumps(dict(b['reasons_exact']), sort_keys=True),
                json.dumps(dict(b['reasons_possible']), sort_keys=True), json.dumps(sorted(b['sources'])))
               for h, b in sorted(buckets.items())]
    diagnostics = pd.DataFrame(records, columns=columns)
    diagnostics['hour_start_utc'] = pd.to_datetime(diagnostics.hour_start_utc, utc=True)
    summary = {'scope': 'citywide_for_source_checkpoint_not_assigned_to_each_station',
        'unique_rejected_records_overlapping_slice': unique_overlapping,
        'overlapping_rejection_reasons': dict(sorted(overlapping_reasons.items())),
        'unlocalizable_rejected_records_all_sources': unlocalized,
        'unlocalizable_reasons_all_sources': dict(sorted(unlocalized_reasons.items())),
        'unlocalizable_scope': 'time_unknown_no_claim_of_in_slice_assignment',
        'sha256': revision_for(json.loads(diagnostics.to_json(orient='records', date_format='iso')))}
    return diagnostics, summary


class AcceptedRecordedStarts:
    """Certify bounded slices for a frozen, training-supported station vocabulary.

    The existing positive review supplies the validated-row enumeration and
    fail-closed provenance checks. Its *mask decisions* are intentionally not
    used: rejected records cannot alter this accepted-only target. Nonzero
    duplicate checkpoints remain unsupported by that recount and fail closed.
    """

    def __init__(self, history, station_codes, *, cohort_start, cohort_end):
        start, end = utc(cohort_start), utc(cohort_end)
        codes = dict(station_codes)
        if (start >= end or not codes or len(set(codes.values())) != len(codes)
                or any(not isinstance(s, str) or not s.strip() or type(c) is not int or c < 1
                       for s, c in codes.items())):
            raise ValueError('Frozen exact station IDs, unique positive codes and ordered cohort bounds required')
        history._unchanged()
        history.verify_local_months(local_months(start, end))
        membership = history._query("""SELECT start_station_id AS station_id,
            sum(ride_count)::BIGINT AS training_starts
            FROM read_parquet(?, hive_partitioning=false)
            WHERE hour_start >= ? AND hour_start < ?
              AND start_station_id IN (SELECT unnest(?)) GROUP BY 1 ORDER BY 1""", [start, end, list(codes)])
        if (set(membership.station_id) != set(codes) or not membership.training_starts.gt(0).all()):
            raise ValueError('Station lacks recorded membership in the declared training interval')
        self.history = history
        self.codes = MappingProxyType(dict(sorted(codes.items(), key=lambda x: x[1])))
        self.cohort = {'station_codes': dict(self.codes), 'train_start_utc': start.isoformat(),
            'train_end_utc_exclusive': end.isoformat(), 'source_identity': history.identity,
            'membership_training_starts': dict(zip(membership.station_id, membership.training_starts.map(int)))}
        self.cohort_identity = revision_for(self.cohort)
        self.source_identity = history.identity

    def review(self, *, start, end, max_rows=10_000):
        start, end = utc(start), utc(end)
        if self.history.identity != self.source_identity or revision_for(self.cohort) != self.cohort_identity:
            raise ValueError('Source or frozen cohort identity changed')
        # This checks bounds/budget before allocating a grid, verifies boundary
        # coverage, hashes and duplicate audit, and scans all accepted inputs.
        enumeration = review_positive_labels(self.history, self.codes, start=start, end=end, max_rows=max_rows)
        proof = enumeration.evidence
        required = set(local_months(start, end))
        required |= {str(pd.Period(m, freq='M')+1) for m in tuple(required)}
        if any(not proof['sources'].get(m, {}).get('source_member_inventory_verified',
                proof['sources'].get(m, {}).get('archive_member_inventory_verified', False)) for m in required):
            raise ValueError('Required source-period inventory coverage is unverified')
        recount = enumeration.lineage.groupby(['station_id','hour_start_utc'], as_index=False).ride_count_accepted.sum()
        # Exact outer-grid comparison proves both positive counts and absence of
        # accepted rows. No raw rejected record is reclassified or assigned.
        decisions = positive_decisions(enumeration.observed, recount, stations=list(self.codes),
            start=start, end=end, blocked_hours=set(), unlocalizable=0, evidence_id='enumeration-check')
        diagnostics, diagnostic_summary = _rejection_diagnostics(self.history, start, end)
        evidence = {'policy': ACCEPTED_RECORDED_POLICY, 'source_identity': self.source_identity,
            'audit_path': proof['audit_path'], 'audit_sha256': proof['audit_sha256'],
            'manifest_sha256': proof['manifest_sha256'], 'sources': proof['sources'],
            'finalized_accepted_local_months': proof['finalized_accepted_local_months'],
            'duplicate_ride_ids': proof['duplicate_ride_ids'], 'lineage_sha256': proof['lineage_sha256'],
            'cohort': self.cohort, 'cohort_identity': self.cohort_identity,
            'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
            'grid_rows': len(decisions), 'count_comparison': 'dense_grid_equals_complete_accepted_row_enumeration',
            'positive_keys': int(decisions.ride_count_accepted.notna().sum()),
            'zero_keys': int(decisions.ride_count_accepted.isna().sum()),
            'accepted_starts': int(decisions.ride_count_accepted.sum()),
            'rejection_diagnostics': diagnostic_summary,
            'label_certification_scope': 'frozen_accepted_records_not_physical_demand',
            'station_operation_evidence': 'not_provided_unknown',
            'publication_timestamps': 'unknown_retrospective_replay_only'}
        evidence['identity'] = revision_for(evidence)
        decisions['ride_count'] = decisions.ride_count_accepted.fillna(0).astype('int64')
        decisions['count_complete'] = True
        decisions['reason'] = decisions.ride_count.map(lambda n:
            'certified_positive_accepted_recorded_starts' if n > 0 else 'certified_zero_accepted_recorded_starts')
        decisions['policy_version'] = ACCEPTED_RECORDED_POLICY['version']
        decisions['evidence_id'] = [revision_for({'certificate': evidence['identity'], 'station_id': s,
            'hour_start_utc': h.isoformat(), 'count': int(n)}) for s,h,n in
            decisions[['station_id','hour_start_utc','ride_count']].itertuples(index=False, name=None)]
        quality = decisions[['station_id','hour_start_utc','count_complete','evidence_id']].copy()
        manifest = json.loads(self.history.manifest_path.read_text())
        for month, source in proof['sources'].items():
            for name in ('validated_trips.parquet', 'rejected_trips.csv'):
                path = Path(manifest['archives'][month]['output_dir'])/name
                if fingerprint(path) != source['processed_sha256'][name]:
                    raise ValueError('Processed certification input changed')
        self.history._unchanged()
        return AcceptedRecordedReview(enumeration.observed, quality, decisions,
                                      enumeration.lineage, evidence, diagnostics)
