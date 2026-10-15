"""Framework-neutral station panels and causal 336/24 TFT preparation.

No implicit trusted positives/zeros, no weather, no fitted model, no I/O.
Coverage decisions are caller-reviewed evidence, never inferred from activity.
"""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NYC = ZoneInfo('America/New_York')
ENCODER_HOURS, HORIZON_HOURS = 336, 24
KNOWN_FEATURES = ('hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
                  'month_sin', 'month_cos', 'is_weekend', 'is_dst',
                  'utc_offset_hours', 'dst_fold', 'relative_position')
OBSERVED_FEATURES = ('scaled_count', 'input_count_mask', 'input_age_fraction',
                     'station_available', 'availability_known_mask')


def utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError('Explicit timezone-aware timestamp required')
    value = value.tz_convert('UTC')
    if value != value.floor('h'):
        raise ValueError('Expected elapsed-hour boundary')
    return value


def _keys(frame):
    result = frame.copy()
    if (result.station_id.isna().any()
            or not result.station_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all()):
        raise ValueError('Preserve nonblank exact string station IDs')
    if not isinstance(result.hour_start_utc.dtype, pd.DatetimeTZDtype):
        raise ValueError('Hourly keys must have an explicit timezone')
    result['hour_start_utc'] = result.hour_start_utc.dt.tz_convert('UTC')
    if (result.hour_start_utc.isna().any()
            or not result.hour_start_utc.eq(result.hour_start_utc.dt.floor('h')).all()
            or result.duplicated(['station_id', 'hour_start_utc']).any()):
        raise ValueError('Missing, off-hour or duplicate station-hour keys')
    return result


def _counts(values, *, positive=False):
    counts = pd.to_numeric(values, errors='raise')
    if (counts.isna().any() or not np.isfinite(counts.to_numpy(dtype=float)).all()
            or not counts.ge(1 if positive else 0).all() or not counts.mod(1).eq(0).all()):
        raise ValueError('Counts must be finite nonnegative integers')
    return counts.astype('int64')


class InsufficientStations(ValueError):
    def __init__(self, candidates, requested):
        self.candidates = candidates
        super().__init__(f'Only {int(candidates.eligible.sum())} eligible stations; {requested} required')


@dataclass
class StationSelection:
    candidates: pd.DataFrame
    codes: dict[str, int]
    metadata: dict


def select_candidate_stations(activity, *, train_start, train_end, finalized_months,
                              size=100, min_active_dates=15, max_inactive_dates=14):
    """Development candidate set only; all dates/counts outside training excluded.

    Activity is at least one accepted start, not evidence of operating status.
    The eventual 18-month final cohort still needs the design's quality review.
    """
    start, end = utc(train_start).tz_convert(NYC), utc(train_end).tz_convert(NYC)
    if (start >= end or any(t.day != 1 or t.hour != 0 for t in (start, end))
            or size < 1 or min_active_dates < 1 or max_inactive_dates < 0):
        raise ValueError('Use full local training months and positive selection thresholds')
    months = list(pd.period_range(start.strftime('%Y-%m'),
                                 (end - pd.Timedelta(seconds=1)).strftime('%Y-%m'), freq='M').astype(str))
    if set(months) - set(finalized_months):
        raise ValueError('Missing finalized training-source coverage')
    data = activity.copy()
    data['local_date'] = pd.to_datetime(data.local_date).dt.date
    # Filter before count/ranking statistics: July never determines Jan-Jun membership.
    data = data.loc[data.local_date.ge(start.date()) & data.local_date.lt(end.date())].copy()
    if (not data.station_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
            or data.duplicated(['station_id', 'local_date']).any()):
        raise ValueError('Exact unique station/local-date activity keys required')
    data['ride_starts'] = _counts(data.ride_starts, positive=True)
    length = (end.date() - start.date()).days
    rows = []
    for station, group in data.groupby('station_id', sort=True):
        offsets = sorted((day - start.date()).days for day in group.local_date)
        gaps = [offsets[0], length - 1 - offsets[-1]]
        gaps.extend(b - a - 1 for a, b in zip(offsets, offsets[1:]))
        monthly = group.groupby(group.local_date.map(lambda d: d.strftime('%Y-%m'))).size()
        counts = {f'active_dates_{m}': int(monthly.get(m, 0)) for m in months}
        reason = []
        if min(counts.values()) < min_active_dates: reason.append('insufficient_monthly_activity')
        if max(gaps) > max_inactive_dates: reason.append('long_no_record_gap')
        rows.append({'station_id': station, 'training_starts': int(group.ride_starts.sum()),
                     **counts, 'longest_no_record_dates': max(gaps),
                     'eligible': not reason, 'exclusion_reason': ';'.join(reason)})
    columns = ['station_id', 'training_starts', *[f'active_dates_{m}' for m in months],
               'longest_no_record_dates', 'eligible', 'exclusion_reason']
    candidates = pd.DataFrame(rows, columns=columns).sort_values(
        ['training_starts', 'station_id'], ascending=[False, True], kind='stable').reset_index(drop=True)
    candidates['rank'] = pd.Series(pd.NA, index=candidates.index, dtype='Int64')
    eligible = candidates.index[candidates.eligible.astype(bool)]
    candidates.loc[eligible, 'rank'] = np.arange(1, len(eligible) + 1)
    candidates['selected'] = candidates['rank'].le(size).fillna(False)
    if len(eligible) < size:
        raise InsufficientStations(candidates, size)
    selected = candidates.loc[candidates.selected, 'station_id'].tolist()
    return StationSelection(candidates, {s: i + 1 for i, s in enumerate(selected)}, {
        'scope': 'development_candidates_not_final_cohort', 'train_start_utc': utc(start).isoformat(),
        'train_end_utc_exclusive': utc(end).isoformat(), 'months': months,
        'min_active_dates_per_month': min_active_dates, 'max_no_record_local_dates': max_inactive_dates,
        'station_count': size, 'count_target': 'accepted_records_under_frozen_validation_policy',
        'quality_acceptance': 'activity_screen_only_not_complete_label_or_operating_status_certification',
    })


def _calendar(hours):
    local = hours.tz_convert(NYC)
    result = pd.DataFrame({'hour_start_utc': hours, 'hour_start_local': local})
    result['hour'], result['weekday'], result['month'] = local.hour, local.dayofweek, local.month
    result['is_weekend'] = local.dayofweek >= 5
    dates = local.to_pydatetime()
    result['is_dst'] = [bool(d.dst()) for d in dates]
    result['utc_offset_minutes'] = [int(d.utcoffset().total_seconds() / 60) for d in dates]
    result['utc_offset_hours'] = result.utc_offset_minutes / 60
    result['dst_fold'] = [d.fold for d in dates]
    for name, period, value in [('hour', 24, local.hour), ('weekday', 7, local.dayofweek),
                                ('month', 12, local.month - 1)]:
        result[name + '_sin'] = np.sin(2 * np.pi * value / period)
        result[name + '_cos'] = np.cos(2 * np.pi * value / period)
    return result


def build_station_panel(observed, codes, *, start, end, quality=None, status=None,
                        dataset_start=None, max_rows=1_000_000):
    """Dense UTC grid, separate audit counts/labels/status. Unknown never implies zero.

    quality: per-key count_complete bool + nonempty evidence_id + optional
    coverage_status_available_at_utc. A true decision certifies enumeration of
    accepted starts under the declared frozen policy (including absent-row zeros).
    Its substantive review is external to this builder; processed=True is not it.
    """
    start, end = utc(start), utc(end)
    if start >= end or not codes or len(set(codes.values())) != len(codes):
        raise ValueError('Nonempty unique frozen station codes and ordered bounds required')
    if any(not isinstance(s, str) or not s.strip() or not isinstance(c, int) or c < 1
           for s, c in codes.items()):
        raise ValueError('Exact string station IDs and positive integer codes required')
    hours = pd.date_range(start, end, freq='h', inclusive='left')
    if len(hours) * len(codes) > max_rows:
        raise ValueError('Requested panel exceeds explicit row budget; build smaller slices')
    data = _keys(observed)
    data['ride_count_accepted'] = _counts(data.ride_count_accepted)
    if (not data.station_id.isin(codes).all() or not data.hour_start_utc.ge(start).all()
            or not data.hour_start_utc.lt(end).all()):
        raise ValueError('Observed data must lie within frozen stations and panel bounds')
    names = ['station_id', 'hour_start_utc']
    keep = names + ['ride_count_accepted']
    if 'observation_available_at_utc' in data: keep.append('observation_available_at_utc')
    panel = pd.MultiIndex.from_product([list(codes), hours], names=names).to_frame(index=False)
    panel = panel.merge(data[keep], on=names, how='left', validate='one_to_one')
    panel['ride_count_accepted'] = panel.ride_count_accepted.astype('Int64')
    panel['count_complete'] = False
    panel['evidence_id'] = pd.NA
    if quality is not None:
        q = _keys(quality)
        q = q[names + ['count_complete', 'evidence_id'] +
              (['coverage_status_available_at_utc'] if 'coverage_status_available_at_utc' in q else [])]
        if (not q.station_id.isin(codes).all() or not q.hour_start_utc.ge(start).all()
                or not q.hour_start_utc.lt(end).all()):
            raise ValueError('Quality decisions outside frozen panel bounds')
        if not q.count_complete.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError('Coverage decisions must be explicit booleans')
        if not q.loc[q.count_complete, 'evidence_id'].map(
                lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Complete coverage needs reviewed evidence identifiers')
        panel = panel.drop(columns=['count_complete', 'evidence_id']).merge(
            q, on=names, how='left', validate='one_to_one')
        panel['count_complete'] = panel.count_complete.eq(True)
    panel['ride_count'] = panel.ride_count_accepted.where(panel.count_complete).astype('Int64')
    zero = panel.count_complete & panel.ride_count_accepted.isna()
    panel.loc[zero, 'ride_count'] = 0
    panel['label_observed_mask'] = panel.count_complete
    panel['source_coverage_status'] = np.where(panel.count_complete, 'certified_complete', 'unknown')
    panel['label_quality_reason'] = np.where(panel.ride_count_accepted.notna(),
                                            'accepted_count_unverified', 'unknown_enumeration')
    panel.loc[panel.count_complete, 'label_quality_reason'] = 'trusted_positive_complete_enumeration'
    panel.loc[panel.ride_count.eq(0).fillna(False), 'label_quality_reason'] = 'trusted_zero_complete_enumeration'
    if status is not None:
        s = _keys(status)
        s = s[names + ['station_available'] +
              (['status_available_at_utc'] if 'status_available_at_utc' in s else [])]
        if (not s.station_id.isin(codes).all() or not s.hour_start_utc.ge(start).all()
                or not s.hour_start_utc.lt(end).all()):
            raise ValueError('Status decisions outside frozen panel bounds')
        if not s.station_available.map(lambda x: pd.isna(x) or isinstance(x, (bool, np.bool_))).all():
            raise ValueError('Operating status must be boolean or unknown')
        panel = panel.merge(s, on=names, how='left', validate='one_to_one')
    if 'station_available' not in panel: panel['station_available'] = pd.NA
    panel['station_available'] = panel.station_available.astype('boolean')
    panel['availability_known_mask'] = panel.station_available.notna()
    for name in ['observation_available_at_utc', 'coverage_status_available_at_utc', 'status_available_at_utc']:
        if name not in panel: panel[name] = pd.NaT
        # Reject naive publication instants instead of guessing a timezone.
        for v in panel[name].dropna():
            if pd.Timestamp(v).tzinfo is None: raise ValueError('Publication timestamps must be timezone-aware')
        panel[name] = pd.to_datetime(panel[name], utc=True)
    panel['station_code'] = panel.station_id.map(codes).astype('int64')
    base = utc(dataset_start) if dataset_start is not None else start
    if base > start: raise ValueError('Shared dataset start cannot follow this slice')
    panel['time_idx'] = ((panel.hour_start_utc - base) / pd.Timedelta(hours=1)).astype('int64')
    return panel.merge(_calendar(hours), on='hour_start_utc', validate='many_to_one').sort_values(
        names, kind='stable').reset_index(drop=True)


@dataclass(frozen=True)
class CountScaler:
    mean: float
    scale: float
    sample_count: int
    train_start_utc: pd.Timestamp
    train_end_utc: pd.Timestamp


def fit_count_scalers(panel, *, train_start, train_end):
    """Unique trusted training hours only; never weight repeated forecast windows."""
    data = _keys(panel);start, end = utc(train_start), utc(train_end)
    if start >= end: raise ValueError('Ordered scaler training bounds required')
    result = {}
    for station in sorted(data.station_id.unique()):
        mask = (data.station_id.eq(station) & data.hour_start_utc.ge(start)
                & data.hour_start_utc.lt(end) & data.label_observed_mask)
        values = data.loc[mask, 'ride_count'].dropna().astype(float)
        if values.empty: raise ValueError(f'No trusted training labels for {station}')
        result[station] = CountScaler(float(values.mean()), float(values.std(ddof=0)) or 1.,
                                     len(values), start, end)
    return result


@dataclass
class ForecastWindow:
    station_id: str
    origin_utc: pd.Timestamp
    target_hours_utc: pd.DatetimeIndex
    static_station_code: np.ndarray
    encoder_known: np.ndarray
    encoder_observed: np.ndarray
    decoder_known: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    input_eligible: bool
    primary_evaluation_eligible: bool
    metadata: dict

    @property
    def horizon_steps(self):
        return np.arange(1, HORIZON_HOURS + 1, dtype=np.int64)

    def evaluation_keys(self):
        """One row per issued forecast, preserving DST overlaps across origins."""
        return pd.DataFrame({'station_id': self.station_id, 'origin_utc': self.origin_utc,
                             'target_hour_start_utc': self.target_hours_utc,
                             'horizon_step': self.horizon_steps})


def make_window(panel, station, origin, *, split_start, split_end, scalers=None,
                mode='retrospective', latency_hours=0, split_role='evaluation'):
    """Targets/masks are loss-only. Future demand/status never enters feature arrays.

    With scalers=None this is explicitly an unscaled development diagnostic,
    not fitted preprocessing for a model. Missing values still become finite
    zero placeholders only in arrays, always with false masks.
    """
    if split_role not in ('training', 'evaluation'): raise ValueError('Unknown split role')
    origin = utc(origin);local = origin.tz_convert(NYC)
    if local.hour != 0: raise ValueError('Forecast origin must be NYC-local midnight')
    if mode not in ('retrospective', 'operational') or not isinstance(latency_hours, int) or latency_hours < 0:
        raise ValueError('Declare replay mode and nonnegative integer latency')
    if mode == 'operational' and latency_hours: raise ValueError('Operational mode uses actual publication instants')
    stop = origin + pd.Timedelta(hours=HORIZON_HOURS)
    if origin < utc(split_start) or stop > utc(split_end):
        raise ValueError('Entire 24 elapsed-hour target must lie in one split')
    start = origin - pd.Timedelta(hours=ENCODER_HOURS)
    selected = panel.loc[panel.station_id.eq(station) & panel.hour_start_utc.ge(start)
                         & panel.hour_start_utc.lt(stop)]
    block = _keys(selected).sort_values('hour_start_utc')
    hours = pd.date_range(start, stop, freq='h', inclusive='left')
    if len(block) != 360 or not pd.DatetimeIndex(block.hour_start_utc).equals(hours):
        raise ValueError('Full encoder/decoder grid required; no pre-history padding')
    if block.station_code.nunique() != 1:
        raise ValueError('Station code changed inside a window')
    _counts(block.loc[block.label_observed_mask, 'ride_count'])
    encoder, target = block.iloc[:336], block.iloc[336:]
    known = block[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=np.float32)
    known = np.column_stack([known, np.arange(360, dtype=np.float32) / 359])
    ended = encoder.hour_start_utc + pd.Timedelta(hours=1)
    if mode == 'retrospective':
        available = ended + pd.Timedelta(hours=latency_hours)
        accessible = encoder.label_observed_mask & available.le(origin)
    else:
        accessible = (encoder.label_observed_mask & ended.le(origin)
                      & encoder.observation_available_at_utc.notna()
                      & encoder.observation_available_at_utc.le(origin)
                      & encoder.coverage_status_available_at_utc.notna()
                      & encoder.coverage_status_available_at_utc.le(origin))
    count_mask = accessible.to_numpy(dtype=bool)
    values = np.zeros(336, dtype=np.float32)
    if scalers is not None:
        scaler = scalers[station]
        if (scaler.sample_count < 1 or scaler.train_start_utc >= scaler.train_end_utc
                or (split_role == 'evaluation' and scaler.train_end_utc > utc(split_start))
                or (split_role == 'training' and (scaler.train_start_utc < utc(split_start)
                                                or scaler.train_end_utc > utc(split_end)))):
            raise ValueError('Scaler fitting period leaks beyond the permitted training interval')
        if not np.isfinite([scaler.mean, scaler.scale]).all() or scaler.scale <= 0:
            raise ValueError('Invalid frozen scaler')
        values[count_mask] = (encoder.loc[accessible, 'ride_count'].to_numpy(dtype=float)
                              - scaler.mean) / scaler.scale
    else:
        values[count_mask] = encoder.loc[accessible, 'ride_count'].to_numpy(dtype=float)
    ages = np.full(336, 336., dtype=np.float32);last = None
    for i, valid in enumerate(count_mask):
        if valid: last = i
        if last is not None: ages[i] = min(336, i - last)
    status_mask = (encoder.availability_known_mask & encoder.status_available_at_utc.notna()
                   & encoder.status_available_at_utc.le(origin)).to_numpy(dtype=bool)
    statuses = np.zeros(336, dtype=np.float32)
    statuses[status_mask] = encoder.loc[status_mask, 'station_available'].to_numpy(dtype=float)
    observed = np.column_stack([values, count_mask, ages / 336, statuses, status_mask]).astype('float32')
    target_mask = target.label_observed_mask.to_numpy(dtype=bool).reshape(24, 1)
    labels = target.ride_count.fillna(0).to_numpy(dtype=np.float32).reshape(24, 1)
    labels[~target_mask] = 0  # Placeholder, never a zero-label assertion.
    return ForecastWindow(station, origin, hours[336:],
        np.array([int(block.station_code.iloc[0])], dtype=np.int64), known[:336], observed,
        known[336:], labels, target_mask, bool(count_mask.any()),
        bool(count_mask.any() and target_mask.all()),
        {'replay': mode, 'latency_hours': latency_hours,
         'observation_assumption': 'finalized_quality_approved_counts_at_bucket_end' if mode == 'retrospective'
                                  else 'recorded_publication_instants_required',
         'scaling': 'frozen_training_only' if scalers is not None else 'raw_units_development_only',
         'feature_order': {'known': KNOWN_FEATURES, 'observed': OBSERVED_FEATURES}})


def origin_manifest(*, data_start, data_end, split_start, split_end):
    """NYC calendar midnights, including spring overlap/fall gap of target UTC hours."""
    left, right = utc(split_start), utc(split_end)
    if left >= right: raise ValueError('Ordered split boundaries required')
    dates = pd.date_range(left.tz_convert(NYC).date(), right.tz_convert(NYC).date(), tz=NYC, freq='D')
    rows = []
    for local in dates:
        origin = local.tz_convert('UTC')
        if origin < left or origin >= right: continue
        reasons = []
        if origin - pd.Timedelta(hours=336) < utc(data_start): reasons.append('encoder_warmup')
        if origin + pd.Timedelta(hours=24) > min(right, utc(data_end)): reasons.append('target_boundary')
        rows.append({'origin_utc': origin, 'origin_local_date': local.date(),
                     'structurally_eligible': not reasons, 'exclusion_reason': ';'.join(reasons)})
    return pd.DataFrame(rows, columns=['origin_utc', 'origin_local_date',
                                       'structurally_eligible', 'exclusion_reason'])


def iter_windows(panel, *, split_start, split_end, **window_options):
    """Yield one station-origin sample at a time; do not store repeated histories.

    Structural exclusions come from origin_manifest. Input/label ineligibility
    remains visible on each returned sample instead of silently dropping rows.
    """
    data = _keys(panel)
    if data.empty: return
    origins = origin_manifest(data_start=data.hour_start_utc.min(),
        data_end=data.hour_start_utc.max()+pd.Timedelta(hours=1),
        split_start=split_start, split_end=split_end)
    stations = data[['station_id','station_code']].drop_duplicates().sort_values('station_code')
    if stations.station_id.duplicated().any(): raise ValueError('Station codes changed in panel')
    for origin in origins.loc[origins.structurally_eligible.astype(bool), 'origin_utc']:
        for station in stations.station_id:
            yield make_window(data, station, origin, split_start=split_start,
                              split_end=split_end, **window_options)
