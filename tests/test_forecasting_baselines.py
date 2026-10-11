"""Forecasts must use history available at issuance, not future observed targets."""

import math
import json

import pandas as pd
import pytest

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.forecasting.baselines import (
    LOCAL_ZONE, prepare_citywide, seasonal_naive, evaluate_midnights, error_metrics,
)
from citypulse_ai.forecasting.experiment import START, END, load_development_citywide


def history():
    hours = pd.date_range(START.tz_convert('UTC'), END.tz_convert('UTC'), freq='h', inclusive='left')
    return pd.Series(range(1, len(hours) + 1), index=hours)


def test_leap_day_and_missing_hours_are_explicit():
    observed = history().drop(history().index[50])
    frame = prepare_citywide(observed, START, END)
    assert len(frame) == 1440
    assert frame.ride_count.isna().sum() == 1
    assert not frame.is_zero_filled.any()
    assert frame.ride_count.sum() == observed.sum()
    assert frame.hour_start.dt.tz_convert(LOCAL_ZONE).dt.strftime('%Y-%m-%d').eq('2024-02-29').sum() == 24
    filled = prepare_citywide(observed, START, END, assume_missing_zero=True)
    assert filled.is_zero_filled.sum() == 1 and filled.ride_count.iloc[50] == 0


@pytest.mark.parametrize('lag', [24, 168])
def test_forecast_sources_are_correct_and_never_future(lag):
    series = history();origin = pd.Timestamp('2024-02-01', tz=LOCAL_ZONE)
    frame = seasonal_naive(series, origin, lag_hours=lag)
    assert len(frame) == 24 and frame.source_hour.lt(frame.origin).all()
    assert (frame.hour_start - frame.source_hour).eq(pd.Timedelta(hours=lag)).all()
    assert frame.prediction.tolist() == series.reindex(pd.DatetimeIndex(frame.source_hour)).tolist()
    changed = series.copy();changed.loc[changed.index >= origin] = 999999
    pd.testing.assert_frame_equal(frame, seasonal_naive(changed, origin, lag_hours=lag))


def test_lags_must_exist_before_issuance():
    series = history();origin = pd.Timestamp('2024-02-01', tz=LOCAL_ZONE)
    with pytest.raises(ValueError, match='unavailable'):
        seasonal_naive(series, START, lag_hours=168)
    with pytest.raises(ValueError, match='unavailable'):
        seasonal_naive(series.drop(origin.tz_convert('UTC') - pd.Timedelta(hours=1)), origin, lag_hours=24)
    with pytest.raises(ValueError, match='lag'):
        seasonal_naive(series, origin, lag_hours=23)


def test_midnight_splits_and_february_29_horizon():
    series = history();left = pd.Timestamp('2024-02-01', tz=LOCAL_ZONE)
    middle = pd.Timestamp('2024-02-15', tz=LOCAL_ZONE)
    validation, scores = evaluate_midnights(series, left, middle)
    test, test_scores = evaluate_midnights(series, middle, END)
    assert scores.forecast_origins.tolist() == [14, 14]
    assert scores.target_hours.tolist() == [336, 336]
    assert test_scores.forecast_origins.tolist() == [15, 15]
    assert test_scores.target_hours.tolist() == [360, 360]
    assert validation.hour_start.max() < middle and test.hour_start.min() == middle
    assert test.hour_start.max() == END.tz_convert('UTC') - pd.Timedelta(hours=1)
    assert test.origin.dt.tz_convert(LOCAL_ZONE).dt.hour.eq(0).all()
    # Linear synthetic history increases one ride/hour: exact lag error is known.
    assert scores.MAE.tolist() == [24, 168]
    assert test_scores.RMSE.tolist() == [24, 168]


def test_earlier_completed_validation_days_can_be_used_as_history():
    series = history();origin = pd.Timestamp('2024-02-02', tz=LOCAL_ZONE)
    forecast = seasonal_naive(series, origin, lag_hours=24)
    assert forecast.source_hour.min() == pd.Timestamp('2024-02-01', tz=LOCAL_ZONE)
    changed = series.copy();changed.loc[changed.index >= origin] = 100000
    assert seasonal_naive(changed, origin, lag_hours=24).prediction.equals(forecast.prediction)


def test_missing_evaluation_labels_fail_without_silent_zeros():
    series = history().drop(pd.Timestamp('2024-02-01', tz=LOCAL_ZONE))
    with pytest.raises(ValueError, match='target'):
        evaluate_midnights(series, pd.Timestamp('2024-02-01', tz=LOCAL_ZONE), pd.Timestamp('2024-02-02', tz=LOCAL_ZONE))


def test_metrics_known_example_and_zero_denominator():
    scores = error_metrics([10, 20], [13, 16])
    assert scores['MAE'] == 3.5
    assert scores['RMSE'] == pytest.approx(math.sqrt(12.5))
    assert scores['WAPE_percent'] == pytest.approx(100 * 7 / 30)
    assert math.isnan(error_metrics([0, 0], [0, 1])['WAPE_percent'])


@pytest.mark.parametrize('actual,predicted', [([], []), ([1], [1, 2]), ([math.nan], [1]), ([1], [math.inf]), ([-1], [1])])
def test_metrics_reject_invalid_arrays(actual, predicted):
    with pytest.raises(ValueError):
        error_metrics(actual, predicted)


@pytest.mark.parametrize('kind', ['naive', 'duplicate', 'fraction', 'negative', 'off_hour'])
def test_grid_rejects_invalid_observations(kind):
    series = history()
    if kind == 'naive':series.index = series.index.tz_localize(None)
    elif kind == 'duplicate':series = pd.concat([series, series.iloc[:1]])
    elif kind == 'fraction':series = series.astype(float);series.iloc[0] = 1.5
    elif kind == 'negative':series.iloc[0] = -1
    else:series.index = series.index + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):prepare_citywide(series, START, END)


def synthetic_partitions(tmp_path, *, duplicate=False):
    archives = {}
    rows = [('2024-01', 'JAN', '2024-01-01', 3),
            ('2024-02', 'BOUNDARY_JAN', '2024-01-31', 2),
            ('2024-03', 'BOUNDARY_FEB', '2024-02-29', 4)]
    for month, ride, local, count in rows:
        folder = tmp_path / month;folder.mkdir()
        hour = pd.Timestamp(local, tz=LOCAL_ZONE).tz_convert('UTC')
        # Individual IDs and aggregate counts agree; multiple stations add city demand.
        ids = [f'{ride}_{i}' for i in range(count)]
        if duplicate and month == '2024-03':ids[0] = 'JAN_0'
        pd.DataFrame({'ride_id': ids}).to_parquet(folder/'validated_trips.parquet', index=False)
        pd.DataFrame({'ride_id': pd.Series(dtype='str')}).to_csv(folder/'rejected_trips.csv', index=False)
        pd.DataFrame({'start_station_id': ['001.00_'], 'hour_start': [hour], 'ride_count': [count]}).to_parquet(folder/'hourly_rides.parquet', index=False)
        archives[month] = {'status': 'processed', 'output_dir': str(folder), 'input_rows': count,
                           'accepted_rows': count, 'rejected_rows': 0,
                           'source': {'url': 'https://example.invalid/' + month},
                           'verification': {'start_month_counts': {local[:7]: count}},
                           'output_sha256': {p.name: fingerprint(p) for p in folder.iterdir()}}
    (tmp_path/'manifest.json').write_text(json.dumps({'version': 1, 'archives': archives}))


def test_boundary_archives_conserve_counts_and_missing_hours(tmp_path):
    synthetic_partitions(tmp_path)
    city, report = load_development_citywide(tmp_path)
    assert report['local_start_month_counts'] == {'2024-01': 5, '2024-02': 4}
    assert len(city) == 1440 and city.ride_count.sum() == 9
    assert report['missing_citywide_hours'] == 1437
    assert report['cross_archive_duplicate_ids'] == 0


def test_cross_archive_duplicates_refuse_hourly_double_counting(tmp_path):
    synthetic_partitions(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='cross-archive duplicate'):
        load_development_citywide(tmp_path)


def test_changed_input_is_not_silently_trusted(tmp_path):
    synthetic_partitions(tmp_path)
    (tmp_path/'2024-01'/'rejected_trips.csv').write_text('ride_id\nCHANGED\n')
    with pytest.raises(ValueError, match='changed'):
        load_development_citywide(tmp_path)


def test_ids_in_rejection_audits_also_block_cross_archive_aggregation(tmp_path):
    synthetic_partitions(tmp_path)
    manifest_path = tmp_path/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['archives']['2024-03']
    rejected = tmp_path/'2024-03'/'rejected_trips.csv'
    rejected.write_text('ride_id\nJAN_0\n')
    entry['input_rows'] += 1
    entry['rejected_rows'] += 1
    entry['output_sha256']['rejected_trips.csv'] = fingerprint(rejected)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cross-archive duplicate'):
        load_development_citywide(tmp_path)


def test_missing_boundary_archive_is_not_treated_as_complete(tmp_path):
    synthetic_partitions(tmp_path)
    path = tmp_path/'manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['archives']['2024-03']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Complete processed source partition'):
        load_development_citywide(tmp_path)
