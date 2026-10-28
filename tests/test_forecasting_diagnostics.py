"""Synthetic diagnostic mathematics; no models or real held-out data."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from citypulse_ai.evaluation.forecasting_diagnostics import COLUMNS, activity_groups, tables, analyze


def test_training_groups_bias_temporal_fold_and_distribution():
    scalers={str(i):SimpleNamespace(mean=float(i)) for i in range(100)}
    groups=activity_groups(scalers)
    assert list(groups.values()).count('lower')==34 and list(groups.values()).count('higher')==33
    hours=pd.to_datetime(['2025-11-02T05:00:00Z','2025-11-02T06:00:00Z','2025-11-03T13:00:00Z','2025-11-01T16:00:00Z'],utc=True)
    data=pd.DataFrame([(s,h,2.) for s in scalers for h in hours],columns=['station_id','target_hour_utc','actual'])
    for col in COLUMNS.values():data[col]=data.actual+1
    result=tables(data,scalers)
    assert result['overall'].MAE.eq(1).all() and result['overall'].mean_signed_error.eq(1).all()
    assert len(result['station_id'])==600 and result['station_id'].targets.eq(4).all()
    assert result['hour'].loc[result['hour'].hour.eq(1),'targets'].eq(200).all()
    assert result['period'].loc[result['period'].period.eq('weekday_commute_peak'),'targets'].eq(100).all()
    assert activity_groups(scalers)==groups  # no test actuals determine grouping


def test_unverified_predictions_fail_before_read(tmp_path):
    p=tmp_path/'bad.parquet';p.write_bytes(b'not verified')
    with pytest.raises(ValueError,match='checksum'):
        analyze(p,tmp_path/'missing-preprocessing',tmp_path/'out',tmp_path/'figures')
    assert not (tmp_path/'out').exists()
