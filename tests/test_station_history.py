"""Read existing tiny synthetic audit assets; no real data or network."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.test_temporary_reconciliation import backed_history, setup, audit
from citypulse_ai.features.station_history import StationHistory


def test_read_completed_boundary_counts_without_reconciliation(backed_history,tmp_path,monkeypatch):
    root,_,_,_=backed_history
    result=audit(root,tmp_path)
    before={p:p.read_bytes() for p in (tmp_path/'audits').rglob('*') if p.is_file()}
    h=StationHistory(Path(result['audit_path']),root/'manifest.json')
    assert h.verify_local_months(['2025-02'])=={'2025-02':1}
    start=pd.Timestamp('2025-02-01',tz='America/New_York');end=pd.Timestamp('2025-03-01',tz='America/New_York')
    activity=h.activity(start,end)
    assert activity.station_id.tolist()==['6569.09_'] and activity.ride_starts.tolist()==[1]
    observed=h.hourly(['6569.09_'],start,end)
    assert observed.ride_count_accepted.tolist()==[1]
    assert all(p.read_bytes()==b for p,b in before.items())
    with pytest.raises(ValueError,match='boundary'):h.verify_local_months(['2025-03'])


def test_corrupt_daily_asset_fails_closed(backed_history,tmp_path):
    root,_,_,_=backed_history;r=audit(root,tmp_path)
    h=StationHistory(Path(r['audit_path']),root/'manifest.json')
    p=Path(h.paths[0]);p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='changed'):h.activity(pd.Timestamp('2025-02-01',tz='UTC'),pd.Timestamp('2025-03-01',tz='UTC'))


def test_changed_manifest_or_incomplete_audit_rejected(backed_history,tmp_path):
    root,_,_,_=backed_history;r=audit(root,tmp_path)
    p=Path(r['audit_path']);d=json.loads(p.read_text());d['payload']['status']='incomplete';p.write_text(json.dumps(d))
    with pytest.raises(ValueError):StationHistory(p,root/'manifest.json')


def test_naive_query_bounds_and_changed_manifest_fail_closed(backed_history,tmp_path):
    root,_,_,_=backed_history;r=audit(root,tmp_path)
    h=StationHistory(Path(r['audit_path']),root/'manifest.json')
    with pytest.raises(ValueError,match='timezone'):
        h.activity(pd.Timestamp('2025-02-01'),pd.Timestamp('2025-03-01'))
    p=root/'manifest.json';d=json.loads(p.read_text());d['unexpected_revision']='changed';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='manifest'):h.activity(pd.Timestamp('2025-02-01',tz='UTC'),pd.Timestamp('2025-03-01',tz='UTC'))
