"""CLI plumbing uses synthetic in-memory history, never local historical datasets."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import prototype_station_dataset as cli


def test_small_cli_reports_only_and_handles_duckdb_date_dtype(tmp_path, monkeypatch):
    dates=pd.date_range('2024-01-01','2024-06-30',freq='D')
    activity=pd.DataFrame([(f'S{i:03d}',day,1) for i in range(100) for day in dates],
                          columns=['station_id','local_date','ride_starts'])
    class History:
        def __init__(self,*args):
            self.audit_sha256=self.manifest_sha256=self.identity='synthetic-test-only'
            self.paths=['synthetic'];self.input_paths=['synthetic']
        def verify_local_months(self,months):
            return {m:pd.Period(m).days_in_month*100 for m in months}
        def activity(self,start,end):return activity.copy()
        def hourly(self,stations,start,end):
            hours=pd.date_range(start,end,freq='h',inclusive='left')
            return pd.DataFrame([(s,h,1) for s in stations for h in hours],
                columns=['station_id','hour_start_utc','ride_count_accepted'])
        def _unchanged(self):pass
    monkeypatch.setattr(cli,'StationHistory',History)
    destination=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['probe','--audit',str(tmp_path/'synthetic-audit.json'),
                                   '--output-dir',str(destination)])
    cli.main()
    report=json.loads((destination/'report.json').read_text())
    assert report['training_activity_local_dates']==['2024-01-01','2024-06-30']
    assert report['training_accepted_starts']==18200
    assert report['total_small_panel_rows_examined']==1440 and report['windows_examined']==4
    assert all(p['trusted_label_hours']==0 and not p['input_eligible'] for p in report['probes'])
    assert {p.name for p in destination.iterdir()}=={'report.json','station_candidates.csv'}
    assert sum(p.stat().st_size for p in destination.iterdir())<1024**2
    with pytest.raises(ValueError,match='never overwrite'):cli.main()
