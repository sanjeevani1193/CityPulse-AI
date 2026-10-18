"""Small CLI regression: station-code order need not be lexical panel order."""

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from citypulse_ai.etl.archives import fingerprint
from scripts import validate_accepted_recorded_labels as cli


def test_probe_compares_labels_by_keys_and_keeps_diagnostics_separate(tmp_path,monkeypatch):
    codes={'Z':1,'A':2}
    origin=pd.Timestamp('2024-07-01T04:00Z')
    cohort=tmp_path/'cohort.json'
    cohort.write_text(json.dumps({'source_revision':'synthetic','selected_codes':codes,
        'selection':{'train_start_utc':'2024-01-01T05:00Z','train_end_utc_exclusive':origin.isoformat()}}))
    previous=tmp_path/'previous.json'
    previous.write_text(json.dumps({'source_identity':'synthetic','manifest_sha256':'manifest',
        'source_audit_sha256':'audit','frozen_cohort_report_sha256':fingerprint(cohort),
        'evidence_reviews':[{'audit_path':'synthetic-audit'}], 'station_codes':codes,
        'probes':[{'origin_utc':origin.isoformat()}], 'positive_labels':2,'zero_labels':0}))
    class History:
        def __init__(self,*args):
            self.identity='synthetic';self.audit_sha256='audit';self.manifest_sha256='manifest'
        def _unchanged(self):pass
    class Policy:
        def __init__(self,*args,**kwargs):pass
        def review(self,*,start,end):
            hours=pd.date_range(start,end,freq='h',inclusive='left')
            grid=pd.MultiIndex.from_product([list(codes),hours],names=['station_id','hour_start_utc']).to_frame(index=False)
            grid['ride_count']=0
            grid.loc[grid.hour_start_utc.eq(origin)&grid.station_id.eq('Z'),'ride_count']=2
            grid.loc[grid.hour_start_utc.eq(origin)&grid.station_id.eq('A'),'ride_count']=7
            grid['count_complete']=True;grid['evidence_id']='synthetic-certified'
            observed=grid.loc[grid.ride_count.gt(0),['station_id','hour_start_utc','ride_count']].rename(columns={'ride_count':'ride_count_accepted'})
            lineage=observed.assign(source_archive_month='2024-07',source_file='synthetic.csv')
            diagnostics=pd.DataFrame({'hour_start_utc':[origin], 'rejected_record_count':[1]})
            return SimpleNamespace(observed=observed,quality=grid,decisions=grid,lineage=lineage,
                diagnostics=diagnostics,evidence={'rejection_diagnostics':{
                    'overlapping_rejection_reasons':{'missing_start_station_id':1},
                    'unique_rejected_records_overlapping_slice':1}})
    monkeypatch.setattr(cli,'StationHistory',History);monkeypatch.setattr(cli,'AcceptedRecordedStarts',Policy)
    out=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['review','--previous-review',str(previous),'--cohort-report',str(cohort),'--output-dir',str(out)])
    cli.main()
    report=json.loads((out/'report.json').read_text())
    assert report['hours_reviewed']==720 and report['positive_labels']==2 and report['zero_labels']==718
    assert report['forecast_target_positions']==48 and report['complete_24_target_windows']==2
    assert report['overlapping_rejected_records']==1 and report['diagnostics_are_model_inputs'] is False
    assert set(p.name for p in out.iterdir())=={'report.json','hourly_certificates.csv.gz','positive_lineage.csv.gz','rejection_diagnostics.csv.gz'}
    with pytest.raises(ValueError,match='preserve previous'):cli.main()
