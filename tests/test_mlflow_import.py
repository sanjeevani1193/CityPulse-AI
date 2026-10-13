"""Tiny synthetic metadata and real temporary SQLite stores; no model fitting."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from citypulse_ai.experiments.mlflow_import import build_plans, import_plans


def fixture(root):
    (root/'contract.md').write_text('Synthetic contract')
    record = {'metrics': [{'model':'Lag 24', 'split':split, 'MAE':mae, 'RMSE':mae+1,
                          'WAPE_percent':mae+2, 'forecast_origins':2, 'target_hours':48}
                         for split,mae in [('validation',3.), ('test',7.)]],
              'provenance':{'manifest_sha256':'synthetic'}, 'versions':{}}
    (root/'experiment.json').write_text(json.dumps(record))
    return {'contract_sources':['contract.md'], 'evaluation_contract':{'timezone':'America/New_York'},
            'records':[{'path':'experiment.json', 'models':['Lag 24']} ]}


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = fixture(self.root)

    def test_split_metrics_exact_and_baseline_has_no_checkpoint(self):
        plan = build_plans(self.root, self.config)[0]
        self.assertEqual(plan['metrics']['validation_MAE'], 3.)
        self.assertEqual(plan['metrics']['test_MAE'], 7.)
        self.assertEqual(plan['params']['training'], 'no fitting')
        self.assertNotIn('checkpoint_reference', plan['params'])
        self.assertEqual(plan['tags']['training_code_commit'], 'not recorded')

    def test_duplicate_source_model_rejected(self):
        self.config['records'] *= 2
        with self.assertRaisesRegex(ValueError, 'twice'):
            build_plans(self.root, self.config)

    def test_missing_split_rejected(self):
        p=self.root/'experiment.json';record=json.loads(p.read_text());record['metrics'].pop()
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            build_plans(self.root, self.config)

    def test_nonfinite_metric_rejected(self):
        p=self.root/'experiment.json';record=json.loads(p.read_text());record['metrics'][0]['MAE']=float('nan')
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'finite'):
            build_plans(self.root, self.config)

    def test_outside_project_path_rejected(self):
        self.config['records'][0]['path']='../outside.json'
        with self.assertRaisesRegex(ValueError, 'within'):
            build_plans(self.root, self.config)


@unittest.skipUnless(importlib.util.find_spec('mlflow'), 'Use .venv-mlflow for real-store tests')
class StoreTests(PlanTests):
    def test_reimport_preserves_run_and_single_metric_history(self):
        from mlflow.tracking import MlflowClient
        plans = build_plans(self.root, self.config)
        store=self.root/'tracking'
        first=import_plans(store, 'synthetic', plans, 'import-only-commit')
        second=import_plans(store, 'synthetic', plans, 'different-import-commit')
        self.assertTrue(first[0]['created']);self.assertFalse(second[0]['created'])
        self.assertEqual(first[0]['run_id'],second[0]['run_id'])
        client=MlflowClient(tracking_uri=f'sqlite:///{store / "tracking.sqlite"}')
        saved=client.get_run(first[0]['run_id'])
        self.assertEqual(saved.data.params,plans[0]['params'])
        self.assertEqual(saved.data.metrics,plans[0]['metrics'])
        self.assertEqual(saved.data.tags['import_code_commit'],'import-only-commit')
        for name in plans[0]['metrics']:
            self.assertEqual(len(client.get_metric_history(saved.info.run_id,name)),1)
        self.assertEqual(client.list_artifacts(saved.info.run_id),[])

    def test_incomplete_run_resumes_without_duplicate(self):
        from mlflow.tracking import MlflowClient
        store=self.root/'tracking';store.mkdir()
        client=MlflowClient(tracking_uri=f'sqlite:///{store / "tracking.sqlite"}')
        experiment=client.create_experiment('synthetic',artifact_location=(store/'artifacts').as_uri())
        plan=build_plans(self.root,self.config)[0]
        run=client.create_run(experiment,tags={**plan['tags'],'citypulse_import_key':plan['key'],
            'citypulse_plan_sha256':plan['plan_sha'],'import_timestamp_ms':'1','import_state':'incomplete'})
        client.log_param(run.info.run_id,'timezone','America/New_York')
        client.log_metric(run.info.run_id,'test_MAE',7.,timestamp=1,step=0)
        result=import_plans(store,'synthetic',[plan])
        self.assertEqual(result[0]['run_id'],run.info.run_id)
        self.assertFalse(result[0]['created'])
        self.assertEqual(client.get_run(run.info.run_id).info.status,'FINISHED')
        self.assertEqual(len(client.get_metric_history(run.info.run_id,'test_MAE')),1)

    def test_changed_immutable_metadata_rejected(self):
        plan=build_plans(self.root,self.config)[0];store=self.root/'tracking'
        import_plans(store,'synthetic',[plan])
        changed=deepcopy(plan);changed['plan_sha']='changed'
        with self.assertRaisesRegex(ValueError,'immutable'):
            import_plans(store,'synthetic',[changed])
