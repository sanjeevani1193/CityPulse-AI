"""CD policy and failure tests: no cloud calls, build, inference or credentials."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

spec = importlib.util.spec_from_file_location('cloudrun_cd', Path('scripts/cloudrun_cd.py'))
cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)
SHA = 'a'*40


def runs():
    return [dict(id=i, name=name, head_sha=SHA, head_branch='main', event='push',
                 status='completed', conclusion='success') for i,name in enumerate(
        ['Inference CI','SQLPilot deterministic core','ExperimentLab statistics'],1)]


def test_gate_requires_all_exact_successful_ci():
    assert cd.ci_result(runs(), SHA)
    assert not cd.ci_result(runs()[:-1], SHA)
    assert not cd.ci_result(runs(), 'b'*40)
    for field,value in [('head_branch','other'),('event','pull_request'),('status','in_progress')]:
        data=runs(); data[0][field]=value
        assert not cd.ci_result(data,SHA)


@pytest.mark.parametrize('conclusion',['failure','cancelled','skipped','timed_out'])
def test_ci_never_silently_skips_failed_runs(conclusion):
    data=runs(); data[0]['conclusion']=conclusion
    with pytest.raises(ValueError,match='Required CI failed'): cd.ci_result(data,SHA)


def test_latest_ci_attempt_must_pass():
    data=runs(); newer={**data[0], 'id':100, 'conclusion':'failure'}
    with pytest.raises(ValueError): cd.ci_result(data+[newer],SHA)


@pytest.mark.parametrize('sha',[None,'main','a'*39,'b'*40])
def test_stale_or_untrusted_source_rejected(monkeypatch,sha):
    monkeypatch.setattr(cd,'github',lambda _: {'object':{'sha':SHA}})
    with pytest.raises((ValueError,TypeError)): cd.require_head(sha)


def test_service_update_changes_only_image_and_preserves_container_settings():
    service={'etag':'current-etag', 'template':{'containers':[{'image':'old',
        'resources':{'limits':{'cpu':'0.5','memory':'512Mi'}}, 'ports':[{'containerPort':8080}],
        'startupProbe':{'httpGet':{'path':'/health'}}}]}}
    original=copy.deepcopy(service)
    image=cd.IMAGE+'@sha256:'+'b'*64
    body=cd.updated_containers(service,image)
    assert service==original and body['etag']=='current-etag'
    assert set(body)=={'name','etag','template'}
    body['template']['containers'][0]['image']='old'
    assert body['template']==original['template']
    with pytest.raises(ValueError): cd.updated_containers(service,cd.IMAGE+':latest')
    with pytest.raises(ValueError): cd.updated_containers(service,'other@sha256:'+'b'*64)


def test_missing_generation_stops_before_gcs_access(tmp_path,monkeypatch):
    root=tmp_path/'repo'; (root/'infra').mkdir(parents=True)
    (root/'infra/cloudrun-cd-inputs.json').write_text(json.dumps({'inputs':{'demo':{'generation':None}}}))
    monkeypatch.setattr(cd,'ROOT',root)
    client=Mock()
    # Proposed demo is deliberately unprovisioned until explicit upload approval.
    with pytest.raises(ValueError, match='generation pinning'): cd.restore(tmp_path/'new',client)
    client.bucket.assert_not_called()
    assert not (tmp_path/'new').exists()


def test_checksum_mismatch_rejected():
    data=b'frozen'
    record={'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    cd.check_bytes(data,record)
    with pytest.raises(ValueError): cd.check_bytes(b'changed',record)


def test_restoration_generation_and_checksum_fail_closed(tmp_path,monkeypatch):
    root=tmp_path/'repo'; (root/'infra').mkdir(parents=True)
    data=b'frozen'; record={'generation':'123','size_bytes':len(data),
        'sha256':hashlib.sha256(data).hexdigest(),'cloud_path':'gs://private/key'}
    (root/'infra/cloudrun-cd-inputs.json').write_text(json.dumps({'inputs':{'model':record}}))
    monkeypatch.setattr(cd,'ROOT',root)
    blob=Mock(); blob.generation=123; blob.size=len(data); blob.download_as_bytes.return_value=b'broken'
    client=Mock(); client.bucket.return_value.blob.return_value=blob
    with pytest.raises(ValueError,match='checksum'):cd.restore(tmp_path/'out',client)
    client.bucket.return_value.blob.assert_called_once_with('key',generation=123)
    blob.download_as_bytes.assert_called_once_with(retry=None,timeout=60,if_generation_match=123)
    blob.generation=124
    with pytest.raises(ValueError,match='generation'):cd.restore(tmp_path/'second',client)


def test_workflow_trust_disabled_automation_and_sanitized_outputs():
    workflow=yaml.safe_load(Path('.github/workflows/cloudrun-cd.yml').read_text())
    assert workflow['on']['push']['branches']==['main']
    assert workflow['concurrency']['cancel-in-progress'] is False
    assert workflow['jobs']['ci-gate']['if'] == "github.ref == 'refs/heads/main' && github.event_name == 'workflow_dispatch'"
    assert workflow['jobs']['deploy']['needs']=='ci-gate'
    assert workflow['jobs']['deploy']['permissions']=={'contents':'read','id-token':'write'}
    assert 'id-token' not in workflow['jobs']['ci-gate'].get('permissions',{})
    for step in workflow['jobs']['deploy']['steps']:
        if 'uses' in step:
            import re
            assert re.search(r'@[a-f0-9]{40}$',step['uses'])
        assert not step.get('continue-on-error')
    published=workflow['jobs']['deploy']['steps'][-1]['with']['path'].splitlines()
    assert published==['outputs/cloudrun-cd/deployment.json','outputs/cloudrun-cd/verification.json']


def test_terraform_ignores_only_image_and_preserves_existing_provider():
    text=Path('infra/terraform/main.tf').read_text()
    assert 'ignore_changes = [template[0].containers[0].image]' in text
    policy=Path('infra/terraform/cd.tf').read_text()
    for claim in ['repository_id','repository_owner_id','workflow_ref','refs/heads/main']:
        assert claim in policy
    assert 'citypulse-github-verification' not in policy
    assert 'roles/owner' not in policy and 'roles/editor' not in policy
    assert 'storage.objects.delete' not in policy and 'run.services.create' not in policy
    assert 'cloudrun-build-source/cd/' in policy
    assert 'roles/iam.workloadIdentityUser' in policy


def test_failed_build_is_submitted_once_and_never_deploys(tmp_path,monkeypatch):
    root=tmp_path/'repo'; (root/'outputs/cloudrun-cd/context').mkdir(parents=True)
    (root/'infra').mkdir()
    (root/'infra/cloudbuild-xgboost.yaml').write_text(Path('infra/cloudbuild-xgboost.yaml').read_text())
    monkeypatch.setattr(cd,'ROOT',root)
    monkeypatch.setattr(cd,'require_head',lambda _: None)
    monkeypatch.setattr(cd,'cloud_token',lambda: 'synthetic-token')
    monkeypatch.setenv('GITHUB_RUN_ID','123'); monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')
    calls=[]
    def api(url,token,**kwargs):
        calls.append((url,kwargs))
        if kwargs.get('method')=='POST': return {'metadata':{'build':{'id':'one-build'}}}
        return {'status':'FAILURE'}
    monkeypatch.setattr(cd,'request',api)
    client=Mock(); client.bucket.return_value.blob.return_value.generation=123
    with pytest.raises(RuntimeError,match='no automatic retry'):cd.build_deploy(SHA,client)
    assert len(calls)==2
    assert [k.get('method') for _,k in calls].count('POST')==1
    assert not any(k.get('method')=='PATCH' for _,k in calls)
    client.bucket.return_value.blob.return_value.upload_from_filename.assert_called_once()
    assert client.bucket.return_value.blob.return_value.upload_from_filename.call_args.kwargs['if_generation_match']==0
    source=calls[0][1]['data']['source']['storageSource']
    assert source['generation']==123 and source['object'].startswith('cloudrun-build-source/cd/')


def test_untrusted_verification_url_cannot_receive_token(tmp_path,monkeypatch):
    monkeypatch.delenv('SERVICE_ID_TOKEN',raising=False)
    with pytest.raises(ValueError,match='existing authenticated'):cd.verify('https://attacker.invalid',tmp_path)


def test_http_verification_contract_with_controlled_runtime_only(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from citypulse_ai.api.cloudrun_xgboost import create_app
    from tests.ci.test_station_contracts import ContractRuntime,payload
    import urllib.error
    data=payload()
    root=tmp_path/'inputs';root.mkdir()
    (root/'demo-request.json').write_text(json.dumps(data))
    with TestClient(create_app(ContractRuntime)) as client:
        result=client.post('/predict',json=data).json()
        result['model_version']='model.ubj:f7a64cc25e38'
        (root/'demo-references.json').write_text(json.dumps({'request_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),'expected':{'xgboost':result['predicted_ride_counts']}}))
        stations=client.get('/stations').json()['station_ids']
        # The controlled runtime has one station; emulate the 100-ID listing only.
        stations += [str(n) for n in range(99)]
        (root/'preprocessing.json').write_text(json.dumps({'station_codes':dict.fromkeys(stations,1)}))
        def api(url,token,**kwargs):
            endpoint=url.split('.app')[-1]
            if endpoint=='/stations': return {'station_ids':stations}
            if endpoint=='/predict':
                response=client.post(endpoint,json=kwargs['data'])
                if response.status_code!=200:
                    raise urllib.error.HTTPError(url,response.status_code,'invalid',{},None)
                return result
            return client.get(endpoint).json()
        monkeypatch.setattr(cd,'request',api)
        monkeypatch.setenv('SERVICE_ID_TOKEN','synthetic-token')
        def anonymous(*args,**kwargs): raise urllib.error.HTTPError(args[0],403,'denied',{},None)
        monkeypatch.setattr(cd.urllib.request,'urlopen',anonymous)
        cd.verify('https://citypulse-xgboost-lw3wuexzqq-ey.a.run.app',root)
    receipt=json.loads((tmp_path/'verification.json').read_text())
    assert receipt['parity_max_abs_error']==0 and receipt['future_status']==422


def test_wif_pool_display_name_fits_gcp_limit():
    import re
    source=Path('infra/terraform/cd.tf').read_text()
    block=source.split('resource "google_iam_workload_identity_pool" "cd" {',1)[1].split('\n}',1)[0]
    display=re.search(r'display_name\s*=\s*"([^"]+)"',block).group(1)
    assert 0 < len(display) <= 32
