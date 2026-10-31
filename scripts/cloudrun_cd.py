"""Bounded CD utilities. Network operations require explicit CLI subcommands.

No credentials, source archives, model bytes or request histories are reported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

PROJECT = 'citypulse-ai-511315'
REGION = 'europe-west3'
REPO = 'sanjeevani1193/CityPulse-AI'
IMAGE = f'{REGION}-docker.pkg.dev/{PROJECT}/citypulse-forecasting/xgboost'
SERVICE = f'projects/{PROJECT}/locations/{REGION}/services/citypulse-xgboost'
SA = f'citypulse-xgb-deployer@{PROJECT}.iam.gserviceaccount.com'
ROOT = Path(__file__).resolve().parents[1]


def request(url, token, *, data=None, method=None):
    headers = {'Authorization': 'Bearer '+token, 'User-Agent': 'CityPulse-CD'}
    if data is not None:
        data = json.dumps(data).encode(); headers['Content-Type'] = 'application/json'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers, data=data, method=method), timeout=60) as r:
        return json.load(r)


def github(path):
    return request('https://api.github.com/repos/'+REPO+'/'+path, os.environ['GH_TOKEN'])


def require_head(sha):
    if not re.fullmatch('[0-9a-f]{40}', sha) or github('git/ref/heads/main')['object']['sha'] != sha:
        raise ValueError('Only the current trusted main commit can deploy')


def ci_result(runs, sha):
    required = {'Inference CI', 'SQLPilot deterministic core', 'ExperimentLab statistics'}
    latest = {}
    for run in runs:
        if run['name'] in required and run['head_sha'] == sha and run['event'] == 'push' and run['head_branch'] == 'main':
            if run['name'] not in latest or run['id'] > latest[run['name']]['id']:
                latest[run['name']] = run
    for run in latest.values():
        if run['status'] == 'completed' and run['conclusion'] != 'success':
            raise ValueError('Required CI failed: '+run['name'])
    return set(latest) == required and all(r['status'] == 'completed' for r in latest.values())


def gate(sha):
    # No cloud credentials until the exact commit has passed all three CI suites.
    for _ in range(60):
        require_head(sha)
        if ci_result(github('actions/runs?per_page=100&head_sha='+sha)['workflow_runs'], sha):
            return
        time.sleep(15)
    raise TimeoutError('Required CI did not finish successfully within 15 minutes')


def check_bytes(data, record):
    if len(data) != record['size_bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Frozen input size/checksum mismatch')


def restore(destination, client=None):
    inputs = json.loads((ROOT/'infra/cloudrun-cd-inputs.json').read_text())['inputs']
    # Missing planned demo generations must fail before any download or build.
    if any(not str(r['generation']).isdigit() or int(r['generation']) <= 0 for r in inputs.values()):
        raise ValueError('Approved demo upload and generation pinning are still required')
    if client is None:
        from google.cloud import storage
        client = storage.Client(project=PROJECT)
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Restore destination must be new')
    destination.mkdir(parents=True)
    for name, record in inputs.items():
        bucket, key = record['cloud_path'][5:].split('/', 1)
        blob = client.bucket(bucket).blob(key, generation=int(record['generation']))
        blob.reload(retry=None, timeout=60)
        if int(blob.generation) != int(record['generation']) or blob.size != record['size_bytes']:
            raise ValueError('Pinned GCS generation/size mismatch')
        data = blob.download_as_bytes(retry=None, timeout=60, if_generation_match=int(record['generation']))
        check_bytes(data, record)
        path = destination/name; path.parent.mkdir(exist_ok=True, parents=True); path.write_bytes(data)
    # Existing create-only packager verifies vocabulary and artifact hashes again.
    from citypulse_ai.api.cloudrun_bundle import prepare
    prepare(destination, destination.parent/'context', ROOT)


def cloud_token():
    import google.auth
    from google.auth.transport.requests import Request
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    credentials.refresh(Request())
    return credentials.token


def updated_containers(service, image):
    if not re.fullmatch(re.escape(IMAGE)+r'@sha256:[a-f0-9]{64}', image):
        raise ValueError('Only a digest from the approved image is deployable')
    containers = json.loads(json.dumps(service['template']['containers']))
    if len(containers) != 1: raise ValueError('Unexpected multi-container service')
    containers[0]['image'] = image
    return {'name': SERVICE, 'etag': service['etag'], 'template': {'containers': containers}}


def build_deploy(sha, storage_client=None):
    require_head(sha)
    tag = f'git-{sha}-run-{os.environ["GITHUB_RUN_ID"]}-{os.environ["GITHUB_RUN_ATTEMPT"]}'
    # Upload a new owned source object with a create-only generation precondition.
    # The client does not need bucket-list/get or historical-object write access.
    import tarfile
    import yaml
    folder = ROOT/'outputs/cloudrun-cd'
    archive = folder/'source.tgz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(folder/'context', arcname='.')
    key = f'cloudrun-build-source/cd/{tag}.tgz'
    if storage_client is None:
        from google.cloud import storage
        storage_client = storage.Client(project=PROJECT)
    blob = storage_client.bucket(PROJECT+'-data').blob(key)
    blob.upload_from_filename(str(archive), if_generation_match=0, retry=None, timeout=60)
    config = yaml.safe_load((ROOT/'infra/cloudbuild-xgboost.yaml').read_text())
    config['source'] = {'storageSource': {'bucket': PROJECT+'-data', 'object': key, 'generation': blob.generation}}
    config['substitutions'] = {'_IMAGE': IMAGE+':'+tag}
    base = f'https://cloudbuild.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/builds'
    # Exactly one create call, no automatic retry or resubmit.
    submitted = request(base, cloud_token(), data=config, method='POST')
    build_id = submitted['metadata']['build']['id']
    print('Cloud Build submitted once: '+build_id, flush=True)
    for _ in range(72):
        build = request(base+'/'+build_id, cloud_token())
        if build['status'] == 'SUCCESS': break
        if build['status'] not in ('QUEUED', 'WORKING', 'PENDING'):
            raise RuntimeError('Cloud Build failed; no automatic retry: '+build['status'])
        time.sleep(15)
    else: raise TimeoutError('Build monitoring timed out; inspect job, do not resubmit')
    image = IMAGE+'@'+build['results']['images'][0]['digest']
    # Recheck main to prevent stale builds deploying after a newer push.
    require_head(sha)
    url = 'https://run.googleapis.com/v2/'+SERVICE
    service = request(url, cloud_token())
    body = updated_containers(service, image)
    # Repeated-field mask requires retaining every existing container setting.
    operation = request(url+'?updateMask=template.containers', cloud_token(), data=body, method='PATCH')
    if operation.get('error'):
        raise RuntimeError('Cloud Run rejected image update; no automatic retry')
    # Service GET polling avoids a project-wide operation-reader grant.
    for _ in range(40):
        actual = request(url, cloud_token())
        observed = str(actual.get('observedGeneration')) == str(actual.get('generation'))
        if observed and not actual.get('reconciling'):
            if actual.get('terminalCondition', {}).get('state') == 'CONDITION_FAILED':
                raise RuntimeError('Cloud Run rollout failed; no automatic retry')
            if (actual.get('terminalCondition', {}).get('state') == 'CONDITION_SUCCEEDED'
                    and actual.get('latestReadyRevision') == actual.get('latestCreatedRevision')
                    and actual['template']['containers'][0]['image'] == image):
                break
        time.sleep(10)
    else: raise TimeoutError('Cloud Run readiness timed out; inspect rollout, do not resubmit')
    receipt = {'source_sha': sha, 'build_id': build_id, 'image': image,
               'service_url': actual['uri'], 'revision': actual['latestReadyRevision']}
    (ROOT/'outputs/cloudrun-cd/deployment.json').write_text(json.dumps(receipt, indent=2)+'\n')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out: out.write('url='+actual['uri']+'\n')


def verify(url, root):
    # ID token comes from short-lived WIF service-account impersonation, never a key.
    if url != 'https://citypulse-xgboost-lw3wuexzqq-ey.a.run.app':
        raise ValueError('Only the existing authenticated service can be verified')
    token = os.environ['SERVICE_ID_TOKEN']
    root = Path(root)
    payload = json.loads((root/'demo-request.json').read_text())
    refs = json.loads((root/'demo-references.json').read_text())
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != refs['request_sha256']:
        raise ValueError('Validation demo identity mismatch')
    start = time.monotonic()
    health = request(url+'/health', token)
    assert health['model_ready'] is True
    stations = request(url+'/stations', token)
    # Public response uses station_ids; compare exact trusted vocabulary.
    prep = json.loads((root/'preprocessing.json').read_text())
    assert len(stations['station_ids']) == 100
    assert set(stations['station_ids']) == set(prep['station_codes'])
    models = request(url+'/models', token)
    assert [m['id'] for m in models['models']] == ['xgboost'] and models['models'][0]['ready']
    response = request(url+'/predict', token, data=payload)
    assert response['station_id'] == payload['station_id'] and response['model'] == 'xgboost'
    expected_version = 'model.ubj:'+json.loads((ROOT/'infra/cloudrun-cd-inputs.json').read_text())['inputs']['xgboost/model.ubj']['sha256'][:12]
    assert response['model_version'] == expected_version
    assert response['target_policy'] == 'accepted_recorded_starts_v1'
    assert response['timezone'] == 'America/New_York'
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    timestamps = [datetime.fromisoformat(t) for t in response['target_timestamps_utc']]
    assert len(timestamps) == 24 and timestamps[0] == datetime.fromisoformat(payload['origin'])
    assert all(b-a == timedelta(hours=1) for a,b in zip(timestamps, timestamps[1:]))
    assert all(t.utcoffset() == timedelta(0) for t in timestamps)
    assert response['target_timestamps_local'] == [t.astimezone(ZoneInfo('America/New_York')).isoformat() for t in timestamps]
    values = response['predicted_ride_counts']; expected = refs['expected']['xgboost']
    assert len(values) == len(expected) == 24
    assert all(v >= 0 and abs(v-e) <= 1e-4+1e-5*abs(e) for v,e in zip(values,expected))
    assert request(url+'/predict', token, data=payload) == response
    bad = json.loads(json.dumps(payload)); bad['history'][-1]['timestamp'] = payload['origin']
    try: request(url+'/predict', token, data=bad)
    except urllib.error.HTTPError as error: assert error.code == 422
    else: raise AssertionError('Future observations were accepted')
    try:
        with urllib.request.urlopen(url+'/health', timeout=30): pass
    except urllib.error.HTTPError as error: assert error.code in (401,403)
    else: raise AssertionError('Anonymous invocation unexpectedly allowed')
    receipt = {'health': 'ready', 'stations': 100, 'predictions': 24, 'future_status':422,
               'parity_max_abs_error': max(abs(v-e) for v,e in zip(values, expected)),
               'deterministic': True, 'elapsed_seconds': time.monotonic()-start}
    (root.parent/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['gate','restore','deploy','verify'])
    parser.add_argument('--sha'); parser.add_argument('--url')
    args = parser.parse_args()
    if args.command == 'gate': gate(args.sha)
    elif args.command == 'restore': restore(ROOT/'outputs/cloudrun-cd/inputs')
    elif args.command == 'deploy': build_deploy(args.sha)
    else: verify(args.url, ROOT/'outputs/cloudrun-cd/inputs')
