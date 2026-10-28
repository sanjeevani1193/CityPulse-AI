"""Lightweight runner-monitor and public-summary checks, no model loading."""
import json
from pathlib import Path
from types import SimpleNamespace
from citypulse_ai.evaluation.serving_verification import Verification


def test_guard_stops_before_services_on_swap(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from citypulse_ai.evaluation import serving_verification as m
    memory={'MemTotal':16*1024**3,'MemAvailable':12*1024**3,'SwapTotal':1024,'SwapFree':0}
    monkeypatch.setattr(m,'host_memory',lambda:memory)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=30*1024**3))
    v=Verification('unused','native');v.monitor()
    assert v.failure and v.children==[]
    assert v.summary['peak_swap_used_bytes']==1024
    assert 'history' not in json.dumps(v.summary)


def test_manual_workflow_exact_trust_and_private_payload_exclusion():
    import yaml
    w=yaml.safe_load(Path('.github/workflows/forecasting-verification.yml').read_text())
    assert set(w['on'])=={'workflow_dispatch'}
    assert w['on']['workflow_dispatch']['inputs']['scope']['default']=='full'
    job=w['jobs']['verify']
    assert job['runs-on']=='ubuntu-24.04-arm' and job['permissions']=={'contents':'read','id-token':'write'}
    upload=[s for s in job['steps'] if s.get('uses','').startswith('actions/upload-artifact@')][0]
    assert upload['with']['path']=='verification-reports/'
    auth=[s for s in job['steps'] if s.get('uses','').startswith('google-github-actions/auth@')][0]
    assert 'service_account' not in auth['with'] and len(auth['uses'].split('@')[1])==40


def test_restart_waits_for_application_readiness(monkeypatch):
    import httpx
    from citypulse_ai.evaluation import serving_verification as m
    monkeypatch.setattr(m,'host_memory',lambda:{'MemAvailable':12*1024**3})
    v=Verification('unused','docker-recovery');calls=[]
    monkeypatch.setattr(v,'run',lambda command,name:calls.append(command))
    replies=iter([False,True])
    transport=httpx.MockTransport(lambda request:httpx.Response(200,json={'models':[
        {'id':'tensorflow-lstm','ready':next(replies)}]}))
    real=httpx.Client
    monkeypatch.setattr(m.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(httpx,'Client',lambda **kw:real(transport=transport,**kw))
    v.restart_worker()
    assert calls[0][-2:]==['start','lstm-worker'] and '--wait' not in calls[0]
