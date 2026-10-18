"""One bounded, resumable source-month cycle using the existing ETL/cloud APIs.

Run in its own CLI process: native DuckDB/file limits are process scoped.
No bulk mode, persistent catalog creation, retention cleanup or automatic retry.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid

import duckdb

from citypulse_ai.etl.archives import discover_archive, inspect_archive, fingerprint, validate_month, write_json
from citypulse_ai.etl.historical import load_manifest, process_month
from citypulse_ai.etl.historical_cli import single_writer
from citypulse_ai.etl.storage_state import output_receipts, source_receipts, check_surviving_outputs, source_files
from citypulse_ai.etl.temporary_reconciliation import reconcile_temporary
from citypulse_ai.cloud.gcs import GCSStorage
from citypulse_ai.cloud.historical import publish_month
from citypulse_ai.cloud.retention import publish_sources, inspect_remote

GIB = 1024**3
START_GATE = 28 * GIB  # General policy is unchanged; no lower-gate CLI switch.
WORKSPACE = 10 * GIB
FILE_CAP = 2 * GIB
RESERVE = 15 * GIB
PROJECT, BUCKET = 'citypulse-ai-511315', 'citypulse-ai-511315-data'


def verify_publication(entry, kind, store):
    receipts = output_receipts(entry) if kind == 'cloud_publication' else source_receipts(entry)
    result = []
    for name, receipt in receipts.items():
        _, info = inspect_remote(store, entry, entry[kind], name, receipt)
        result.append({'name': name, 'kind': kind, 'cloud_path': receipt['cloud_path'],
            'generation': info['generation'], 'sha256': receipt['sha256'],
            'size_bytes': info['size_bytes'], 'checks': receipt['checks']})
    return result


def inventory(root):
    """Only required reconciliation inputs and persistent catalogs, not a broad audit."""
    manifest = load_manifest(root)
    inputs = []
    for month, entry in manifest['archives'].items():
        if entry.get('status') != 'processed':
            continue
        output_receipts(entry); source_receipts(entry)
        check_surviving_outputs(entry)
        for name in ('validated_trips.parquet', 'rejected_trips.csv'):
            p = Path(entry['output_dir'])/name
            if not p.is_file() or p.is_symlink() or fingerprint(p) != entry['output_sha256'][name]:
                raise ValueError(f'Required local input missing/changed; no automatic restoration: {p}')
            inputs.append({'path': str(p), 'sha256': entry['output_sha256'][name], 'size_bytes': p.stat().st_size})
    canonical = {str(p): fingerprint(p) for p in (root/'canonical').rglob('*') if p.is_file()}
    views = verify_views(root)
    return {'manifest': manifest, 'inputs': inputs, 'canonical_hashes': canonical, 'view_bindings': views}


def verify_views(root):
    bindings = []
    for p in (root/'canonical').rglob('catalog.duckdb'):
        with duckdb.connect(str(p), read_only=True, config={'memory_limit':'512MB'}) as con:
            for (name,) in con.execute('SELECT view_name FROM duckdb_views() WHERE NOT internal').fetchall():
                con.execute('SELECT * FROM "'+name.replace('"','""')+'" LIMIT 0').fetchall()
                bindings.append({'database':str(p), 'view':name})
    return bindings


def preflight(month, root, work, store, *, september_budget=False):
    if (work == root or work in root.parents or root/'canonical' == work
            or root/'canonical' in work.parents):
        raise ValueError('Monthly workspace must not overlap protected historical/catalog roots')
    parent=work
    while not parent.exists():parent=parent.parent
    if parent.stat().st_dev != root.stat().st_dev:
        raise ValueError('This bounded cycle requires raw/work/output on the monitored filesystem')
    free = shutil.disk_usage(root).free
    if september_budget and month != '2024-09':
        raise ValueError('September-specific budget cannot weaken another month preflight')
    starting_gate = 26*GIB if september_budget else START_GATE
    if free < starting_gate:
        raise ValueError(f'{starting_gate/GIB:g} GiB starting gate not met: {free/GIB:.3f} GiB free; no cleanup attempted')
    before = inventory(root)
    source = discover_archive(month)
    source.update(inspect_archive(source))
    # Same archive-derived 10x CSV envelope used by the historical planner.
    estimate = source['bytes'] + 9*source['csv_bytes']
    if estimate > WORKSPACE or source['bytes'] >= FILE_CAP:
        raise ValueError('Archive-derived storage envelope exceeds the bounded monthly budget')
    if free < estimate+16*GIB:
        raise ValueError('Archive envelope plus 16 GiB interruption margin does not fit')
    gate_evidence = None
    if september_budget:
        measured = root/'august-2024-checkpoint'/'results.json'
        august = json.loads(measured.read_text())
        if (august['status'] != 'august_ingested_cloud_verified_and_globally_reconciled'
                or august['duckdb_memory_limit'] != '512MB' or august['guard_triggered']):
            raise ValueError('Verified August resource evidence is unavailable')
        scaled = int(august['peak_new_local_bytes']*source['csv_bytes']/900041984)
        if scaled > WORKSPACE or starting_gate < WORKSPACE+16*GIB:
            raise ValueError('September workspace plus interruption margin is not bounded')
        gate_evidence = {'scope':'September 2024 only; general 28 GiB gate unchanged',
            'august_results_sha256':fingerprint(measured),'august_observed_peak_bytes':august['peak_new_local_bytes'],
            'scaled_observed_peak_bytes':scaled,'live_archive_envelope_bytes':estimate,
            'workspace_bound_bytes':WORKSPACE,'interruption_margin_bytes':16*GIB,
            'justification':'26 GiB = 10 GiB bounded workspace + 16 GiB free interruption margin; archive envelope fits workspace.'}
    store.bucket.reload(timeout=60, retry=None)
    if store.bucket.storage_class != 'STANDARD' or store.bucket.iam_configuration.public_access_prevention != 'enforced':
        raise ValueError('Unexpected bucket class or public-access settings')
    permissions = store.bucket.test_iam_permissions(['storage.objects.get','storage.objects.list','storage.objects.create'], timeout=60, retry=None)
    if set(permissions) != {'storage.objects.get','storage.objects.list','storage.objects.create'}:
        raise ValueError('Required existing bucket read/list/create credentials unavailable')
    existing = []
    for prefix in ('raw/historical','curated/historical'):
        existing.extend({'name': b.name, 'generation':str(b.generation), 'size_bytes':b.size}
            for b in store.client.list_blobs(store.bucket_name, prefix=f'{prefix}/source_archive_month={month}/', timeout=60, retry=None))
    if existing and month not in before['manifest']['archives']:
        raise ValueError('Unregistered cloud artifacts exist for month; investigate instead of downloading')
    if work.exists():
        marker = work/'monthly-cycle-owner.json'
        if not marker.is_file() or json.loads(marker.read_text()) != {'month':month,'output_root':str(root.resolve())}:
            raise ValueError('Existing workspace has no matching ownership marker')
    # Prior inputs need no restoration. Live-check their exact cloud identities.
    prior_cloud = []
    for entry in before['manifest']['archives'].values():
        if entry.get('status') != 'processed':
            continue
        for kind, receipts in [('cloud_publication',output_receipts(entry)),('source_backup',source_receipts(entry))]:
            for name, receipt in receipts.items():
                if name not in ('validated_trips.parquet','rejected_trips.csv','processing_manifest.json'):
                    continue
                _, info = inspect_remote(store, entry, entry[kind], name, receipt)
                prior_cloud.append({'cloud_path':receipt['cloud_path'],'generation':info['generation'],'sha256':receipt['sha256']})
    return {**before, 'month':month, 'source':source, 'free_before_bytes':free,
        'archive_envelope_bytes':estimate, 'starting_gate_bytes':starting_gate,'gate_exception_evidence':gate_evidence,
        'workspace_limit_bytes':WORKSPACE, 'file_cap_bytes':FILE_CAP, 'reserve_bytes':RESERVE,
        'engine_spill_cap_bytes':2*GIB, 'reconciliation_scratch_budget_bytes':4*GIB,
        'duckdb_memory_limit':'512MB','existing_month_cloud_objects':existing,
        'prior_required_cloud_objects':prior_cloud,
        'bucket':{'name':store.bucket_name,'location':store.bucket.location,'storage_class':store.bucket.storage_class},
        'cost_budget':{'estimated_new_bytes_from_August':int(source['csv_bytes']/900041984*1244919562),
            'publication_volume_cap_bytes':3*GIB,'verification_transfer_budget_usd':.36,
            'operations_budget_usd':.01,'monthly_storage_budget_usd':.15,
            'storage_assumption_usd_per_gib_month':.05,'transfer_usd_per_gib':.12,
            'pricing_source':'https://cloud.google.com/storage/pricing',
            'note':'Conservative budget, not billing quotation; Standard ingress free, SHA verification downloads charged.'}}


def cleanup_verification_scratch(publication, owned_tmp, store, entry, kind):
    """Only this attempt's verified disposable download copies; retain failures."""
    verify_publication(entry, kind, store)
    removed = []
    for receipt in publication['files'].values():
        p = Path(receipt['download_path'])
        if not p.resolve().is_relative_to(owned_tmp.resolve()) or p.is_symlink() or p.parent.is_symlink():
            continue  # A prior attempt's retained files are never deleted.
        if (not p.is_file() or receipt['verification_status'] != 'verified'
                or receipt['checks'].get('sha256_match') is not True
                or p.stat().st_size != receipt['size_bytes'] or fingerprint(p) != receipt['sha256']):
            raise ValueError(f'Verification scratch cannot be certified disposable: {p}')
        removed.append({'path':str(p),'size_bytes':p.stat().st_size,
            'cloud_path':receipt['cloud_path'],'generation':receipt['generation'],'sha256':receipt['sha256']})
        p.unlink()  # Exact single file, no recursive directory or source deletion.
    return removed


def source_cleanup_candidates(entry, raw_root, store):
    """Exact verified source copies for a later approval; never delete them."""
    verify_publication(entry,'source_backup',store)
    receipts=source_receipts(entry)
    originals=source_files(entry,raw_root)
    candidates=[]
    for name,item in originals.items():
        path=Path(item['path']);receipt=receipts[name]
        if not path.exists():continue
        if path.is_symlink() or not path.is_file() or fingerprint(path)!=item['sha256']:
            raise ValueError('Source cleanup candidate hash or path differs')
        candidates.append({'path':str(path),'size_bytes':path.stat().st_size,'sha256':item['sha256'],
            'kind':item['role'],'cloud_path':receipt['cloud_path'],'generation':receipt['generation'],
            'cloud_sha256':receipt['sha256'],'restore':'Generation-pinned download and SHA256 verification.',
            'approval_required':True,'local_reconciliation_dependency':False})
    # The official ZIP is the original source backup, not a second CSV upload.
    if entry['source'].get('sha256'):
        zip_receipt=receipts[entry['source']['key']]
        members={m['name']:m for m in entry['source']['members']}
        for item in entry['csv_files']:
            path=Path(item['path'])
            if not path.exists():continue
            if (path.is_symlink() or not path.is_file() or item['member'] not in members
                    or fingerprint(path)!=item['sha256']):
                raise ValueError('Extracted CSV cleanup candidate differs from verified source inventory')
            candidates.append({'path':str(path),'size_bytes':path.stat().st_size,'sha256':item['sha256'],
                'kind':'extracted_csv','cloud_path':zip_receipt['cloud_path'],'generation':zip_receipt['generation'],
                'cloud_sha256':zip_receipt['sha256'],'archive_member':item['member'],
                'processing_manifest':receipts['processing_manifest.json'],
                'restore':'Restore pinned ZIP, extract this exact member, verify CRC and recorded CSV SHA256.',
                'approval_required':True,'local_reconciliation_dependency':False})
    return {'source_archive_month':entry['month'],'files':candidates,
        'logical_bytes':sum(c['size_bytes'] for c in candidates),
        'not_deleted':True,'future_checks_required':'Recheck generations, hashes, active processes and catalog dependencies before approved deletion.'}


class BoundedRuntime:
    """August's process limits and watchdog, generalized to one explicit month."""
    def __init__(self, month, root, work, report):
        self.month,self.root,self.work,self.report = month,root,work,report
        self.baseline = {p.resolve() for p in (root/'reconciliation-audits').rglob('*') if p.is_file()}
        self.done = threading.Event(); self.connections=[]; self.forbid_sql=False
        self.metrics={'stage':'starting','minimum_free_bytes':shutil.disk_usage(root).free,
            'peak_workspace_bytes':0,'peak_rss_bytes':0,'guard_triggered':False,'samples':[]}

    def workspace_bytes(self):
        paths = list(self.work.rglob('*')) + list((self.root/f'source_month={self.month}').rglob('*'))
        paths += list(self.report.rglob('*'))
        for folder in (self.root/'.work').glob(f'{self.month}-*'):
            paths.extend(folder.rglob('*'))
        paths.extend(p for p in (self.root/'reconciliation-audits').rglob('*') if p.resolve() not in self.baseline)
        total=0
        for p in paths:
            try:
                if p.is_file():total+=p.stat().st_size
            except FileNotFoundError:pass
        return total

    def stage(self, name):
        self.metrics['stage']=name
        print('STAGE',name,flush=True)
        write_json(self.report/'progress.json',self.metrics)

    def sample(self):
        swap=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True,timeout=5).strip()
        rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True,timeout=5).strip())*1024
        return {'elapsed_seconds':time.monotonic()-self.started,'stage':self.metrics['stage'],
            'swap':swap,'swap_allocated_bytes':float(re.search(r'total = ([0-9.]+)M',swap).group(1))*1024**2,
            'rss_bytes':rss,'free_bytes':shutil.disk_usage(self.root).free,'workspace_bytes':self.workspace_bytes()}

    def check(self, sample):
        if (sample['free_bytes'] < 16*GIB or sample['workspace_bytes'] > int(9.5*GIB)
                or sample['rss_bytes'] > 2*GIB or sample['swap_allocated_bytes']-self.initial_swap > 2*GIB):
            raise RuntimeError('Free disk/workspace/RSS/swap safety threshold exceeded; no automatic retry')

    def monitor(self):
        last=0
        while not self.done.wait(.25):
            try:
                free,size=shutil.disk_usage(self.root).free,self.workspace_bytes()
                self.metrics['minimum_free_bytes']=min(self.metrics['minimum_free_bytes'],free)
                self.metrics['peak_workspace_bytes']=max(self.metrics['peak_workspace_bytes'],size)
                if free < int(15.5*GIB) or size > WORKSPACE:
                    self.metrics.update(guard_triggered=True,error='Emergency disk threshold crossed')
                    write_json(self.report/'safety-stop.json',self.metrics);os._exit(75)
                if free < 16*GIB or size > int(9.5*GIB):
                    raise RuntimeError('Disk/workspace soft stop crossed')
                if time.monotonic()-last >= 10:
                    s=self.sample(); self.metrics['samples'].append(s)
                    self.metrics['peak_rss_bytes']=max(self.metrics['peak_rss_bytes'],s['rss_bytes'])
                    self.check(s);write_json(self.report/'resources-progress.json',self.metrics);last=time.monotonic()
            except BaseException as error:
                self.metrics.update(guard_triggered=True,error=str(error))
                write_json(self.report/'safety-stop.json',self.metrics)
                for con in self.connections:
                    try:con.interrupt()
                    except Exception:pass
                os.kill(os.getpid(),signal.SIGINT);return

    def __enter__(self):
        hard=resource.getrlimit(resource.RLIMIT_FSIZE)[1]
        if hard != resource.RLIM_INFINITY and hard < FILE_CAP:
            raise ValueError('Existing OS file hard limit is below the required staging cap')
        self.started=time.monotonic(); first=self.sample();self.initial_swap=first['swap_allocated_bytes'];self.check(first)
        self.metrics['samples'].append(first)
        self.original_connect=duckdb.connect
        def connect(*args,**kwargs):
            if self.forbid_sql:raise AssertionError('Cached reconciliation unexpectedly opened SQL')
            config=dict(kwargs.get('config') or {})
            if config.get('memory_limit','512MB') != '512MB':raise ValueError('Memory increase forbidden')
            config.update(memory_limit='512MB');config.setdefault('max_temp_directory_size','2GiB')
            kwargs['config']=config
            con=self.original_connect(*args,**kwargs);self.connections.append(con);return con
        duckdb.connect=connect
        self.old_limit=resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(resource.RLIMIT_FSIZE,(FILE_CAP,self.old_limit[1]))
        self.old_signal=signal.signal(signal.SIGXFSZ,lambda *args: (_ for _ in ()).throw(RuntimeError('2 GiB file cap exceeded')))
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start();return self

    def __exit__(self,kind,error,traceback):
        self.done.set();self.thread.join()
        duckdb.connect=self.original_connect
        resource.setrlimit(resource.RLIMIT_FSIZE,self.old_limit);signal.signal(signal.SIGXFSZ,self.old_signal)
        self.metrics.update(elapsed_seconds=time.monotonic()-self.started,
            free_after_bytes=shutil.disk_usage(self.root).free,
            peak_rss_bytes=max(self.metrics['peak_rss_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            retained_workspace_bytes=self.workspace_bytes())
        if error is not None:self.metrics['error']=repr(error)
        write_json(self.report/'resources.json',self.metrics)


def run_cycle(month, root, work, report, store, *, september_budget=False):
    plan=preflight(month,root,work,store,**({'september_budget':True} if september_budget else {}))
    write_json(report/'preflight.json',plan)
    print('PREFLIGHT',json.dumps({k:plan[k] for k in ('month','free_before_bytes','archive_envelope_bytes','cost_budget')}),flush=True)
    work.mkdir(parents=True,exist_ok=True)
    marker=work/'monthly-cycle-owner.json'
    if not marker.exists():
        with marker.open('x') as f:json.dump({'month':month,'output_root':str(root.resolve())},f)
    tmp=Path(tempfile.mkdtemp(prefix='verification-attempt-',dir=work)).resolve()
    old_temp=tempfile.tempdir;tempfile.tempdir=str(tmp)
    removed=[]
    try:
        with BoundedRuntime(month,root,work,report) as guard:
            guard.stage('ingestion')
            # Check the preflight publisher version again before any download.
            from citypulse_ai.etl import historical
            original_discover=historical.discover_archive
            def pinned_discover(requested):
                source=original_discover(requested)
                if requested != month or any(source[k]!=plan['source'][k] for k in source):
                    raise ValueError('Official source version changed since preflight')
                return source
            historical.discover_archive=pinned_discover
            try:
                ingestion=process_month(month,work/'raw',root,chunk_size=50000,memory_limit='512MB',
                    reserve_bytes=RESERVE,cloud_store=store)
            finally:historical.discover_archive=original_discover
            write_json(report/'ingestion.json',ingestion)
            entry=load_manifest(root)['archives'][month]
            # Bound actual publication volume before initiating any upload.
            sizes=sum((Path(entry['output_dir'])/n).stat().st_size for n in entry['output_sha256']
                      if (n.endswith(('.parquet','.json')) or n=='rejected_trips.csv') and (Path(entry['output_dir'])/n).exists())
            if sizes+entry['source'].get('bytes',0)+5*1024**2 > 3*GIB:
                raise ValueError('Actual cloud artifacts exceed preflight transfer budget')
            for kind, name, publish in [('cloud_publication','processed_publication',publish_month),
                                        ('source_backup','source_backup',publish_sources)]:
                guard.stage(name);entry=load_manifest(root)['archives'][month]
                if entry.get(kind,{}).get('status')=='published':
                    verify_publication(entry,kind,store)  # Reuse receipts, no re-upload/download or identity mutation.
                else:
                    options={'raw_root':work/'raw'} if kind=='source_backup' else {}
                    publish(month,root,project=PROJECT,bucket=BUCKET,store=store,max_file_bytes=FILE_CAP,
                            upload_attempts=1,upload_timeout=600,**options)
                entry=load_manifest(root)['archives'][month]
                live=verify_publication(entry,kind,store)
                write_json(report/f'{kind}-verified.json',live)
                removed.extend(cleanup_verification_scratch(entry[kind],tmp,store,entry,kind))
                write_json(report/'removed-verification-scratch.json',{'files':removed})
            guard.stage('temporary_reconciliation')
            kwargs={'report_root':root/'reconciliation-audits','workspace_parent':work/'engine',
                    'memory_limit':'512MB','spill_limit_bytes':4*GIB,'reserve_bytes':RESERVE}
            audit=reconcile_temporary(root,**kwargs)
            write_json(report/'reconciliation.json',audit)
            guard.stage('cached_idempotence_check');guard.forbid_sql=True
            cached=reconcile_temporary(root,**kwargs)
            guard.forbid_sql=False
            if (not cached['resumed'] or {k:v for k,v in cached.items() if k!='resumed'} !=
                    {k:v for k,v in audit.items() if k!='resumed'}):
                raise ValueError('Cached reconciliation differs')
            guard.stage('integrity')
            after=load_manifest(root)
            for m,e in plan['manifest']['archives'].items():
                if m!=month and after['archives'].get(m)!=e:raise ValueError(f'Prior manifest entry changed: {m}')
            for p,h in plan['canonical_hashes'].items():
                if fingerprint(Path(p))!=h:raise ValueError(f'Persistent canonical asset changed: {p}')
            for p in plan['inputs']:
                if fingerprint(Path(p['path']))!=p['sha256']:raise ValueError('Prior processed input changed')
            bindings=verify_views(root)
            entry=after['archives'][month]
            live=verify_publication(entry,'cloud_publication',store)+verify_publication(entry,'source_backup',store)
            write_json(report/'source-cleanup-candidates.json',source_cleanup_candidates(entry,work/'raw',store))
            write_json(report/'result.json',{'status':'verified','month':month,'ingestion':ingestion,
                'source_counts':{k:entry[k] for k in ('input_rows','accepted_rows','rejected_rows','verification')},
                'reconciliation':audit,'gcs_objects':live,'persistent_view_bindings':bindings,
                'cached_rerun_verified_no_SQL':True,'removed_verification_scratch':removed,
                'retention':'All original ZIP/CSV, monthly outputs, manifests, receipts and durable checkpoints retained.'})
            guard.stage('complete')
    finally:tempfile.tempdir=old_temp
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month',required=True)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/historical'))
    parser.add_argument('--work-root',type=Path)
    parser.add_argument('--report-root',type=Path)
    parser.add_argument('--september-budget',action='store_true',
                        help='Only 2024-09: prove the 26 GiB gate from verified August measurements and live metadata')
    args=parser.parse_args(argv)
    month=validate_month(args.month);root=args.output_root.resolve()
    work=(args.work_root or Path('outputs/monthly-work')/month).resolve()
    parent=(args.report_root or root/'monthly-cycles'/month).resolve()
    report=parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    report.mkdir(parents=True,exist_ok=False)
    try:
        with single_writer(root):
            store=GCSStorage(PROJECT,BUCKET,upload_timeout=600)
            run_cycle(month,root,work,report,store,september_budget=args.september_budget)
    except BaseException as error:
        write_json(report/'stopped.json',{'error':repr(error),'month':month,'automatic_retry':False})
        raise
    print('VERIFIED_MONTHLY_REPORT',str(report),flush=True)


if __name__=='__main__':main()
