"""Explicit recovery-bundle export and verified restore into a separate VM root.

No ETL, cloud writes, canonical-catalog copying, deletion or source restoration.
Export is offline; restore contacts GCS only when explicitly invoked.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from citypulse_ai.cloud.retention import restore_partition
from citypulse_ai.etl.archives import check_disk, fingerprint, validate_month, write_json
from citypulse_ai.etl.storage_state import output_receipts, source_receipts, revision_for

HEX = re.compile(r'[0-9a-f]{64}')
RESERVE = 15*1024**3
INPUTS = ('validated_trips.parquet', 'rejected_trips.csv')


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
        raise ValueError('Unsafe recovery path')
    p = root.joinpath(*relative.parts)
    if any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError('Symlink recovery path forbidden')
    return p


def validate_manifest(manifest):
    for month, entry in manifest['archives'].items():
        if validate_month(month) != entry['month']:
            raise ValueError('Manifest month mismatch')
        if entry.get('status') == 'processed':
            output_receipts(entry)
            source_receipts(entry)


def validate_audits(root):
    """Validate original checkpoint identities/hashes without opening DuckDB."""
    selected = set()
    audits = root/'reconciliation-audits'
    for p in audits.glob('*.json'):
        if not HEX.fullmatch(p.stem):
            continue
        bundle = json.loads(p.read_text()); payload = bundle['payload']
        if (bundle['sha256'] != revision_for(payload) or payload['status'] != 'complete'
                or revision_for(payload['specification']) != p.stem):
            raise ValueError(f'Invalid completed reconciliation audit: {p}')
        selected.add(p)
    for folder in (audits/'checkpoints').glob('*'):
        if not HEX.fullmatch(folder.name) or not folder.is_dir():
            continue
        for p in folder.glob('*.json'):
            bundle = json.loads(p.read_text()); payload = bundle['payload']
            if (bundle['sha256'] != revision_for(payload) or payload['identity'] != folder.name
                    or payload['stage'] != p.stem or payload['format_version'] != 1):
                raise ValueError(f'Invalid reconciliation checkpoint: {p}')
            selected.add(p)
            for receipt in payload['files']:
                if Path(receipt['name']).name != receipt['name']:
                    raise ValueError('Unsafe checkpoint asset name')
                asset = safe_path(root, str((folder/receipt['name']).relative_to(root)))
                if (not asset.is_file() or asset.stat().st_size != receipt['size_bytes']
                        or fingerprint(asset) != receipt['sha256']):
                    raise ValueError(f'Changed checkpoint asset: {asset}')
                selected.add(asset)
    for p in selected:
        safe_path(root, str(p.relative_to(root)))
    return selected


def copy_verified(source, destination, digest, size):
    if (not source.is_file() or source.stat().st_size != size or fingerprint(source) != digest):
        raise ValueError(f'Changed recovery source: {source}')
    if destination.exists():
        if not destination.is_file() or fingerprint(destination) != digest:
            raise ValueError(f'Recovery collision; preserve file: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=1024**2)
    if fingerprint(destination) != digest:
        raise ValueError(f'Recovery copy failed: {destination}')


def export_bundle(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError('Recovery bundle must be separate from historical root')
    manifest = json.loads((root/'manifest.json').read_text())
    validate_manifest(manifest)
    files = {root/'manifest.json', *validate_audits(root)}
    records = [{'name':str(p.relative_to(root)), 'sha256':fingerprint(p),
                'size_bytes':p.stat().st_size} for p in sorted(files)]
    check_disk(destination.parent, sum(r['size_bytes'] for r in records), RESERVE)
    destination.mkdir(parents=True, exist_ok=False)
    for r in records:
        copy_verified(safe_path(root,r['name']),safe_path(destination,r['name']),r['sha256'],r['size_bytes'])
    # Refuse a moving snapshot, retaining the partial bundle for inspection.
    if any(fingerprint(safe_path(root,r['name'])) != r['sha256'] for r in records):
        raise ValueError('Recovery sources changed during export; stop active writers')
    index = {'format_version':1, 'original_root':str(root), 'files':records,
             'excludes':['raw ZIP/CSV','monthly DuckDB databases','canonical snapshots','model artifacts']}
    write_json(destination/'bundle.json',index)
    return fingerprint(destination/'bundle.json')


def verify_bundle(bundle, expected_sha256):
    bundle = Path(bundle).resolve()
    if not HEX.fullmatch(expected_sha256) or fingerprint(safe_path(bundle,'bundle.json')) != expected_sha256:
        raise ValueError('Recovery bundle SHA-256 differs from trusted transfer record')
    index = json.loads((bundle/'bundle.json').read_text())
    if index['format_version'] != 1:
        raise ValueError('Unsupported recovery bundle version')
    names = set()
    for r in index['files']:
        name = r['name']; p = safe_path(bundle,name)
        if (name in names or not HEX.fullmatch(r['sha256']) or not p.is_file()
                or p.stat().st_size != r['size_bytes'] or fingerprint(p) != r['sha256']):
            raise ValueError(f'Invalid recovery file: {name}')
        names.add(name)
    expected = {'manifest.json', *(str(p.relative_to(bundle)) for p in validate_audits(bundle))}
    if names != expected:
        raise ValueError('Recovery index has missing or unexpected files')
    manifest = json.loads((bundle/'manifest.json').read_text())
    validate_manifest(manifest)
    return index, manifest


def restore_bundle(bundle, root, expected_sha256, store):
    """Resume copies/downloads; never overwrite an evolved VM working manifest."""
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    index, original = verify_bundle(bundle,expected_sha256)
    original_root = Path(index['original_root']).resolve()
    marker = {'format_version':1,'bundle_sha256':expected_sha256,'original_root':index['original_root']}
    owner = safe_path(root,'recovery-origin.json')
    owned = owner.is_file() and json.loads(owner.read_text()) == marker
    if root == bundle or bundle in root.parents or root in bundle.parents:
        raise ValueError('VM recovery root must be separate from original/bundle roots')
    # A replacement VM may use the same absolute pathname as a former VM.
    # Protect an existing source tree, while allowing a fresh/owned target there.
    if original_root.exists() and not owned and (
            root == original_root or original_root in root.parents or root in original_root.parents):
        raise ValueError('VM recovery root must be separate from existing original root')
    if root.exists() and any(root.iterdir()):
        if not owned:
            raise ValueError('Existing VM root is not owned by this recovery bundle')
    root.mkdir(parents=True,exist_ok=True)
    if not owner.exists():write_json(owner,marker)
    from citypulse_ai.etl.historical_cli import single_writer
    with single_writer(root):
        return _restore_owned(bundle,root,index,original,store)


def _restore_owned(bundle, root, index, original, store):
    manifest_path = safe_path(root,'manifest.json')
    mapped = deepcopy(original)
    for month, entry in mapped['archives'].items():
        entry['output_dir'] = str(root/f'source_month={month}')
    if not manifest_path.exists():
        write_json(manifest_path,mapped)
    current = json.loads(manifest_path.read_text())
    validate_manifest(current)
    # Finished old archives and their receipts remain immutable. An interrupted
    # September entry may legitimately advance through the normal monthly CLI.
    for month, entry in mapped['archives'].items():
        if entry.get('status') == 'processed' and current['archives'].get(month) != entry:
            raise ValueError('Original processed manifest entry changed')
    for month, entry in current['archives'].items():
        if entry['output_dir'] != str(root/f'source_month={month}'):
            raise ValueError('VM input mapping differs from owned recovery root')
    needed = sum(r['size_bytes'] for r in index['files']
                 if r['name'] != 'manifest.json' and not safe_path(root,r['name']).exists())
    for month, entry in current['archives'].items():
        if entry.get('status') == 'processed':
            for name, receipt in output_receipts(entry).items():
                if (name in INPUTS or name.endswith('.json')) and not safe_path(root,f'source_month={month}/{name}').exists():
                    needed += receipt['size_bytes']
            receipt = source_receipts(entry)['processing_manifest.json']
            if not safe_path(root,f'provenance/source_month={month}/processing_manifest.json').exists():
                needed += receipt['size_bytes']
    check_disk(root,needed,RESERVE)
    for r in index['files']:
        if r['name'] != 'manifest.json':
            copy_verified(safe_path(bundle,r['name']),safe_path(root,r['name']),r['sha256'],r['size_bytes'])
    validate_audits(root)
    restored = []
    for month, entry in current['archives'].items():
        if entry.get('status') != 'processed':
            continue
        receipts = output_receipts(entry)
        names = [n for n in receipts if n in INPUTS or n.endswith('.json')]
        folder = safe_path(root,f'source_month={month}')
        for name in names:safe_path(root,f'source_month={month}/{name}')
        restore_partition(entry,folder,store=store,names=names,reserve_bytes=RESERVE)
        provenance = safe_path(root,f'provenance/source_month={month}')
        safe_path(root,f'provenance/source_month={month}/processing_manifest.json')
        restore_partition(entry,provenance,store=store,kind='sources',
                          names=['processing_manifest.json'],reserve_bytes=RESERVE)
        for name in INPUTS:
            if fingerprint(folder/name) != entry['output_sha256'][name]:
                raise ValueError('Restored reconciliation input differs')
        restored.append(month)
    return {'restored_processed_months':restored,'working_manifest':str(manifest_path),
            'canonical_catalogs_restored':False,'original_manifest_sha256':fingerprint(bundle/'manifest.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    export = commands.add_parser('export')
    export.add_argument('--historical-root',type=Path,required=True)
    export.add_argument('--bundle',type=Path,required=True)
    restore = commands.add_parser('restore')
    restore.add_argument('--bundle',type=Path,required=True)
    restore.add_argument('--bundle-sha256',required=True)
    restore.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        print(json.dumps({'bundle_sha256':export_bundle(args.historical_root,args.bundle)}))
    else:
        # Explicit restore only. Import/authentication is not needed for export.
        from citypulse_ai.cloud.gcs import GCSStorage
        from citypulse_ai.etl.monthly_cycle import PROJECT, BUCKET
        print(json.dumps(restore_bundle(args.bundle,args.output_root,args.bundle_sha256,GCSStorage(PROJECT,BUCKET))))


if __name__ == '__main__':
    main()
