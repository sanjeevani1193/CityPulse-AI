"""Dry-run retention planning only. No cloud requests, file writes, or deletion."""

from pathlib import Path
import json

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.historical import load_manifest
from citypulse_ai.etl.storage_state import output_receipts, source_receipts, source_files


def retention_plan(output_root: Path, *, raw_root: Path, verify_local_hashes: bool = False) -> dict:
    """Refuse eligibility without complete verified sources, audits and output backups.

    Default uses metadata only; candidates require a hash-checked plan and a fresh
    cloud generation/recoverability check before any separately approved cleanup.
    This report never authorizes or executes deletion.
    """
    manifest = load_manifest(Path(output_root)); candidates = []; month_blockers = {}
    # A single-artifact receipt is useful inventory evidence, but never replaces
    # complete historical publication or a generation-pinned recovery contract.
    standalone = {}
    for receipt_path in (Path(output_root).parent / 'gcs-publications').glob('*.json'):
        if receipt_path.stat().st_size > 1024**2:
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get('uploaded') is True and receipt.get('verified') is True
                    and all(receipt.get('checks', {}).get(k) is True
                            for k in ('sha256_match', 'schema_match', 'row_count_match'))):
                standalone[str(Path(receipt['source_path']).resolve())] = {
                    'sha256': receipt['sha256'], 'cloud_path': receipt['destination'],
                    'download_path': receipt.get('download_path')}
        except (ValueError, KeyError, TypeError):
            continue
    for month, entry in sorted(manifest['archives'].items()):
        blockers = []
        try:
            outputs = output_receipts(entry)
        except (ValueError, KeyError, TypeError) as exc:
            outputs = {}; blockers.append(f'Output backup: {exc}')
        try:
            sources = source_receipts(entry)
        except (ValueError, KeyError, TypeError) as exc:
            sources = {}; blockers.append(f'Source/manifest backup: {exc}')
        try:
            originals = source_files(entry, raw_root)
        except (ValueError, KeyError, TypeError) as exc:
            originals = {}; blockers.append(f'Original-source provenance: {exc}')
        month_blockers[month] = list(blockers)
        local = {}
        for name, f in originals.items():
            local[f['path']] = (f['sha256'], 'original_source', sources.get(name), 'restore exact source bytes')
        # Extracted CSVs remain regenerable only when the original ZIP is backed up.
        zip_receipt = next((sources[n] for n, f in originals.items()
                            if f['role'] == 'original_zip' and n in sources), None)
        for item in entry.get('csv_files', []):
            path = item.get('path') or item.get('input_file')
            digest = item.get('sha256') or item.get('input_sha256')
            if path and path not in local:
                local[path] = (digest, 'extracted_csv', zip_receipt,
                               'restore original ZIP and re-extract; check recorded CSV SHA-256')
        for name, digest in entry.get('output_sha256', {}).items():
            if Path(name).name != name:
                raise ValueError('Unsafe output filename in retention manifest')
            path = str(Path(entry['output_dir']) / name)
            if name in outputs:
                local[path] = (digest, 'processed_output', outputs[name], 'restore exact verified output bytes')
            elif name == 'citypulse.duckdb':
                local[path] = (digest, 'derived_database', None,
                               'rebuild SQL tables/views from restored Parquet; byte identity not promised')
            elif standalone.get(str(Path(path).resolve()), {}).get('sha256') == digest:
                local[path] = (digest, 'standalone_published_output', standalone[str(Path(path).resolve())],
                               'standalone publication recorded; complete historical restore evidence still required')
            else:
                local[path] = (digest, 'unbacked_output', None, 'no verified recovery evidence')
        for publication_key, checked in [('cloud_publication', outputs), ('source_backup', sources)]:
            for name, f in entry.get(publication_key, {}).get('files', {}).items():
                copy_path = f.get('download_path')
                if copy_path and copy_path not in local:
                    local[copy_path] = (f.get('sha256'), 'verification_copy', checked.get(name),
                                        'download the same verified cloud generation again')
        for f in standalone.values():
            if f.get('download_path') and f['sha256'] in entry.get('output_sha256', {}).values():
                local.setdefault(f['download_path'], (f['sha256'], 'verification_copy', f,
                                 'standalone receipt recorded; complete historical evidence still required'))
        for path, (digest, role, receipt, method) in local.items():
            p = Path(path)
            if not p.exists():
                continue
            reasons = list(blockers)
            if p.is_symlink() or not p.is_file():
                reasons.append('Not a regular nonsymlink file')
            if role != 'derived_database' and receipt is None:
                reasons.append('Required source or output receipt missing')
            if not verify_local_hashes:
                reasons.append('Local SHA-256 not checked in metadata-only dry run')
            elif not p.is_file() or not digest or fingerprint(p) != digest:
                reasons.append('Local SHA-256 differs from recorded provenance')
            candidates.append({'month': month, 'path': str(p), 'size_bytes': p.stat().st_size,
                               'role': role, 'recovery_method': method,
                               'cloud_path': receipt.get('cloud_path') if receipt else None,
                               'required_cloud_backups': ['complete processed output publication',
                                                          'original ZIP or exact original CSVs',
                                                          'processing manifest snapshot and audit JSONs'],
                               'recorded_backup_eligible': not reasons, 'blockers': reasons,
                               'cleanup_authorized': False,
                               'remaining_requirement': 'Fresh GCS generation checks and explicit user approval'})
    return {'dry_run': True, 'deletion_supported': False, 'candidates': candidates,
            'month_blockers': month_blockers,
            'candidate_bytes': sum(f['size_bytes'] for f in candidates),
            'recorded_backup_eligible_bytes': sum(f['size_bytes'] for f in candidates if f['recorded_backup_eligible']),
            'note': 'Keep live manifests and receipts. Existing catalogue views may depend on restored cache files; never remove that cache while referenced.'}
