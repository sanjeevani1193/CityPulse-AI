"""Explicit evidence for legacy CSV-backed sources; never invent a ZIP inventory.

Certificates supplement, rather than edit, the immutable historical manifest.
Loading requires a separately trusted certificate SHA-256 (for example its
generation-pinned publication receipt), just like historical recovery bundles.
"""

import csv
import json
from pathlib import Path

import duckdb

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.history_catalog import rejected_csv_columns, sql_literal
from citypulse_ai.etl.schema import RAW_COLUMNS
from citypulse_ai.etl.storage_state import revision_for, source_files, source_receipts


def _legacy_source(entry):
    if (entry['source'].get('sha256') or entry['source'].get('members')
            or entry['source_backup'].get('original_zip_available') is not False):
        raise ValueError('CSV-only evidence requires an explicitly unavailable original ZIP')
    receipts = source_receipts(entry)
    files = entry.get('csv_files', [])
    originals = source_files(entry, Path('.'))
    if (not files or any(type(f.get('input_rows')) is not int or f['input_rows'] < 0 for f in files)
            or any(f['role'] != 'original_csv' for f in originals.values())
            or sum(f['input_rows'] for f in files) != entry['input_rows']):
        raise ValueError('Complete original CSV membership and row counts required')
    return receipts, {Path(f.get('path') or f['input_file']).name: f for f in files}


def _binding(history, month, entry):
    return {'source_identity': history.identity,
            'manifest_sha256': fingerprint(history.manifest_path),
            'source_archive_month': month, 'source_entry_sha256': revision_for(entry)}


def certify_csv_source(history, month, restored_files):
    """Stream original CSV structure and verify exact processed row coverage.

    No validation/rejection rule is rerun. The existing accepted/rejected files
    must partition every original data-row ordinal exactly once, per CSV.
    Native query memory is capped at 512 MB; external spill is disabled.
    """
    history._unchanged()
    entry = json.loads(history.manifest_path.read_text())['archives'][month]
    receipts, originals = _legacy_source(entry)
    if set(restored_files) != set(originals):
        raise ValueError('Restore exactly the original CSV set')
    folder = Path(entry['output_dir'])
    processed = [folder/n for n in ('validated_trips.parquet', 'rejected_trips.csv')]
    for p in processed:
        if fingerprint(p) != entry['output_sha256'][p.name]:
            raise ValueError('Processed source changed')
    members = []
    with duckdb.connect(config={'memory_limit': '512MB', 'threads': 1,
                               'max_temp_directory_size': '0B'}) as con:
        rejected = processed[1]
        columns = {name: 'VARCHAR' for name in rejected_csv_columns(rejected)}
        con.read_csv(str(rejected), header=True, auto_detect=False, columns=columns).create_view('rejected_source')
        con.execute('CREATE VIEW accepted_source AS SELECT source_file,source_row FROM read_parquet('
                    + sql_literal(processed[0]) + ',hive_partitioning=false)')
        for table, expected in [('accepted_source', entry['accepted_rows']),
                                ('rejected_source', entry['rejected_rows'])]:
            if con.execute('SELECT count(*) FROM '+table).fetchone()[0] != expected:
                raise ValueError('Processed source row count differs')
        for name, original in sorted(originals.items()):
            path = Path(restored_files[name]); receipt = receipts[name]
            digest = original.get('sha256') or original['input_sha256']
            if (path.name != name or path.is_symlink() or not path.is_file()
                    or path.stat().st_size != receipt['size_bytes'] or fingerprint(path) != digest):
                raise ValueError('Restored CSV filename, size or SHA-256 differs')
            count = 0
            with path.open(newline='', encoding='utf-8-sig') as stream:
                reader = csv.reader(stream, strict=True)
                headers = next(reader, [])
                if (not set(RAW_COLUMNS) <= set(headers) or len(set(headers)) != len(headers)
                        or any(not h for h in headers)):
                    raise ValueError('Raw CSV schema differs')
                for row in reader:
                    if len(row) != len(headers):
                        raise ValueError('Ragged original CSV record')
                    count += 1
            if count != original['input_rows']:
                raise ValueError('Original CSV row count differs')
            source = original.get('path') or original['input_file']
            rows, unique, minimum, maximum = con.execute('''
                SELECT count(*),count(DISTINCT source_row),min(source_row),max(source_row)
                FROM (SELECT source_row FROM accepted_source WHERE source_file=? UNION ALL
                      SELECT CAST(source_row AS BIGINT) FROM rejected_source WHERE source_file=?)
                ''', [source, source]).fetchone()
            if ((rows, unique) != (count, count) or
                    (count and (minimum, maximum) != (2, count+1))):
                raise ValueError('Accepted/rejected source-row coverage differs')
            if fingerprint(path) != digest:
                raise ValueError('Original CSV changed during certification')
            members.append({'filename': name, 'original_source_file': source, 'sha256': digest,
                'size_bytes': path.stat().st_size, 'input_rows': count, 'columns': headers,
                'receipt': {k: receipt[k] for k in ('cloud_path', 'generation', 'sha256', 'size_bytes')},
                'processed_row_coverage': {'rows': rows, 'unique_rows': unique,
                                           'minimum': minimum, 'maximum': maximum}})
    for p in processed:
        if fingerprint(p) != entry['output_sha256'][p.name]:
            raise ValueError('Processed source changed during certification')
    history._unchanged()
    return {'format_version': 1, 'kind': 'verified_original_csv_set_v1',
            **_binding(history, month, entry), 'original_zip_available': False,
            'zip_inventory_verified': False, 'members': members,
            'scope': 'recorded_original_CSV_set_not_publisher_or_ZIP_completeness'}


def source_inventory(history, month, entry):
    """Native archive evidence or an explicitly supplied, hash-pinned CSV sidecar."""
    members = entry['source'].get('members', [])
    files, schemas = entry.get('csv_files', []), entry.get('schemas', [])
    expected = {m['name'] for m in members}
    complete = bool(expected) and (len(expected) == len(members) == len(files) == len(schemas)
        and {f.get('member') for f in files} == expected
        and {s.get('member') for s in schemas} == expected
        and all(f.get('sha256') for f in files))
    sidecars = getattr(history, 'csv_inventories', {})
    if month not in sidecars:
        # Keep native/uncertified evidence byte-for-byte compatible: historical
        # label evidence identities must not change merely by installing this.
        return files, {'archive_member_inventory_verified': complete}
    if complete:
        raise ValueError('CSV sidecar cannot replace an existing archive inventory')
    path, trusted_sha256 = sidecars[month]; path = Path(path)
    if path.is_symlink() or fingerprint(path) != trusted_sha256:
        raise ValueError('CSV inventory certificate SHA-256 differs')
    proof = json.loads(path.read_text())
    receipts, originals = _legacy_source(entry)
    if (proof.get('format_version') != 1 or proof.get('kind') != 'verified_original_csv_set_v1'
            or any(proof.get(k) != v for k, v in _binding(history, month, entry).items())
            or proof.get('original_zip_available') is not False
            or proof.get('zip_inventory_verified') is not False):
        raise ValueError('CSV inventory source identity or manifest differs')
    records = proof['members']
    if len(records) != len(originals) or {r['filename'] for r in records} != set(originals):
        raise ValueError('CSV inventory membership differs')
    normalized = []
    for r in records:
        f, receipt = originals[r['filename']], receipts[r['filename']]
        source = f.get('path') or f['input_file']; count = f['input_rows']
        coverage = r['processed_row_coverage']
        if (r['original_source_file'] != source or r['sha256'] != receipt['sha256']
                or r['size_bytes'] != receipt['size_bytes'] or r['input_rows'] != count
                or r['receipt'] != {k: receipt[k] for k in ('cloud_path','generation','sha256','size_bytes')}
                or not set(RAW_COLUMNS) <= set(r['columns'])
                or len(set(r['columns'])) != len(r['columns']) or any(not c for c in r['columns'])
                or coverage != {'rows': count, 'unique_rows': count,
                                'minimum': 2 if count else None, 'maximum': count+1 if count else None}):
            raise ValueError('CSV inventory schema, receipt or row coverage differs')
        normalized.append({'member': r['filename'], 'path': source, 'sha256': r['sha256'],
                           'input_rows': count})
    return normalized, {'archive_member_inventory_verified': False,
        'source_member_inventory_verified': True, 'source_inventory_kind': proof['kind'],
        'csv_inventory_certificate_sha256': trusted_sha256,
        'original_zip_available': False}
