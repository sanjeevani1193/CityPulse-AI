"""Tiny verified CSV sources only; no real history, GCS or model training."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.test_temporary_reconciliation import backed_history, setup, audit
from tests.test_accepted_recorded_labels import complete, policy, START, END
from tests.test_station_labels import history
from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.csv_source_inventory import certify_csv_source, source_inventory
from citypulse_ai.features.station_history import StationHistory


def save(tmp_path, certificate):
    path = tmp_path/'csv-certificate.json'
    path.write_text(json.dumps(certificate))
    return path, fingerprint(path)


def test_real_receipt_validation_and_optional_reader_sidecar(backed_history, tmp_path):
    root, _, _, _ = backed_history
    result = audit(root, tmp_path)
    h = StationHistory(Path(result['audit_path']), root/'manifest.json')
    entry = json.loads(h.manifest_path.read_text())['archives']['2025-02']
    original = Path(entry['csv_files'][0]['input_file'])
    before = {p: fingerprint(p) for p in root.rglob('*') if p.is_file()}
    certificate = certify_csv_source(h, '2025-02', {original.name: original})
    assert certificate['original_zip_available'] is False
    assert certificate['zip_inventory_verified'] is False
    assert certificate['members'][0]['processed_row_coverage'] == {
        'rows': 2, 'unique_rows': 2, 'minimum': 2, 'maximum': 3}
    ref = save(tmp_path, certificate)
    restored = StationHistory(h.audit_path, h.manifest_path, csv_inventories={'2025-02': ref})
    _, evidence = source_inventory(restored, '2025-02', entry)
    assert evidence['source_member_inventory_verified']
    assert not evidence['archive_member_inventory_verified']
    assert all(fingerprint(p) == sha for p, sha in before.items())
    # Content-pinned optional evidence, not a mutable trust flag.
    ref[0].write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        restored._unchanged()


@pytest.fixture
def legacy(complete, tmp_path, monkeypatch):
    raw = tmp_path/'member.csv'
    pd.DataFrame({
        'ride_id': ['exact0001', 'exact0002', 'badstation'],
        'started_at': ['2024-06-30 23:59:59.123456789', '2024-07-01 00:05:00', '2024-07-01 01:00:00'],
        'ended_at': ['2024-07-01 00:10:00', '2024-07-01 00:15:00', '2024-07-01 01:10:00'],
        'start_station_id': ['6569.09_', '6569.09_', ''], 'end_station_id': ['', '', ''],
    }).to_csv(raw, index=False)
    m = json.loads(complete.manifest_path.read_text()); e = m['archives']['2024-07']
    e['source'] = {'sha256': None}
    e['schemas'] = []
    e['source_backup']['original_zip_available'] = False
    e['csv_files'] = [{'input_file': 'member.csv', 'input_sha256': fingerprint(raw), 'input_rows': 3}]
    complete.manifest_path.write_text(json.dumps(m))
    receipt = {'sha256': fingerprint(raw), 'size_bytes': raw.stat().st_size,
               'cloud_path': 'gs://synthetic/original/member.csv', 'generation': '1'}
    # Only cloud receipt authentication is substituted in this adapter fixture;
    # the test above exercises the real immutable receipt/spec validation.
    monkeypatch.setattr('citypulse_ai.features.csv_source_inventory.source_receipts',
                        lambda entry: {'member.csv': receipt})
    return complete, raw, receipt


def attach(legacy, tmp_path):
    h, raw, _ = legacy
    certificate = certify_csv_source(h, '2024-07', {'member.csv': raw})
    ref = save(tmp_path, certificate)
    h.csv_inventories = {'2024-07': ref}
    return h, certificate, ref


def test_accepted_recorded_policy_and_counts_unchanged_with_csv_evidence(legacy, tmp_path):
    h, certificate, _ = attach(legacy, tmp_path)
    r = policy(h).review(start=START, end=END)
    assert r.decisions.ride_count.tolist() == [1, 0, 0]
    assert r.decisions.count_complete.all()
    e = r.evidence['sources']['2024-07']
    assert e['source_member_inventory_verified'] and not e['archive_member_inventory_verified']
    assert e['source_inventory_kind'] == 'verified_original_csv_set_v1'
    assert r.evidence['policy']['version'] == 'accepted_recorded_starts_v1'
    readiness = StationHistory.verify_source_inventories(h, ['2024-07'])
    assert readiness['2024-07']['source_member_inventory_verified']
    assert certificate['scope'].endswith('not_publisher_or_ZIP_completeness')


@pytest.mark.parametrize('problem', ['missing', 'extra', 'filename', 'bytes'])
def test_restore_membership_filename_and_checksum_fail_closed(legacy, problem, tmp_path):
    h, raw, _ = legacy
    files = {'member.csv': raw}
    if problem == 'missing': files = {}
    if problem == 'extra': files['extra.csv'] = raw
    if problem == 'filename':
        other = tmp_path/'other.csv'; other.write_bytes(raw.read_bytes()); files['member.csv'] = other
    if problem == 'bytes': raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='CSV'):
        certify_csv_source(h, '2024-07', files)


@pytest.mark.parametrize('problem', ['manifest', 'source_identity', 'generation', 'schema', 'rows', 'zip'])
def test_certificate_identity_receipt_schema_and_coverage_fail_closed(legacy, tmp_path, problem):
    h, certificate, ref = attach(legacy, tmp_path)
    if problem == 'manifest': certificate['manifest_sha256'] = 'changed'
    if problem == 'source_identity': certificate['source_identity'] = 'changed'
    if problem == 'generation': certificate['members'][0]['receipt']['generation'] = '2'
    if problem == 'schema': certificate['members'][0]['columns'] = ['ride_id']
    if problem == 'rows': certificate['members'][0]['processed_row_coverage']['unique_rows'] = 2
    if problem == 'zip': certificate['zip_inventory_verified'] = True
    # Even a supplied SHA for a structurally invalid certificate cannot waive checks.
    ref[0].write_text(json.dumps(certificate))
    h.csv_inventories['2024-07'] = (ref[0], fingerprint(ref[0]))
    entry = json.loads(h.manifest_path.read_text())['archives']['2024-07']
    with pytest.raises(ValueError, match='CSV inventory'):
        source_inventory(h, '2024-07', entry)


def test_wrong_raw_schema_and_processed_row_coverage(legacy):
    h, raw, _ = legacy
    p = h.manifest_path.parent/'validated_trips.parquet'
    data = pd.read_parquet(p); data['source_row'] = 2; data.to_parquet(p, index=False)
    m = json.loads(h.manifest_path.read_text());m['archives']['2024-07']['output_sha256'][p.name] = fingerprint(p)
    h.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='source-row coverage'):
        certify_csv_source(h, '2024-07', {'member.csv': raw})


def test_missing_sidecar_keeps_existing_fail_closed_behavior(legacy):
    h, _, _ = legacy
    with pytest.raises(ValueError, match='evidence incomplete'):
        policy(h).review(start=START, end=END)


@pytest.mark.parametrize('problem', ['header', 'duplicate_header', 'row_count', 'ragged'])
def test_raw_csv_structure_is_stream_verified(legacy, problem):
    h, raw, receipt = legacy
    rows = raw.read_text().splitlines()
    if problem == 'header': rows[0] = rows[0].replace('ride_id', 'other_id')
    if problem == 'duplicate_header': rows[0] += ',ride_id'
    if problem == 'row_count': rows.pop()
    if problem == 'ragged': rows[-1] += ',extra'
    raw.write_text('\n'.join(rows)+'\n')
    # Isolate structure checks with a different, explicitly trusted synthetic
    # source identity, rather than bypassing the actual hash verification.
    m = json.loads(h.manifest_path.read_text())
    m['archives']['2024-07']['csv_files'][0]['input_sha256'] = fingerprint(raw)
    h.manifest_path.write_text(json.dumps(m))
    receipt.update(sha256=fingerprint(raw), size_bytes=raw.stat().st_size)
    with pytest.raises(ValueError, match='schema|row count|Ragged'):
        certify_csv_source(h, '2024-07', {'member.csv': raw})


def test_source_certificate_determinism_and_no_invented_zip(legacy):
    h, raw, _ = legacy
    a = certify_csv_source(h, '2024-07', {'member.csv': raw})
    assert a == certify_csv_source(h, '2024-07', {'member.csv': raw})
    assert 'members' not in json.loads(h.manifest_path.read_text())['archives']['2024-07']['source']
    assert a['members'][0]['original_source_file'] == 'member.csv'
    assert a['members'][0]['sha256'] == fingerprint(raw)


def test_native_inventory_evidence_preserves_existing_identity(complete):
    entry = json.loads(complete.manifest_path.read_text())['archives']['2024-07']
    files, evidence = source_inventory(complete, '2024-07', entry)
    assert files == entry['csv_files']
    assert evidence == {'archive_member_inventory_verified': True}


def test_original_zip_and_nonzero_duplicates_not_silently_certified(legacy, tmp_path):
    h, _, _ = attach(legacy, tmp_path)
    h.report['duplicate_ride_ids'] = 1
    with pytest.raises(ValueError, match='Nonzero duplicates'):
        StationHistory.verify_source_inventories(h, ['2024-07'])
    m = json.loads(h.manifest_path.read_text())
    m['archives']['2024-07']['source']['sha256'] = 'known-original-zip'
    h.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='unavailable original ZIP'):
        certify_csv_source(h, '2024-07', {})
