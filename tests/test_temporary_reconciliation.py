"""Small synthetic sources and mocked cloud receipts; no network or real history."""

import json
import fcntl
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest

from tests.test_cloud_retention import setup, outputs, sources
from citypulse_ai.cloud.historical import publish_month
from citypulse_ai.cloud.retention import publish_sources, restore_partition
from citypulse_ai.etl import historical, historical_cli, temporary_reconciliation as temporary
from citypulse_ai.etl.archives import fingerprint, write_json
from citypulse_ai.etl.history_catalog import build_catalog
from citypulse_ai.etl.pipeline import run_pipeline


@pytest.fixture
def backed_history(setup, tmp_path):
    root, raw, output, store, objects = setup
    outputs(root, store)
    sources(root, raw, store)
    march = tmp_path / "march.csv"
    march.write_text(
        "ride_id,started_at,ended_at,start_station_id,end_station_id\n"
        "feb,2025-03-01 01:00:00,2025-03-01 01:10:00,001.01,\n"
        "unique,2025-02-28 23:59:59.123456789,2025-03-01 00:10:00.123456789,6569.09_,\n"
        "boundary,bad,2025-03-01 01:10:00,001.01,\n")
    march_output = tmp_path / "march-output"
    run_pipeline(march, march_output, chunk_size=1)
    historical.register_existing("2025-03", march_output, root)
    publish_month("2025-03", root, project="project", bucket="bucket", store=store)
    publish_sources("2025-03", root, raw_root=raw, project="project", bucket="bucket",
                    max_file_bytes=1024**2, store=store)
    return root, [output, march_output], store, objects


def audit(root, tmp_path, **kwargs):
    return temporary.reconcile_temporary(root, report_root=tmp_path / "audits",
        workspace_parent=tmp_path / "scratch", reserve_bytes=0,
        spill_limit_bytes=1024**2, **kwargs)


def test_temporary_counts_duplicates_precision_and_persistent_compatibility(backed_history, tmp_path, monkeypatch):
    root, folders, store, _ = backed_history
    persistent = build_catalog(root, reserve_bytes=0, chunk_size=1)
    existing = {p: fingerprint(p) for p in root.rglob("*") if p.is_file()}
    source_hashes = {p: fingerprint(p) for folder in folders for p in folder.iterdir() if p.is_file()}
    original = temporary.create_reconciliation_relations
    def inspect(connection, entries, **kwargs):
        schemas = original(connection, entries, **kwargs)
        row = connection.execute("SELECT start_station_id,started_at_utc_ns FROM canonical_trips").fetchone()
        assert row == ("6569.09_", 1740805199123456789)
        return schemas
    monkeypatch.setattr(temporary, "create_reconciliation_relations", inspect)
    store.reset_mock()
    result = audit(root, tmp_path)
    assert (result["input_rows"], result["accepted_rows"], result["rejected_rows"]) == (5, 1, 4)
    assert result["local_accepted_rows"] == 4
    assert result["cross_archive_duplicate_ride_ids"] == 2
    assert result["additional_cross_archive_rejections"] == 3
    assert result["source_counts"] == {
        "2025-02": {"input_rows": 2, "locally_accepted_rows": 2, "locally_rejected_rows": 0},
        "2025-03": {"input_rows": 3, "locally_accepted_rows": 2, "locally_rejected_rows": 1}}
    assert result["source_start_month_counts"] == result["boundary_start_counts"] == [
        {"source_archive_month": "2025-03", "start_month": "2025-02", "ride_starts": 1}]
    assert result["start_month_counts"] == {"2025-02": 1}
    assert result["rejection_reason_counts"]["duplicate_ride_id"] == 4
    assert result["rejection_reason_counts"]["started_at:invalid_timestamp"] == 1
    for key in ["accepted_rows", "rejected_rows", "cross_archive_duplicate_ride_ids",
                "start_month_counts", "boundary_start_counts", "hourly_ride_starts"]:
        assert result[key] == persistent[key]
    assert all(fingerprint(p) == h for p, h in existing.items())
    assert all(fingerprint(p) == h for p, h in source_hashes.items())
    assert not list((tmp_path / "scratch").iterdir())
    assert not list((tmp_path / "audits").rglob("*.duckdb"))
    with duckdb.connect(str(Path(persistent["snapshot"]) / "catalog.duckdb"), read_only=True) as c:
        assert c.execute("SELECT count(*) FROM canonical_trips").fetchone()[0] == 1
    bundle = json.loads(Path(result["audit_path"]).read_text())
    assert bundle["payload"]["manifest_snapshot"] == historical.load_manifest(root)
    assert bundle["sha256"] == temporary.revision_for(bundle["payload"])
    assert bundle["payload"]["persistent_catalog_modified"] is False
    execution = bundle["payload"]["execution"]
    assert execution["duplicates"]["projected_occurrences"] == 5
    assert execution["duplicates"]["partition_count"] == 2
    assert execution["duplicates"]["partition_bytes"] < 1024**2 // 2
    assert execution["checkpoint_budget_bytes"] == 1024**2 // 4
    assert execution["hourly_strategy"] == "disjoint_utc_days"
    assert {key: value for key, value in execution.items()
            if key not in ("duplicates", "checkpoint_budget_bytes", "checkpoint_root", "hourly_strategy")} == {
        "database": ":memory:", "memory_limit": "512MB",
        "spill_limit_bytes": 1024**2, "duckdb_version": duckdb.__version__,
        "engine_spill_limit_bytes": 1024**2 // 2}
    assert bundle["payload"]["provenance"]["2025-02"]["source_backup"]["status"] == "published"
    assert store.mock_calls == []


def test_input_relocation_resume_after_removal_and_survivor_validation(backed_history, tmp_path):
    root, folders, _, _ = backed_history
    moved = {}
    for month, folder in zip(["2025-02", "2025-03"], folders):
        new = tmp_path / f"temporary-{month}"
        folder.rename(new)
        moved[month] = new
    first = audit(root, tmp_path, input_paths=moved)
    saved = Path(first["audit_path"]).read_bytes()
    for folder in moved.values():
        folder.rename(folder.with_name(folder.name + "-preserved"))
    second = audit(root, tmp_path, input_paths=moved)
    assert second["resumed"] is True and second["revision"] == first["revision"]
    assert Path(first["audit_path"]).read_bytes() == saved
    assert not (root / "canonical").exists()
    # A changed surviving input cannot be hidden by a previously complete audit.
    moved["2025-02"].mkdir()
    (moved["2025-02"] / "validated_trips.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        audit(root, tmp_path, input_paths=moved)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interruption_cleans_only_owned_workspace_and_restarts(backed_history, tmp_path, monkeypatch, failure):
    root, folders, _, _ = backed_history
    before = (root / "manifest.json").read_bytes()
    sentinel = tmp_path / "scratch" / "prior-data"
    sentinel.parent.mkdir(); sentinel.write_text("keep")
    original = temporary.reconciliation_report
    def interrupted(*args, **kwargs):
        assert list(sentinel.parent.glob("reconciliation-*"))
        raise failure("simulated interruption")
    monkeypatch.setattr(temporary, "reconciliation_report", interrupted)
    with pytest.raises(failure):audit(root, tmp_path)
    assert list(sentinel.parent.iterdir()) == [sentinel]
    assert not list((tmp_path / "audits").glob("*.json"))
    assert (root / "manifest.json").read_bytes() == before
    assert all((folder / "validated_trips.parquet").is_file() for folder in folders)
    monkeypatch.setattr(temporary, "reconciliation_report", original)
    monkeypatch.setattr(temporary, "build_partitioned_duplicates",
                        Mock(side_effect=AssertionError('Verified duplicates should be reused')))
    assert audit(root, tmp_path)["accepted_rows"] == 1
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("bad", ["source_backup", "cloud_publication", "generation", "hash", "counts"])
def test_incomplete_recovery_or_wrong_counts_refuses_audit(backed_history, tmp_path, bad):
    root, _, _, _ = backed_history
    manifest = historical.load_manifest(root)
    entry = manifest["archives"]["2025-02"]
    if bad in ["source_backup", "cloud_publication"]:
        entry[bad]["status"] = "interrupted"
    elif bad == "generation":
        entry["cloud_publication"]["files"]["validated_trips.parquet"]["generation"] = ""
    elif bad == "hash":
        entry["cloud_publication"]["files"]["validated_trips.parquet"]["sha256"] = "0" * 64
    else:
        entry["input_rows"] += 1
    write_json(root / "manifest.json", manifest)
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError):audit(root, tmp_path)
    assert (root / "manifest.json").read_bytes() == before
    assert not (root / "canonical").exists()


def test_changed_manifest_during_execution_does_not_publish_audit(backed_history, tmp_path, monkeypatch):
    root, _, _, _ = backed_history
    original = temporary.reconciliation_report
    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        manifest = historical.load_manifest(root); manifest["external_update"] = True
        write_json(root / "manifest.json", manifest)
        return report
    monkeypatch.setattr(temporary, "reconciliation_report", changed)
    with pytest.raises(ValueError, match="Manifest changed"):audit(root, tmp_path)
    assert not list((tmp_path / "audits").glob("*.json"))
    assert not list((tmp_path / "scratch").iterdir())


def test_corrupt_completed_audit_is_not_overwritten(backed_history, tmp_path):
    root, _, _, _ = backed_history
    result = audit(root, tmp_path)
    path = Path(result["audit_path"])
    bundle = json.loads(path.read_text()); bundle["payload"]["report"]["accepted_rows"] = 999
    path.write_text(json.dumps(bundle))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete or changed"):audit(root, tmp_path)
    assert path.read_bytes() == before


def test_preflight_workspace_separation_and_offline_cli(backed_history, tmp_path, monkeypatch, capsys):
    root, folders, _, _ = backed_history
    with pytest.raises(ValueError, match="independently"):
        temporary.reconcile_temporary(root, report_root=folders[0] / "audit")
    with pytest.raises(ValueError, match="workspace parent"):
        temporary.reconcile_temporary(root, report_root=tmp_path / "scratch" / "audit",
                                      workspace_parent=tmp_path / "scratch")
    with pytest.raises(ValueError, match="persistent canonical"):
        temporary.reconcile_temporary(root, report_root=root / "canonical" / "audit")
    with pytest.raises(ValueError, match="separate from input"):
        temporary.reconcile_temporary(root, workspace_parent=folders[0] / "scratch")
    monkeypatch.setattr(temporary, "check_disk", Mock(side_effect=ValueError("Insufficient disk space")))
    with pytest.raises(ValueError, match="disk space"):audit(root, tmp_path)
    assert not list((tmp_path / "scratch").iterdir())
    monkeypatch.undo()
    import builtins
    original = builtins.__import__
    def offline(name, *args, **kwargs):
        if name.startswith(("google", "citypulse_ai.cloud")):
            raise AssertionError("Temporary CLI imported cloud")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", offline)
    historical_cli.main(["--reconcile-temporary", "--output-root", str(root),
        "--reconciliation-report-root", str(tmp_path / "cli-audits"),
        "--reconciliation-workspace-parent", str(tmp_path / "cli-scratch"),
        "--reserve-gib", "0", "--reconciliation-spill-gib", "0.001"])
    assert json.loads(capsys.readouterr().out)["accepted_rows"] == 1


def test_audit_writer_lock_and_missing_inputs_preserve_state(backed_history, tmp_path):
    root, folders, _, _ = backed_history
    reports = tmp_path / "audits"; reports.mkdir()
    with (reports / ".reconciliation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another temporary reconciliation"):
            audit(root, tmp_path)
    folders[0].rename(folders[0].with_name("preserved"))
    with pytest.raises(FileNotFoundError, match="Restore verified"):
        audit(root, tmp_path)
    assert not list(reports.glob("*.json"))
    assert not (root / "canonical").exists()


def test_workspace_resource_config_and_source_count_mismatch(backed_history, tmp_path, monkeypatch):
    root, _, _, _ = backed_history
    original_connect = temporary.duckdb.connect
    configs = []
    def connect(*args, **kwargs):
        assert args == (":memory:",)
        configs.append(kwargs["config"])
        return original_connect(*args, **kwargs)
    monkeypatch.setattr(temporary.duckdb, "connect", connect)
    original_relations = temporary.create_reconciliation_relations
    def mismatch(connection, entries, **kwargs):
        schemas = original_relations(connection, entries, **kwargs)
        # A manifest-consistent receipt is not proof that its claimed row count
        # matches the underlying data: execute and check every source partition.
        connection.execute("CREATE OR REPLACE VIEW raw_ride_occurrences AS "
                           "SELECT * FROM locally_valid_trips WHERE false")
        return schemas
    monkeypatch.setattr(temporary, "create_reconciliation_relations", mismatch)
    with pytest.raises(ValueError, match="Source counts differ"):
        audit(root, tmp_path)
    assert configs[0]["memory_limit"] == "512MB"
    assert configs[0]["max_temp_directory_size"] == "524288B"
    assert not list((tmp_path / "scratch").iterdir())


def test_mock_cloud_recovery_supplies_exact_temporary_inputs(backed_history, tmp_path):
    root, _, store, _ = backed_history
    manifest = historical.load_manifest(root)
    before = (root / "manifest.json").read_bytes()
    paths = {}
    for month, entry in manifest["archives"].items():
        destination = tmp_path / "restored" / month
        paths[month] = restore_partition(entry, destination, store=store,
            names=list(temporary.INPUT_NAMES), reserve_bytes=0)
        assert {p.name for p in destination.iterdir()} == set(temporary.INPUT_NAMES)
        for name in temporary.INPUT_NAMES:
            assert fingerprint(destination / name) == entry["output_sha256"][name]
    result = audit(root, tmp_path, input_paths=paths)
    assert result["accepted_rows"] == 1 and result["cross_archive_duplicate_ride_ids"] == 2
    assert (root / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("options", [["--month", "2024-06"], ["--reconcile"], ["--cloud-resume"],
    ["--cloud-publish"], ["--reconciliation-input", "2024-06="],
    ["--reconciliation-input", "2024-06=a", "--reconciliation-input", "2024-06=b"]])
def test_cli_forbids_mixed_modes_and_invalid_overrides(options):
    with pytest.raises(SystemExit):historical_cli.main(["--reconcile-temporary", *options])


def retain_stage_checkpoints(root, tmp_path, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(temporary, 'reconciliation_report', Mock(side_effect=RuntimeError('after stages')))
        with pytest.raises(RuntimeError, match='after stages'):
            audit(root, tmp_path)
    return next((tmp_path / 'audits' / 'checkpoints').iterdir())


def test_verified_zero_checkpoint_reuse_and_filter_plan(setup, tmp_path, monkeypatch):
    root, raw, _, store, _ = setup
    outputs(root, store); sources(root, raw, store)
    stage = retain_stage_checkpoints(root, tmp_path, monkeypatch)
    assert json.loads((stage / 'duplicates.json').read_text())['payload']['data']['duplicate_ride_ids'] == 0
    monkeypatch.setattr(temporary, 'build_partitioned_duplicates',
                        Mock(side_effect=AssertionError('must reuse completed audit')))
    original = temporary.create_reconciliation_relations
    def inspect(connection, entries, **kwargs):
        result = original(connection, entries, **kwargs)
        plan = connection.execute('EXPLAIN SELECT count(*) FROM canonical_trips').fetchone()[1]
        assert 'DELIM' not in plan and 'HASH_GROUP_BY' not in plan
        return result
    monkeypatch.setattr(temporary, 'create_reconciliation_relations', inspect)
    result = audit(root, tmp_path)
    execution = json.loads(Path(result['audit_path']).read_text())['payload']['execution']['duplicates']
    assert execution['checkpoint_reused'] is True
    assert execution['exclusion'] == 'verified_zero'
    assert result['accepted_rows'] == 2
    assert audit(root, tmp_path)['resumed'] is True


@pytest.mark.parametrize('bad', ['checksum','identity','coverage','false_zero','asset','missing_coverage','occurrences'])
def test_checkpoint_fail_closed_before_filter(backed_history, tmp_path, monkeypatch, bad):
    root, _, _, _ = backed_history
    stage = retain_stage_checkpoints(root, tmp_path, monkeypatch)
    path = stage / ('source-counts.json' if bad in ('coverage','missing_coverage') else 'duplicates.json')
    bundle = json.loads(path.read_text())
    if bad == 'asset':
        asset = stage / bundle['payload']['files'][0]['name']
        asset.write_bytes(b'corrupt parquet; preserve')
    elif bad == 'missing_coverage':
        path.rename(path.with_suffix('.preserved'))
    else:
        if bad == 'checksum':bundle['sha256'] = '0'*64
        elif bad == 'identity':bundle['payload']['identity'] = 'different-input-manifest'
        elif bad == 'coverage':bundle['payload']['data']['input'].pop('2025-02')
        elif bad == 'false_zero':bundle['payload']['data'].update(duplicate_ride_ids=0,cross_archive_duplicate_ride_ids=0)
        elif bad == 'occurrences':bundle['payload']['data']['projected_occurrences'] -= 1
        if bad != 'checksum':bundle['sha256'] = temporary.revision_for(bundle['payload'])
        path.write_text(json.dumps(bundle))
    monkeypatch.setattr(temporary, 'build_hourly', Mock(side_effect=AssertionError('must fail before filtering')))
    with pytest.raises(ValueError):audit(root, tmp_path)
    assert not list((tmp_path / 'audits').glob('*.json'))


def test_missing_interrupted_duplicate_checkpoint_recomputes_audit(backed_history, tmp_path, monkeypatch):
    root, _, _, _ = backed_history
    stage = retain_stage_checkpoints(root, tmp_path, monkeypatch)
    path = stage / 'duplicates.json'
    path.rename(path.with_suffix('.interrupted-preserved'))
    original = temporary.build_partitioned_duplicates
    fresh = Mock(wraps=original)
    monkeypatch.setattr(temporary, 'build_partitioned_duplicates', fresh)
    result = audit(root, tmp_path)
    assert fresh.call_count == 1
    assert result['cross_archive_duplicate_ride_ids'] == 2
    assert result['accepted_rows'] == 1
