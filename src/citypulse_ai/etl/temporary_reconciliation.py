"""Offline, restartable reconciliation without publishing a persistent catalogue."""

import fcntl
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb

from citypulse_ai.etl.archives import check_disk, fingerprint, validate_month, write_json
from citypulse_ai.etl.history_catalog import (
    create_reconciliation_relations, reconciliation_report,
)
from citypulse_ai.etl.storage_state import output_receipts, revision_for, source_receipts
from citypulse_ai.etl.partitioned_duplicates import build_partitioned_duplicates
from citypulse_ai.etl.reconciliation_checkpoints import Checkpoints
from citypulse_ai.etl.bounded_hourly import build_hourly
from citypulse_ai.etl.history_catalog import sql_literal, create_canonical_trip_view

INPUT_NAMES = ("validated_trips.parquet", "rejected_trips.csv")
FORMAT_VERSION = 1


def validate_duplicate_result(connection, evidence: dict) -> None:
    """Check persisted audit claims against its actual, content-verified result."""
    schema = [(row[0], row[1]) for row in connection.execute('DESCRIBE duplicate_ids').fetchall()]
    if schema != [('normalized_ride_id','VARCHAR'), ('occurrences','BIGINT'), ('archive_count','BIGINT')]:
        raise ValueError('Duplicate checkpoint schema differs')
    count, cross, unique, invalid = connection.execute("""SELECT count(*),
        count(*) FILTER(WHERE archive_count>1),count(DISTINCT normalized_ride_id),
        count(*) FILTER(WHERE normalized_ride_id IS NULL OR normalized_ride_id=''
          OR normalized_ride_id<>trim(normalized_ride_id) OR occurrences IS NULL
          OR occurrences<2 OR archive_count IS NULL OR archive_count<1 OR archive_count>occurrences)
        FROM duplicate_ids""").fetchone()
    if (invalid or unique != count or count != evidence.get('duplicate_ride_ids')
            or cross != evidence.get('cross_archive_duplicate_ride_ids')):
        raise ValueError('Duplicate checkpoint counts or result integrity differ')


def reconcile_temporary(output_root: Path, *, input_paths: dict[str, Path] | None = None,
                        report_root: Path | None = None, workspace_parent: Path | None = None,
                        memory_limit: str = "512MB", spill_limit_bytes: int = 4 * 1024**3,
                        reserve_bytes: int = 5 * 1024**3) -> dict:
    """Audit all processed months using verified local or restored input paths.

    No network, manifest updates, persistent SQL, source deletion or automatic
    restoration. Complete recorded source/output receipts are mandatory; live
    generation checks and source cleanup remain separate caller responsibilities.
    Only the new private engine workspace is removed, including on interruption.
    A completed content-addressed audit can be reused after inputs are removed.
    """
    root = Path(output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = [entry for _, entry in sorted(manifest["archives"].items())
               if entry.get("status") == "processed"]
    if not entries:
        raise ValueError("No complete source partitions registered")
    for month, entry in manifest["archives"].items():
        if entry.get("status") == "processed" and entry.get("month") != validate_month(month):
            raise ValueError("Archive key differs from registered source month")
    overrides = input_paths or {}
    if set(overrides) - {entry["month"] for entry in entries}:
        raise ValueError("Input overrides must identify registered processed months")
    if spill_limit_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("Spill limit must be positive and reserve nonnegative")
    reports = Path(report_root).resolve() if report_root else root / "reconciliation-audits"
    parent = Path(workspace_parent).resolve() if workspace_parent else reports / ".work"
    canonical = root / "canonical"
    if any(path == canonical or canonical in path.parents for path in (reports, parent)):
        raise ValueError("Temporary audit paths cannot modify the persistent canonical directory")
    if reports == parent or parent in reports.parents:
        raise ValueError("Audit reports cannot be inside the temporary workspace parent")
    resolved_entries, provenance = [], {}
    for entry in entries:
        month = entry["month"]
        outputs, sources = output_receipts(entry), source_receipts(entry)
        folder = Path(overrides.get(month, entry["output_dir"])).resolve()
        if reports == folder or folder in reports.parents:
            raise ValueError("Audit reports must survive independently of input directories")
        if parent == folder or folder in parent.parents:
            raise ValueError("Engine workspace must be separate from input directories")
        resolved_entries.append({**entry, "output_dir": str(folder)})
        provenance[month] = {
            "source": entry["source"], "input_rows": entry["input_rows"],
            "accepted_rows": entry["accepted_rows"], "rejected_rows": entry["rejected_rows"],
            "output_sha256": entry["output_sha256"],
            "cloud_publication": entry["cloud_publication"], "source_backup": entry["source_backup"],
            "required_reconciliation_inputs": {name: outputs[name] for name in INPUT_NAMES},
            "processing_manifest": sources["processing_manifest.json"],
        }
    # Paths/attempt timing/current snapshot are deliberately excluded from identity.
    # Retain the full immutable manifest separately for provenance/reconstruction.
    specification = {"format_version": FORMAT_VERSION,
                     "modelling_specification": manifest["modelling_specification"],
                     "inputs": {month: {
                         "source": p["source"], "input_rows": p["input_rows"],
                         "accepted_rows": p["accepted_rows"], "rejected_rows": p["rejected_rows"],
                         "output_sha256": p["output_sha256"],
                         "output_revision": p["cloud_publication"]["revision"],
                         "source_revision": p["source_backup"]["revision"],
                         "cloud_objects": {kind: {name: {
                             key: receipt[key] for key in ("cloud_path", "generation", "sha256", "size_bytes")
                         } for name, receipt in p[kind]["files"].items()}
                             for kind in ("cloud_publication", "source_backup")},
                     } for month, p in provenance.items()}}
    revision = revision_for(specification)
    reports.mkdir(parents=True, exist_ok=True)
    target = reports / f"{revision}.json"
    with (reports / ".reconciliation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another temporary reconciliation owns this report root") from error
        try:
            # Surviving inputs must always match, even when an audit is reusable.
            for entry in resolved_entries:
                for name in INPUT_NAMES:
                    path = Path(entry["output_dir"]) / name
                    if path.exists() and (not path.is_file() or path.is_symlink()
                                          or fingerprint(path) != entry["output_sha256"][name]):
                        raise ValueError(f"Reconciliation input changed: {path}")
            if target.exists():
                bundle = json.loads(target.read_text())
                payload = bundle.get("payload", {})
                if (bundle.get("sha256") != revision_for(payload)
                        or payload.get("specification") != specification
                        or payload.get("status") != "complete"):
                    raise ValueError(f"Existing audit is incomplete or changed: {target}")
                if manifest_path.read_bytes() != manifest_bytes:
                    raise ValueError("Manifest changed during reconciliation; retry with current provenance")
                return {"audit_path": str(target), "revision": revision,
                        "resumed": True, **payload["report"]}
            for entry in resolved_entries:
                for name in INPUT_NAMES:
                    if not (Path(entry["output_dir"]) / name).is_file():
                        raise FileNotFoundError(f"Restore verified {entry['month']}/{name} before reconciliation")
            parent.mkdir(parents=True, exist_ok=True)
            # An in-memory database avoids a separately growing on-disk database;
            # projections and engine spill share the explicitly bounded scratch budget.
            check_disk(parent, spill_limit_bytes, reserve_bytes)
            started = perf_counter()
            # The existing scratch ceiling covers both ID projections and spill.
            partition_budget = spill_limit_bytes // 4
            checkpoint_budget = spill_limit_bytes // 4
            engine_spill_budget = spill_limit_bytes - partition_budget - checkpoint_budget
            checkpoints = Checkpoints(reports / 'checkpoints' / revision, revision,
                                      checkpoint_budget, reserve_bytes)
            partitions = 2
            while partitions < (sum(entry["input_rows"] for entry in entries) + 249_999) // 250_000:
                partitions *= 2
            duplicate_execution = {}
            hourly_validation = None
            with TemporaryDirectory(prefix="reconciliation-", dir=parent) as work:
                with duckdb.connect(":memory:", config={
                    "memory_limit": memory_limit, "threads": 2,
                    "temp_directory": str(Path(work) / "spill"),
                    "max_temp_directory_size": f"{engine_spill_budget}B",
                }) as connection:
                    connection.execute("SET TimeZone = 'UTC'")
                    def duplicate_builder(engine):
                        source_stage = checkpoints.load('source-counts')
                        saved = checkpoints.load('duplicates')
                        if saved is not None and source_stage is None:
                            raise ValueError('Duplicate checkpoint lacks verified source coverage')
                        if source_stage is None:
                            raw = dict(engine.execute('SELECT source_archive_month,count(*) '
                                'FROM raw_ride_occurrences GROUP BY 1').fetchall())
                            valid = dict(engine.execute('SELECT source_archive_month,count(*) '
                                'FROM locally_valid_trips GROUP BY 1').fetchall())
                            for entry in entries:
                                if (raw.get(entry['month'], 0) != entry['input_rows']
                                        or valid.get(entry['month'], 0) != entry['accepted_rows']):
                                    raise ValueError('Source counts differ from manifest')
                            source_stage = checkpoints.save('source-counts', {'input': raw, 'accepted': valid,
                                'rejected': {month: count-valid.get(month, 0) for month,count in raw.items()}})
                        expected = {kind: {entry['month']: entry[field] for entry in entries}
                                    for kind, field in [('input','input_rows'), ('accepted','accepted_rows'),
                                                        ('rejected','rejected_rows')]}
                        if source_stage['data'] != expected:
                            raise ValueError('Duplicate checkpoint source coverage differs from manifest')
                        nonempty = engine.execute("SELECT count(*) FROM raw_ride_occurrences "
                                                  "WHERE normalized_ride_id <> ''").fetchone()[0]
                        if saved is not None:
                            if saved['data'].get('projected_occurrences') != nonempty:
                                raise ValueError('Duplicate checkpoint occurrence coverage differs from inputs')
                            if len(saved['files']) != 1:
                                raise ValueError('Duplicate checkpoint must contain exactly one result file')
                            path = checkpoints.root / saved['files'][0]['name']
                            engine.execute('CREATE TABLE duplicate_ids AS SELECT * FROM read_parquet('
                                           + sql_literal(path) + ', hive_partitioning=false)')
                            duplicate_execution.update(saved['data'])
                            validate_duplicate_result(engine, duplicate_execution)
                            duplicate_execution['checkpoint_reused'] = True
                            return
                        duplicate_execution.update(build_partitioned_duplicates(
                            engine, Path(work) / "ids", partition_count=partitions,
                            disk_budget_bytes=partition_budget, reserve_bytes=reserve_bytes))
                        count, cross = engine.execute('SELECT count(*),count(*) FILTER '
                            '(WHERE archive_count>1) FROM duplicate_ids').fetchone()
                        duplicate_execution.update(duplicate_ride_ids=count,
                                                   cross_archive_duplicate_ride_ids=cross)
                        if duplicate_execution['projected_occurrences'] != nonempty:
                            raise ValueError('Duplicate audit occurrence coverage differs from inputs')
                        validate_duplicate_result(engine, duplicate_execution)
                        path = checkpoints.asset('duplicates')
                        engine.execute('COPY duplicate_ids TO ' + sql_literal(path) + ' (FORMAT PARQUET)')
                        checkpoints.save('duplicates', duplicate_execution, [path])
                        duplicate_execution['checkpoint_reused'] = False
                    def canonical_builder(engine):
                        # Reached only after hash/identity, complete source coverage,
                        # and actual duplicate-result invariants have passed.
                        mode = 'verified_zero' if duplicate_execution['duplicate_ride_ids'] == 0 else 'anti'
                        create_canonical_trip_view(engine, exclusion=mode)
                        duplicate_execution['exclusion'] = mode
                    def hourly_builder(engine):
                        nonlocal hourly_validation
                        hourly_validation = build_hourly(engine, checkpoints)
                    schemas = create_reconciliation_relations(
                        connection, resolved_entries, duplicate_builder=duplicate_builder,
                        hourly_builder=hourly_builder, canonical_builder=canonical_builder)
                    observed = dict(connection.execute(
                        "SELECT source_archive_month,count(*) FROM raw_ride_occurrences "
                        "GROUP BY source_archive_month").fetchall())
                    accepted = dict(connection.execute(
                        "SELECT source_archive_month,count(*) FROM locally_valid_trips "
                        "GROUP BY source_archive_month").fetchall())
                    for entry in entries:
                        month = entry["month"]
                        if (observed.get(month, 0) != entry["input_rows"]
                                or accepted.get(month, 0) != entry["accepted_rows"]):
                            raise ValueError(f"Source counts differ from manifest: {month}")
                    report = reconciliation_report(connection, resolved_entries, manifest, schemas, started,
                                                   hourly_validation=hourly_validation)
                    report["source_counts"] = {entry["month"]: {
                        "input_rows": observed.get(entry["month"], 0),
                        "locally_accepted_rows": accepted.get(entry["month"], 0),
                        "locally_rejected_rows": observed.get(entry["month"], 0) - accepted.get(entry["month"], 0),
                    } for entry in entries}
                    report["source_start_month_counts"] = [
                        {"source_archive_month": source, "start_month": month, "ride_starts": count}
                        for source, month, count in connection.execute(
                            "SELECT source_archive_month,start_month,count(*) FROM canonical_trips "
                            "GROUP BY source_archive_month,start_month ORDER BY 1,2").fetchall()]
                    report["rejection_reason_counts"] = dict(connection.execute(
                        "SELECT reason,count(*) FROM canonical_rejections, "
                        "unnest(string_split(rejection_reasons,';')) AS reasons(reason) "
                        "WHERE reason <> '' GROUP BY reason ORDER BY reason").fetchall())
            # Engine handles are closed and its owned scratch directory is gone.
            for entry in resolved_entries:
                for name in INPUT_NAMES:
                    path = Path(entry["output_dir"]) / name
                    if fingerprint(path) != entry["output_sha256"][name]:
                        raise ValueError(f"Input changed during reconciliation: {path}")
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError("Manifest changed during reconciliation; retry with current provenance")
            payload = {"status": "complete", "specification": specification,
                       "manifest_snapshot": manifest, "provenance": provenance, "report": report,
                       "mode": "temporary_audit_only", "persistent_catalog_modified": False,
                       "execution": {"database": ":memory:", "memory_limit": memory_limit,
                                     "spill_limit_bytes": spill_limit_bytes,
                                     "duckdb_version": duckdb.__version__,
                                     "duplicates": duplicate_execution,
                                     "checkpoint_budget_bytes": checkpoint_budget,
                                     "checkpoint_root": str(checkpoints.root),
                                     "hourly_strategy": "disjoint_utc_days",
                                     "engine_spill_limit_bytes": engine_spill_budget},
                       "cloud_evidence": "recorded full-content receipts; live generations not checked here"}
            write_json(target, {"sha256": revision_for(payload), "payload": payload})
            return {"audit_path": str(target), "revision": revision, "resumed": False, **report}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
