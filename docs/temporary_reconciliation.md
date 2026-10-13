# Temporary historical reconciliation and recovery

`--reconcile-temporary` is a standalone, offline audit mode. It runs the same
cross-archive duplicate, count, station-hour and NYC start-month SQL as the
existing canonical catalog, using registered partitions or explicitly supplied
restored directories. It retains a small JSON audit and never promotes a DuckDB
database, canonical trip/hourly Parquet exports, `canonical/CURRENT.json`, or a
manifest update. Existing `--reconcile` behavior remains available and unchanged.

## Existing dependencies

The June storage audit identified these dependencies in both existing snapshots:

- `outputs/historical/canonical/snapshots/03ed2464c76747d5b6c026af8b5c932a/catalog.duckdb`
- `outputs/historical/canonical/snapshots/754f8610e41e4d3c94e17323cd28dd3c/catalog.duckdb`

`locally_valid_trips` reads `validated_trips.parquet`; `raw_ride_occurrences`
reads that Parquet and `rejected_trips.csv`. `canonical_trips` and
`canonical_rejections` depend on those views. The first snapshot references all
seven registered source months; the second predates May and references six.
Their exact absolute input paths are recorded under `snapshot_dependency_files`
in [the storage audit](../outputs/storage-audits/june-2024-cloud-first/cleanup_candidates.json).
The exact input files are both `validated_trips.parquet` and `rejected_trips.csv`
inside each of these repository-relative directories (14 files in the current
snapshot; May's two files are absent from the older snapshot):

- `outputs/historical/source_month=2024-01/`
- `outputs/historical/source_month=2024-02/`
- `outputs/historical/source_month=2024-03/`
- `outputs/historical/source_month=2024-04/`
- `outputs/historical/source_month=2024-05/`
- `outputs/phase13-february-reviewed/`
- `outputs/historical/source_month=2025-03/`

The corresponding manifest `output_dir` values identify these same files. A
cloud-restored persistent catalog instead depends on the exact restore-cache
paths recorded when it was built. Materialized `hourly_rides` and `duplicate_ids`
tables do not remove the upstream view dependencies.

Temporary mode prevents **new** persistent dependencies. It does not rewrite
either existing snapshot or permit deletion of their inputs. Existing development
forecasting also requires January–March 2024 validated, rejection and hourly
files locally. Those consumers still need restoration before reuse if their
dependencies are eventually changed under a separately approved task.

## Running an audit

Example only; this task does not ingest or access cloud data:

```sh
PYTHONPATH=src python -m citypulse_ai.etl.historical_cli \
  --output-root outputs/historical \
  --reconcile-temporary \
  --reconciliation-input 2024-06=/temporary/run/june-processed \
  --reconciliation-report-root outputs/historical/reconciliation-audits \
  --reconciliation-workspace-parent /temporary/run/engine
```

The manifest must already register every included month as fully processed,
with complete verified processed publication and original-source/immutable
processing-manifest backup receipts. June cannot be included before these
requirements are satisfied. All registered processed months are included; a
single-month override changes its input location, not duplicate-check scope.
Only `validated_trips.parquet` and `rejected_trips.csv` are needed locally for
each month. Missing files must be restored separately with the existing
generation-pinned restore workflow. Temporary mode never downloads them.

The API is `citypulse_ai.etl.temporary_reconciliation.reconcile_temporary`.
It accepts the same path overrides and resource options. Input directories are
read-only to this function. Reports must be outside input directories and the
temporary workspace parent; callers must also keep the report root outside any
larger run directory they intend to clean up.

An exclusively created child workspace contains bounded ID projections and engine spill files; the
DuckDB database itself is in memory and is never written to a database file.
All relations disappear when its connection closes and that child is removed.
Only this function's newly created engine scratch is automatically removed,
including after Python exceptions or interruption. Input Parquets, rejection
CSVs, other scratch directories, old datasets and source archives are never
removed. Abrupt process termination or power loss can leave engine scratch;
do not delete unidentified leftovers without checking ownership and active use.

## Counts and provenance

The shared SQL retains exact station strings and integer UTC nanosecond epochs,
validates epoch consistency, aggregates elapsed UTC hours, and assigns start
months in `America/New_York`. It inherits existing DST/missing/invalid timestamp
validation rather than introducing new rules. Duplicate detection includes valid
and locally rejected IDs from every included archive and rejects every occurrence.

Audits retain local source accepted/rejected/input counts, globally accepted and
rejected counts, additional cross-archive rejections, duplicate-ID counts,
global NYC start-month totals, accepted source/start-month counts (including
boundaries), hourly uniqueness/count reconciliation, and rejection-reason counts.
Reason counts can overlap when a row has several reasons. All local source counts
must match the manifest. The audit includes schema alignment, exact source/output
hashes, cloud paths/generations/verification receipts, modelling specification and
a full manifest snapshot. It is a count/provenance audit, not a new row-level
canonical dataset. Per-row rejections and duplicates are reproducible from the
retained cloud inputs and shared SQL; original local rejection audits remain backed up.

No missing station-hours become zero labels. Source archive completeness does
not establish observation coverage, station operating status, or final start-month
completeness. Later adjacent archives can change earlier start-month totals.

## Restart and receipt trust

The audit filename is the SHA-256 identity of the input specification, including
source hashes, processed hashes, counts, publication revisions, GCS object
generations and modelling specification. Input locations, current canonical
snapshot and elapsed runtime do not define data identity. A complete audit is
published atomically under a report-root writer lock, with a checksum of its
payload. The format version must be incremented when reconciliation semantics
change; retain the code revision/environment alongside audit metadata for replay.

Repeated execution reuses a complete matching audit even after temporary inputs
have been removed. Surviving inputs must still hash correctly. Changed or corrupt
audit files are refused and preserved, not overwritten. Interrupted computation
without a completed audit restarts from verified inputs; there is no partial SQL
checkpoint. The full manifest is checked for concurrent changes before publication.

Receipt validation is offline. It checks complete sets, publication identities,
sizes, positive generations, source hashes and recorded full downloaded-byte
verification. It cannot prove current GCS object availability or unchanged live
generations. **Before cleanup**, the authorized ingestion caller must recheck
live objects/generations and perform the existing generation-pinned SHA-256
verification or restoration. A completed audit alone never authorizes cleanup.

## Resource limits and June readiness

The memory default remains `512MB`; this mode does not increase it. The explicit
scratch ceiling defaults to 4 GiB (`--reconciliation-spill-gib`). Half is reserved
for compressed duplicate-ID projections and half for engine spill. Before opening the
engine, preflight requires the full scratch budget plus the configured reserve (default
5 GiB) on the engine filesystem. There is no separately growing engine database
file and no canonical Parquet export in temporary mode. This bounds engine spill
usage, not total process RAM or other pipeline stages. Raw extraction,
monthly ETL, restored inputs and publication-verification copies require separate
space checks on their actual filesystems. The existing monthly ETL still extracts
CSV members and its staging spill has no explicit byte ceiling.

June's persistent-catalog blocker is resolved **when temporary mode is used**.
A future June retry still needs a fresh combined storage plan, run-private
raw/output/TMPDIR locations, completed source/processed backups, live generation
and content verification, and a separately authorized June-only cleanup ledger.
Publication currently retains verification downloads; track their exact paths.
Temporary mode does not implement that ingestion orchestration or cleanup, does
not make the old snapshots portable, and does not prove a real June run will fit
the approximately 21 GiB available during the earlier audit. A failed space check
must stop processing; do not raise memory or spill budgets repeatedly to force it.

## Reconstructing canonical history from GCS

1. Retain the audit bundle and immutable source-month processing manifests. Match
   their source/CSV hashes, accepted/rejected counts, output hashes, schema/timezone
   policies and exact GCS object generations. Check live generations first.
2. Restore required processed Parquet and rejection CSV objects with
   `restore_partition`, selected by recorded generation and expected SHA-256.
   Matching local files may be reused; conflicting files must be preserved and
   investigated. Restore the original ZIP (or exact CSV backups where no ZIP was
   retained) and recheck member hashes if processed data must be regenerated.
3. Rebuild a working manifest from immutable entries and their verified receipts;
   explicitly replace obsolete processing locations with restored paths. Preserve
   source identity, exact station strings and UTC epochs. Do not overwrite the
   live manifest or silently alter an old snapshot to perform replay.
4. Run temporary reconciliation to reproduce small audit counts, or run the
   existing persistent `build_catalog` in a **new** reconstruction root with
   persistent restored input paths. Include every original source month, including
   rejection CSVs, to reproduce duplicate rejection. A different source set can
   legitimately change boundary totals and canonical membership.
5. Compare accepted/rejected/hourly/boundary/duplicate totals and preserved epoch
   precision with the audit. Logical reconstruction is supported; byte-identical
   DuckDB files, Parquet encodings or random snapshot UUIDs are not promised.

Existing cloud objects are sufficient inputs for logical reconstruction when
their generations remain available. Temporary mode checks the receipt contract,
not live availability; this task performs no live restoration or GCS access.

## Durable cloud canonical snapshots still need support

A future durable canonical publication needs an immutable, content-addressed
snapshot manifest listing every canonical trip/hourly/rejection/duplicate artifact
with SHA-256, bytes, row counts, schema, source lineage, policy/code versions and
verified object generations. Publication must be create-only, resumable and
verified by retrieving bytes. Recovery must restore a complete self-contained
artifact set, reconcile counts, and rebuild relations against restored locations.
Views must reference that set or fully materialized tables, never abandoned
source caches or run workspaces. A completion receipt/pointer can be promoted
only after every artifact is verified. Old snapshots and their dependencies
must remain intact until an explicitly approved migration verifies replacement
and recovery. No canonical snapshot was built or uploaded for this task.

## Bounded global duplicate recovery

Temporary mode scans all registered accepted Parquet rows and rejected CSV rows
once to project only the existing trimmed nonempty ride IDs and archive months.
It routes these projections by the first 64 bits of MD5, using at least two
partitions and doubling the partition count until the average is at most 250,000
input occurrences. MD5 only chooses a partition: aggregation compares original
full normalized strings, so hash collisions cannot introduce false duplicates.
Case sensitivity, whitespace normalization, within-source duplicates,
nonadjacent cross-archive duplicates, and rejection of every occurrence are
unchanged. Persistent catalog construction retains its original SQL path.

Arrow batches contain at most 50,000 rows. Each partition is then processed with
the original `count(*)` and `count(DISTINCT source_archive_month)` expressions.
Partition bytes, limits and strategy are recorded in the audit. Projections and
spill are owned reproducible scratch and disappear with the temporary workspace.
No source or processed input is removed. The data identity/version is unchanged
because this changes physical execution, not reconciliation semantics. Existing
matching complete audits remain reusable; an interrupted attempt restarts from
verified inputs rather than trusting partial partitions.

For June recovery use `--memory-limit 512MB --reserve-gib 12` with the existing
4 GiB scratch ceiling. Free space is checked before execution and during every
projection batch and partition. The caller should also monitor the complete run
and interrupt above the reserve if other processes consume storage. DuckDB's
memory setting is not a total Python process RSS cap. No memory increase or
fallback to approximate/adjacent-month duplicate checks occurs on failure.

Partitioning reduces aggregate state as history grows, but runtime and projected
scratch grow with the full history; disk preflight remains required each time.
Highly skewed partitions or a very large duplicate-result table can still exhaust
the fixed memory limit. Later station-hour aggregation also retains the existing
512 MB engine limit. These failures stop the audit without publishing incomplete
counts. Full 24-month scalability requires measurement, not an assumption based
on one successful June run.
