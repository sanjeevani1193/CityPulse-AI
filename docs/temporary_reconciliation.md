# Temporary historical reconciliation and recovery

`--reconcile-temporary` is a standalone, offline audit mode. It runs the same
cross-archive duplicate, count, station-hour and NYC start-month rules as the
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

An exclusively created child workspace contains bounded ID projections and engine
spill files. The DuckDB database is in memory. Verified duplicate results and
hourly partitions survive separately under the report root for recovery. Engine
relations disappear when its connection closes and its child workspace is removed.
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
without a completed audit reuses independently verified stage checkpoints after
rechecking input hashes. SQL connections themselves are never checkpointed. The
full manifest is checked for concurrent changes before publication.

Receipt validation is offline. It checks complete sets, publication identities,
sizes, positive generations, source hashes and recorded full downloaded-byte
verification. It cannot prove current GCS object availability or unchanged live
generations. **Before cleanup**, the authorized ingestion caller must recheck
live objects/generations and perform the existing generation-pinned SHA-256
verification or restoration. A completed audit alone never authorizes cleanup.

## Resource limits and June readiness

The memory default remains `512MB`; this mode does not increase it. The explicit
scratch ceiling defaults to 4 GiB (`--reconciliation-spill-gib`). Half is reserved
for engine spill; one quarter is reserved for compressed duplicate-ID projections
and one quarter for retained checkpoints. Before opening the
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
Partition bytes, limits and strategy are recorded in the audit. ID projections and
spill are owned reproducible scratch and disappear with the temporary workspace. Verified duplicate results are retained as checkpoints.
No source or processed input is removed. The data identity/version is unchanged
because this changes physical execution, not reconciliation semantics. Existing
matching complete audits remain reusable; an interrupted attempt verifies input
hashes and checkpoint integrity before reusing completed stages. Uncommitted partition files are never reused.

For June recovery use `--memory-limit 512MB --reserve-gib 12` with the existing
4 GiB scratch ceiling. Free space is checked before execution and during every
projection batch and partition. The caller should also monitor the complete run
and interrupt above the reserve if other processes consume storage. DuckDB's
memory setting is not a total Python process RSS cap. No memory increase or
fallback to approximate/adjacent-month duplicate checks occurs on failure.

Partitioning reduces aggregate state as history grows, but runtime and projected
scratch grow with the full history; disk preflight remains required each time.
Highly skewed partitions or a very large duplicate-result table can still exhaust
the fixed memory limit. Station-hour aggregation also retains the existing 512 MB engine limit and
uses the disjoint daily strategy below. These failures stop the audit without publishing incomplete
counts. Full 24-month scalability requires measurement, not an assumption based
on one successful June run.


## Daily station-hour checkpoints

Temporary mode groups one elapsed UTC day at a time across **all** source archives,
then writes the original four-column hourly schema to Parquet. A station/hour can
only belong to one UTC day; NYC `start_month` still comes from the original trip
instant. Boundary rides from adjacent source archives are combined before counting.
Repeated local DST hours remain separate UTC hours; no hours or labels are filled.
The existing rule rejects every occurrence of duplicate ride IDs; there is no winner
or new precedence rule. Persistent catalog building keeps its original SQL path.

Each daily file is checked for expected accepted ride totals, positive counts,
nonempty exact station IDs, hour alignment, day bounds, NYC month, and exact
station/hour uniqueness. Disjoint day ranges plus those local checks prove global
uniqueness without constructing another history-wide distinct hash table. The
union's row and ride totals are checked again before the shared final report.

Under `checkpoints/<input-specification-SHA>/` the importer retains source counts,
a full-ID duplicate-result Parquet and counts, canonical source/local-month and
UTC-day subtotals, and completed daily Parquets. Checksummed versioned stage JSON
records include file SHA-256, bytes and partition counts. These are intermediate
results, not final canonical publication. They remain useful if a later step fails.
All original input hashes are checked before reuse. Completed daily files are
rehashed and their actual invariants rechecked on resume. A completed matching final
audit remains idempotent; no daily rows are appended or counted twice.

An interrupted write without stage metadata is preserved and excluded from reads.
Restart uses a new filename rather than overwriting it. Orphaned files count toward
the per-input checkpoint quota; quota exhaustion stops safely and requires explicit
review, never automatic deletion. Old input revisions are not reused for new source
sets. Retained revisions consume disk, so later historical months still require
preflight and an eventual explicitly approved checkpoint retention policy.

With the default 4 GiB total budget: ID projections have 1 GiB, retained checkpoints
1 GiB, and engine spill 2 GiB. Both report and workspace filesystems are checked.
Checkpoints are budget-checked between partitions, and the controlled caller must
monitor overall disk use with a margin above its minimum reserve. No fixed memory
setting proves all future daily workloads fit; extreme station cardinality or a
large duplicate-result table must fail safely rather than increase memory.


### Controlled attempt status (2026-10-14)

The daily algorithm passes synthetic equivalence, boundary, DST and interruption
checks. The one full attempt failed **before** its daily loop, during canonical
UTC-day subtotal discovery. EXPLAIN shows the inherited correlated `NOT EXISTS`
expands to `LEFT_DELIM_JOIN` plus a hash grouping over ride IDs, introducing
history-wide state even for a query with very few day groups. The next temporary
mode change should test an explicit exact anti join that avoids this correlation;
then review every canonical scan before another authorized full attempt.

The duplicate checkpoint survived and verifies zero duplicates over all eight
archives. No daily partition or complete canonical audit was produced. May is
not finalized and June is provisional. Thus the full method is **not yet proven
memory-safe** at 512 MB, despite passing the synthetic tests. Retain checkpoints
and failure evidence; do not repeat the full run automatically.

## Verified duplicate exclusion

Temporary mode now removes the inherited correlated `NOT EXISTS` filter. Its
replacement is chosen only after validating complete source-count coverage against
the current manifest, input-specification identity, input file hashes, checkpoint
metadata and result-file hashes. It also counts actual nonempty normalized ID
occurrences and requires the audit's projected occurrence count to match. The
loaded duplicate table must have the exact expected schema, unique nonempty trimmed
IDs, valid occurrence/archive counts, and row/cross-archive counts matching the audit.

If this verified result contains zero duplicate IDs, the canonical trip view reads
locally accepted trips directly. If nonzero, an explicit `ANTI JOIN` excludes all
occurrences whose trimmed ride ID matches a duplicate ID. Case sensitivity, exact
stored ride/station strings, original timestamps, and output schemas are unchanged.
There is no winning duplicate or new precedence policy. Persistent catalog building
retains its original exclusion query.

A missing duplicate stage causes a new exact audit; it never implies zero. A present
but corrupt, mismatched or incomplete stage fails closed. A duplicate stage without
its independently verified per-source coverage record is refused. SQL plan tests
check that both temporary strategies remove `DELIM` operators and history-wide ID
grouping, while matching original canonical rows and station-hour counts on fixtures.
Large nonzero duplicate-result tables remain subject to the same 512 MB memory cap;
correct semantics do not constitute a guarantee that every future workload fits.


### Verified controlled result

The subsequent correlated-filter fix completed one full attempt at 512 MB and
reused the verified zero-duplicate checkpoint. The actual eight-archive plan has
no DELIM operator under either optimized strategy; the original plan had one.
All 245 daily partitions and final invariants passed: 23,993,387 accepted starts,
13,622 rejected rows, 5,876,513 unique station-hours. May's boundary-reconciled
count is 4,133,356 for this source set; June's 4,780,481 remains provisional.
A cached rerun produced the identical audit without opening a database connection.

210 relevant tests passed. Observed peak process RSS was 174.17 MiB; engine spill
was zero, and retained checkpoints were approximately 17.79 MiB. Swap allocation
remained 3 GiB and free space stayed above 29.89 GiB. These measurements apply to
this run, which reused the duplicate audit; they do not measure a fresh full-history
duplicate construction or establish resource safety for every future nonzero set.
All original inputs, manifests and persistent snapshots remain unchanged. Detailed
local evidence is under `outputs/historical/june-2024-duplicate-filter-fix/`.

### Rejected CSV binding at larger archive counts

The April 2025 attempt failed before duplicate auditing when DuckDB bound the
combined `raw_ride_occurrences` view. `all_varchar=true` still performs automatic
CSV sniffing. A 24-source tiny-file reproduction at 512 MB exhausted memory while
requesting another 30.5 MiB block; this is independent of the daily aggregation
and cannot be solved by its external spill directory.

The reader now supplies the complete checked header as explicit VARCHAR columns
and disables automatic inference, using the comma/double-quote dialect emitted by
the ETL audit writer. It preserves optional end-station compatibility, original
strings, strict malformed-record handling, maximum line size, and every rejected
occurrence used by the exact global duplicate audit. It does not materialize a
new historical dataset or change checkpoint identity or canonical rules.

The regression scans 24 synthetic source partitions at 512 MB and compares their
rejected values with the original reader applied one file at a time, including
embedded delimiters/newlines, leading-zero IDs and cross-source duplicates.

### Duplicate projection metadata and process RSS

May 2025 stopped at the existing 2 GiB process-RSS safeguard while aggregating
the hash partitions. The algorithm did not retain all IDs in Python: it streamed
50,000-row batches and grouped exact full strings in one hash partition at a time.
However, writing every small bucket slice created a Parquet row group for every
input batch/bucket pair. Footer metadata grew with history, and the dictionary of
closed native Parquet writers remained alive during DuckDB aggregation. DuckDB's
512 MB engine limit does not bound all Arrow/native process allocations.

ID projection now coalesces slices into 4,096-row Parquet groups. Shared pending
Arrow slices are capped at 64 MiB between writes (plus the current input batch/slice
and writer/codec overhead); the largest bucket is flushed if that byte cap is
reached. Projection runs in its own scope, releasing readers, buffers and closed
writers before aggregation. These are disposable ID projections only: original
records, full-ID equality, duplicate exclusion, partition routing, source coverage,
checkpoint identity, canonical schemas and the disk ceilings are unchanged.

The process-RSS guard remains 2 GiB on both Mac and VM; no 3 GiB exception is used.
This reduces metadata overhead rather than promising every future input fits.
Regression tests check increasing archive counts, exact duplicate/no-duplicate
results, writer destruction before scans, tiny batches and forced byte-cap flushing.
In a 200,000-row, 24-source synthetic probe, peak RSS was about 190 MiB before and
166 MiB after; projection bytes fell from 2,547,681 to 1,075,987. These are local
fixture measurements, not a forecast of full-history memory usage. The initial
writer-lifetime-only experiment did not reduce RSS, so lifetime isolation alone
was not accepted as sufficient evidence for recovery.
