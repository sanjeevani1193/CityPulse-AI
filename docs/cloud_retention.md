# Cloud-backed historical storage and retention

This workflow preserves the full 24-month modelling specification and January
2026 boundary input. It changes storage/resume, not trip validation, forecasting,
source/archive-month assignment, or cross-month duplicate rules.

## Backup order and identity

1. Finish and validate local ETL; publish the complete processed source partition
   with the existing `--publish-existing-only` command.
2. Separately back up original sources and processing metadata with the new
   storage command. It does not run ETL or download official archives.
3. Inspect a retention dry run, verify live cloud generations/recovery, and seek
   explicit approval before any future cleanup. No deletion command exists.

When the historical manifest records a retained ZIP hash, backup requires that
exact ZIP at `--raw-root/archives/<source-key>`. It does not substitute CSVs for a
missing ZIP. For registered February 2025, the original ZIP was not retained:
backup instead requires all exact original CSV paths/hashes in the manifest.
Objects are labelled `original_csv`, and `original_zip_available=false`; no ZIP
is recreated or claimed to be backed up. Original ZIP bytes are hashed, not
repacked. Audits and an immutable per-month processing-manifest snapshot are
included. The snapshot contains source identity, CSV/output hashes, ETL counts,
publication receipts, model-window specification, and attempts; it excludes its
own mutable `source_backup` state to avoid a self-referencing checksum.

Source publication uses a separate prefix and revision:

```text
gs://<bucket>/raw/historical/source_archive_month=YYYY-MM/revision=<SHA-256>/<filename>
```

`source_backup.status` and per-file receipts are separate from local ETL status
and `cloud_publication.status`. Immutable revisions include source provenance,
artifact roles, hashes, and paths. A source, destination, or snapshot change
refuses reuse of the existing revision; investigate rather than overwrite it.
A later change to processed-publication receipts can therefore require a newly
reviewed source snapshot. Keep the live manifest and receipts permanently; they
are not cleanup candidates. ZIP/extraction sidecars, unregistered old runs,
notebooks, and arbitrary temporary files are also outside this planner's scope.

All transfers reuse the existing generation-zero upload precondition, bounded
transient retries/backoff, and generation-pinned SHA-256 download verification.
Prior verified objects are rechecked, not overwritten. Failed/partial files and
remote objects are retained. The month is backed up only after every selected
source/metadata object verifies. No cloud resource or IAM change is made.

## Explicit commands — separately authorize real transfers

The source size ceiling is mandatory; choose it from actual preflight sizes and
available disk space, not from these placeholders. Verification retains an
additional downloaded copy of each uploaded source, so budget space for it.

```sh
# Existing original ZIP (or February's original CSVs): upload and verify.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.storage_cli backup-sources \
  --month YYYY-MM --output-root outputs/historical \
  --raw-root data/raw/citibike --project citypulse-ai-511315 \
  --bucket citypulse-ai-511315-data --max-file-mib <reviewed-limit>

# Offline, metadata-only dry run; no ADC, network, hashing, or deletion.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.storage_cli plan

# Stronger offline evidence: hashes local candidate files, still no deletion.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.storage_cli plan --verify-local-hashes

# Explicit recovery to a NEW or hash-matching persistent cache.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.storage_cli restore \
  --month YYYY-MM --kind outputs --destination /path/to/cache/YYYY-MM \
  --project citypulse-ai-511315 --bucket citypulse-ai-511315-data
# --kind sources restores original ZIP/CSVs plus audits and manifest snapshot.
```

Restore requires complete publication receipts, expected SHA-256, positive
recorded generations, source-month provenance and byte sizes. Each live remote
object must still match that evidence; each restored file is hashed. Existing
local files must match or fail without replacement. Failed partial downloads
require a new destination on retry. Space is checked before downloads with the
existing 5 GiB safety reserve. Restore never writes over original source files
or changes their manifest paths.

## Opt-in cloud resume and catalogue access

Local-only processing stays unchanged. `--cloud-resume` explicitly permits a
processed partition with missing local outputs to skip reprocessing only when
its complete output receipts match the ETL identity and every live cloud
object's generation, size and provenance still match. Changed surviving local
files fail even when backups exist. This path performs metadata checks, not a
new full download: the prior verified SHA-256 plus immutable generation identifies
exact object bytes. Missing, incomplete, changed or inaccessible backups fail
without falling through to official archive download or raw ETL.

```sh
# Skip an already processed, cloud-backed month when working files are absent.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month YYYY-MM --cloud-resume --cloud-project citypulse-ai-511315 \
  --cloud-bucket citypulse-ai-511315-data

# Explicit future catalogue construction; restores only missing catalogue inputs.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --reconcile --cloud-resume --cloud-project citypulse-ai-511315 \
  --cloud-bucket citypulse-ai-511315-data --cloud-restore-root /path/to/cache
```

Catalogue recovery restores `validated_trips.parquet` and `rejected_trips.csv`
to a persistent source-month/revision cache. Existing exact UTC epoch precision,
identifiers, provenance, month boundaries, and duplicate auditing are unchanged.
Catalogue SQL views reference that cache: do not remove it while snapshots rely
on it. Source partition paths/hashes remain unchanged in the manifest. Rebuilding
24-month catalogue snapshots still requires separate disk budgeting; restoring
all required trip partitions is not a promise of constant-space global analysis.

The existing local DuckDB is derived from the validated/hourly Parquet tables
and the documented February reporting view. A future rebuild can reproduce those
logical tables/views and reconcile accepted/hourly counts; byte-identical DuckDB
files are not promised. This tool does not rebuild DuckDB or claim that its old
hash matches a regenerated database. Keep regenerated databases in separate
locations, rather than replacing the immutable source partition's database.

## Meaning of the retention dry run

Every candidate lists its role, bytes, required backups, recovery method, receipt
path where applicable, and blockers. Eligibility requires complete processed
outputs **and** original-source/audit/manifest backups, plus matching local hashes.
Metadata-only plans deliberately mark no file eligible: they have not checked
current bytes. Even hash-checked eligible files still require fresh cloud
inspection, recovery checks, dependency review, and explicit cleanup approval.
No report authorizes deletion, and monthly ingestion never deletes files.

For an approved future rolling workflow, retain one active month's ZIP, CSVs,
outputs and verification copies. Preserve immutable originals and processed
partitions remotely, permanent small control metadata locally, and restore
catalogue inputs only when needed. Refresh the monthly preflight because source
ZIP size, extraction size, spill and verification copies vary. The existing
largest-month conservative workspace check is about 12.69 GiB including reserve;
allow roughly 13–15 GiB plus extra space for backups/recovery/cache and retries.
Existing archive limits are not automatically increased.

## Actual verification and current blockers

The completed implementation passed **235 tests and 6 subtests** in 14.83 seconds,
including 33 new synthetic cases. Coverage includes exact original ZIP/CSV backups,
multiple CSV parts, interrupted/failed/colliding uploads, cloud-backed skipping,
missing/changed generations, changed local survivors, pinned restore and partial
file retention, disk preflight, and cross-month catalogue duplicate reconciliation.
Storage operations were mocked; no real backup, restore, official archive download
or historical ETL was run. Tests also assert that storage code does not unlink
local files and that the plan remains offline with unchanged control manifests.

The actual metadata-only dry run lists **52 candidates / 3,584,265,804 bytes
(3.34 GiB)**, including recorded verification-download copies. **Zero bytes are
eligible**: original-source and processing-manifest backups remain absent, and
February's complete historical output publication remains absent. February's
standalone hourly receipt is recognized as a partial backup and does not enable
whole-partition cleanup. Local hashes and fresh cloud checks were not performed
in this real-data dry run. Original ZIPs/CSVs, processed artifacts, live manifest,
and the unrelated EDA notebook remain intact.

After separately authorized full backups and verification, candidate categories
include January/March retained ZIPs, their extracted CSVs, validated output copies,
derived DuckDB databases, February's exact original CSVs/outputs, and recorded
verification-download copies. Keep permanent control metadata and source snapshots.
Unregistered older runs/test caches need separate review; no cleanup is implemented.

For `--cloud-resume`, unprocessed months still follow normal ingestion, which may
download archives and run ETL. Authorize those operations separately and confirm
that the requested partition is already processed when only verifying resume.
