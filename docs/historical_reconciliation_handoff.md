# Historical reconciliation handoff

## Current verified result — 2026-10-14

The correlated-filter fix completed one controlled reconciliation at **512 MB**.
The verified duplicate checkpoint was reused: zero duplicate IDs across all eight
archives and 24,007,009 occurrences. All 245 daily partitions and global checks
passed: 23,993,387 accepted, 13,622 rejected, 5,876,513 station-hours, and zero
duplicate station-hour keys. May is boundary-reconciled at **4,133,356** for the
current source checkpoint; June remains provisional at **4,780,481** until July.

Temporary mode omits exclusion only for a fully verified empty duplicate result;
otherwise it uses an explicit anti join. Source/occurrence coverage, hashes,
identity and actual duplicate-table invariants are checked before selection.
Both optimized EXPLAIN plans eliminate the correlated DELIM/ride-ID grouping;
persistent catalog behavior is unchanged. 210 relevant regression tests passed.

The cached rerun returned identical audit bytes/results without opening DuckDB.
All original inputs, manifests, snapshots and ten persistent view bindings remain
unchanged. Peak RSS was 174.17 MiB, no engine spill was observed, retained
checkpoints were 17.79 MiB, and free disk stayed above 29.89 GiB. Swap allocation
stayed at 3 GiB. No new GCS operations, ingestion, downloads, memory increases,
commits or pushes occurred.

Evidence: `outputs/historical/june-2024-duplicate-filter-fix/report.md` and the
content-addressed final audit under `outputs/historical/reconciliation-audits/`.
This milestone preserves the verified implementation. Future source sets require a fresh global
audit and resource preflight; large nonzero duplicate tables can still exhaust
512 MB. Existing canonical local-file dependencies and checkpoint retention
remain separate follow-up work. The failure notes below are historical and are
superseded by this verified result.

## Update — 2026-10-14

Daily UTC station-hour aggregation and checksummed restart checkpoints are now
implemented and tested. The single controlled full attempt still failed at 512 MB
while discovering canonical UTC-day subtotals, before any daily file completed.
Its compile-only EXPLAIN reveals `LEFT_DELIM_JOIN` and an additional ride-ID
`HASH_GROUP_BY` from the correlated canonical `NOT EXISTS` filter. This is the
remaining unbounded operator, not the daily aggregation itself.

The independently retained duplicate audit now verifies **0 duplicate IDs and
0 cross-archive duplicate IDs across 24,007,009 occurrences in eight archives**.
Per-source accepted/rejected totals also survived. May is still not finalized;
June remains provisional; no completed canonical audit was published. All inputs,
manifests, snapshots and ten persistent view bindings remain unchanged.

Next: test an explicit anti join (or equivalent exact non-correlated lookup) in
**temporary mode**, compare it with original rejection semantics and confirm its
plan avoids ride-ID delim grouping. Review other canonical scans and report
queries too. Preserve persistent catalog behavior. Do not repeat full recovery
until this fix, tests and fresh disk preflight pass; today's one attempt is used.
The new checkpoints can reuse the duplicate phase after verifying input hashes.
Keep 512 MB and at least 15 GiB free, with a monitored margin.

198 relevant tests passed; the final focused rerun passed 30 tests. Observed peak
process RSS was 1.044 GiB, peak scratch plus checkpoints 1.749 GiB, and minimum
free space 28.559 GiB. During this session allocated OS swap grew from 2 to 3 GiB;
the exact cause of yesterday's free-space change cannot be proven retrospectively.
Full evidence is in `outputs/historical/june-2024-bounded-completion/`, including
`failed-query-plan.txt`. No GCS access, data deletion, retry or memory increase.

The following records the original 2026-10-13 checkpoint; restart details above
and in `temporary_reconciliation.md` supersede its no-intermediate-checkpoint note.

## Completed

June 2024 source ingestion and validation completed, and its artifacts have
verified GCS backups. Do not reingest, download or restore them. All eight
registered source months already have the local Parquet and rejection CSV inputs
needed for recovery. Existing source records, manifests and persistent catalogs
remain intact.

Temporary reconciliation now projects normalized ride IDs and archive months in
50,000-row batches, routes them into stable MD5 partitions, and groups full IDs
within each partition. Hashes route records only; they do not define duplicates.
All accepted and rejected rows across all registered archives remain in scope.
The partitioned duplicate table was successfully constructed during June recovery.
Persistent catalog construction retains its existing behavior.

The previous recovery regression run passed 131 tests. This WIP checkpoint is
not a completed reconciliation milestone.

## Remaining failure

`create_reconciliation_relations()` in `history_catalog.py` still executes:

```sql
CREATE TABLE hourly_rides AS
SELECT start_month, start_station_id, date_trunc('hour', started_at) AS hour_start,
       count(*)::BIGINT AS ride_count
FROM canonical_trips
GROUP BY start_month, start_station_id, hour_start;
```

This global aggregation/materialization exhausted DuckDB's **512 MB** memory
limit, even with external spill. The limit was not increased. Peak process RSS
was 860.1 MiB; the DuckDB managed-memory limit is not a whole-process RAM cap.
Peak observed scratch was 2.17 GiB and minimum observed free disk was 20.22 GiB.
Owned scratch was removed on exception; original artifacts remain untouched.

No completed reconciliation audit was published. The duplicate count was not
captured before failure and must not be reported as zero. May 2024 is not globally
finalized because station-hour uniqueness and source/count reconciliation checks
did not finish. June's previously verified source-local count of 4,780,481 remains
provisional pending complete global reconciliation and July's boundary input.

Local recovery evidence is retained under
`outputs/historical/june-2024-reconciliation-recovery/` (`run.log`, `resources.json`,
`integrity.json`, `gcs-live-check.json`, `outcome.json`, and `report.md`). The original
June checkpoint is under `outputs/historical/june-2024-checkpoint/`. These are
gitignored local evidence, not files included in this commit.

## Resume safely

1. Inspect this WIP branch and the retained failure trace. Run the small synthetic
   regression tests before changing implementation:

   ```sh
   .venv/bin/python -m pytest -q \
     tests/test_partitioned_duplicates.py tests/test_temporary_reconciliation.py
   git diff --check
   ```

2. First implement bounded station-hour aggregation **only in temporary mode**.
   Consider grouping disjoint NYC-local start months into owned temporary,
   disk-backed intermediates. Aggregate across all source archives for each
   month; source-month partitioning alone would mishandle boundary overlaps.
   Preserve exact station strings, elapsed UTC hour keys, global duplicate
   exclusion, total-count reconciliation and station-hour uniqueness checks.
   Keep existing persistent snapshots and behavior unchanged. Test against the
   original SQL using small boundary, duplicate and DST fixtures.

3. Review the scratch budget: the existing 4 GiB ceiling currently allocates
   2 GiB to ID projections and 2 GiB to spill. Any new intermediates must fit a
   reviewed total budget. Keep DuckDB at 512 MB and at least 12 GiB host free;
   monitor the full run and interrupt above that reserve. Do not delete data
   or increase memory to make recovery fit.

4. Only after the fix and preflight pass, resume standalone
   `--reconcile-temporary` using the existing validated inputs. There is no
   resumable partial SQL checkpoint: the duplicate construction must be repeated.
   Do not run ingestion, build a persistent snapshot, or upload unverified counts.
   Finalize May only after every global check passes; June remains provisional
   until July's source boundary is available. No July ingestion is part of this
   checkpoint.

Exact ID partitioning improves the duplicate phase, but the complete workflow
is not yet verified safe for June or the remaining historical months at 512 MB.
