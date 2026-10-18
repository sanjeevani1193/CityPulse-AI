# Repeatable bounded historical source-month cycle

Run one explicit missing source archive at a time in its own process:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -B \
  -m citypulse_ai.etl.monthly_cycle --month 2024-09 --september-budget
```

This wraps the existing `process_month`, processed/source GCS publishers and
`reconcile_temporary`; it does not replace validation, canonical selection or
the memory-bounded duplicate/daily aggregation algorithms. The default private
destination remains `gs://citypulse-ai-511315-data/`. No infrastructure is created.

Each invocation writes a new attempt directory under
`outputs/historical/monthly-cycles/YYYY-MM/`. Month-owned raw work is under
`outputs/monthly-work/YYYY-MM/`, with a checked ownership marker. Source ZIPs,
CSVs, processed partitions, monthly databases, source hashes, manifests,
receipts and durable reconciliation checkpoints are retained.

The serialized sequence is preflight, monthly ETL, processed publication,
original-source backup, global temporary reconciliation, cached idempotence
check and integrity checks. Publication precedes global reconciliation because
the existing temporary reconciler requires verified recovery receipts. Monthly
published counts remain source-archive counts; final local-start counts are
reported only after the global audit succeeds.

Preflight checks current free space, official archive metadata/ZIP directory,
local processed-input hashes, prior immutable receipt identities, live cloud
generations for required recovery inputs/manifests, bucket privacy/class and
read/list/create credentials. Existing registered work is verified and reused.
An unregistered cloud collision, missing input or changed publisher version
stops the command. It never silently restores unrelated datasets.

The general starting gate remains **28 GiB**. The explicit `--september-budget`
option is accepted only for `2024-09`. It verifies August's successful resource
report and computes the scaled observed peak and conservative envelope from
live September metadata. September's envelope is 9,765,264,862 bytes (9.095 GiB),
below the 10 GiB workspace ceiling. A **26 GiB** September gate is justified by
10 GiB workspace plus 16 GiB interruption margin. The measured August peak was
4,230,665,764 bytes; scaled for September it is approximately 4.275 GiB. This
exception does not change any future month or existing ETL default.

Runtime safeguards reproduce the August protections:

- DuckDB memory remains 512 MB on every connection; engine spill is capped at
  2 GiB and reconciliation receives its existing combined 4 GiB scratch budget.
- An OS-enforced 2 GiB **per-file** cap bounds staging databases, WALs and
  individual spill files. It is not a substitute for the aggregate workspace guard.
- Workspace and host free space are sampled every 0.25 seconds. Soft stops are
  9.5 GiB workspace or less than 16 GiB free; emergency exits are 10 GiB or less
  than 15.5 GiB free, protecting the required 15 GiB reserve with margins.
- Process RSS/host swap are sampled every 10 seconds. RSS above 2 GiB or swap
  allocation growth above 2 GiB stops the run. Samples are observations, not a
  continuous RAM guarantee or a filesystem quota.
- Actual publication volume is checked against a 3 GiB cap before uploads;
  create-only uploads use one application attempt with SDK upload retries off.
  Costs are budgeted before transfers, including verification downloads.

Full generation-pinned downloads still verify SHA-256 and Parquet row/schema
integrity. After publication and live generation/provenance verification, only
successful verification copies inside **this invocation's private temporary
directory** are deleted individually. Their paths, sizes and cloud identities
are recorded in `removed-verification-scratch.json`. Changed/unverified copies
and prior-attempt files are retained. Original sources and published artifacts
are never deleted by this runner. Existing ETL/reconciliation private disposable
scratch retains its normal lifecycle; durable checkpoints live outside it.

For interruption, inspect `stopped.json`, `resources.json` and existing ETL/cloud
manifest state. Correct the stated blocker, then explicitly rerun the same
single-month command. Verified ETL skips, published objects are checked by
receipt/generation without redundant uploads/downloads, and reconciliation
resumes its content-addressed checkpoints. A successful audit is read back with
SQL forbidden to verify cached idempotence. There are no automatic repeated
retries or bulk continuations.

After verification, `source-cleanup-candidates.json` lists exact original
ZIP/CSV paths, sizes, hashes, pinned cloud backing and restoration instructions.
It does not delete those files. Any original ZIP/CSV cleanup requires review of
that report and user authorization. All processed reconciliation inputs and persistent
canonical dependencies remain protected. The command has no retention/prune
mode. For the next month use a fresh explicit `--month` command **without** the
September option; insufficient resources stop before downloading.

Synthetic tests: `tests/test_monthly_cycle.py` plus existing historical,
publication/recovery, duplicate, daily aggregation and catalog regression tests.
No forecasting/model code is changed by this workflow.
