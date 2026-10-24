# Accepted recorded starts: versioned station-hour labels

Policy identifier: **`accepted_recorded_starts_v1`**.

CSV-backed legacy imports can supply an explicit, hash-pinned
[CSV-only provenance supplement](csv_source_provenance.md) for original-file
membership, schemas and row coverage. It does not claim a ZIP inventory or
change this target policy. Missing or invalid supplements still fail closed.

For a frozen, verified canonical source set R, station ID s and elapsed UTC hour
`[t, t + 1h)`, the target is the number of **canonical accepted recorded rides**
in R whose exact start-station ID is s and whose UTC start lies in that interval.
The ETL acceptance, timestamp and duplicate definitions are unchanged.

This is a recorded-start forecasting target. It does not measure all physical
customer demand, attempted/lost rentals, station operation, or upstream publisher
completeness. A zero means **no accepted records in the certified source scope**,
not that no customer wanted a bicycle or that the station was closed.

## Certification requirements

`features/accepted_recorded_labels.py` implements `AcceptedRecordedStarts` as a
separate policy. The original `archived-accepted-positive-v1` code and its reports
remain intact. Certification requires:

- A complete, hash-verified temporary reconciliation audit and exact current
  manifest identity; offline generation-pinned backup receipts, verified local
  processed inputs and immutable daily station-hour checkpoints.
- Source accepted/rejected conservation and timestamp-epoch checks, a completed
  global duplicate audit and canonical count/unique-key invariants.
- Finalized requested NYC-local start months, including the required adjacent
  source archives and their member/hash/schema inventories. Provisional months,
  missing archives or missing evidence are errors, never zero labels.
- An explicitly supplied frozen station-code mapping. Each exact ID must have
  accepted training-period membership, verified using only the declared cohort
  training interval. This checks membership, not a new station ranking or an
  operating schedule. Unseen IDs cannot receive synthetic zero histories.
- A bounded expected station × elapsed-UTC-hour grid with valid keys.
- Exact comparison of every sparse positive canonical count against selected
  validated-row enumeration across **every source in the pinned set**, including
  cross-month contributions. Dense comparison includes absent keys: only after
  the complete selected enumeration finds no accepted rows is a zero certified.

The shared positive-review routine provides the existing fail-closed enumeration
checks; its rejection-based mask decisions are deliberately not adopted. A
nonzero duplicate checkpoint remains unsupported by that recount adapter and
fails closed until exclusion-aware recount support is added. The underlying
historical pipeline's duplicate rules are preserved. Legacy source inventories
that cannot support a required/contributing period also fail closed.

Hashes of accepted and rejected inputs are checked before/after enumeration and
again at certification completion. Missing/changed inputs, checkpoints, manifests
or count disagreement stop the operation. No partial failed review is returned
as a fully certified panel.

Each review records the policy, exact UTC bounds and station mapping, training
membership period/counts, source/manifest/audit identities, source inventory and
processed hashes, local-month totals, duplicate result and positive lineage
digest. Each station/hour decision links its count and positive/zero reason to
the review through a deterministic content hash. Changed source sets require
new certificates; a policy-version string alone is not coverage evidence.

Finalized boundaries follow the project's existing accepted-record convention.
They do not prove that a publisher can never add a late record in a future
archive. A later approved source set is a new target revision, not an unnoticed
rewrite of this certificate. Naive source timestamps retain the existing
America/New_York interpretation and DST rejection assumptions.

## Rejection diagnostics and operating availability

Rejected rides are excluded under the original ETL rules. They cannot alter the
count of accepted records. Consequently, an unknown start station or a rejected
timestamp does **not** mask an otherwise certified accepted-record label.

Rejections remain independently auditable. A separate citywide hourly table
contains exact localized rejected-record counts and reason counts, possible
assignments for ambiguous DST folds, and source archive membership. Exact and
possible counts are distinct. The review summary retains unique overlapping
record counts and unlocalizable record counts/reasons across the source set;
unlocalizable records are never falsely assigned to the requested interval.
These diagnostics are not station-level allocations and must not be summed once
per station. They are **not model inputs or future-known covariates**.

Station operating status remains null with `availability_known_mask=False` when
independent historical status evidence is absent. Count observation masks and
operation masks remain separate: a source-scoped zero is observable even when
operation is unknown. No closure, continuous operation or station identity alias
is inferred from silence, names or nearby activity.

## Dataset and TFT compatibility

The policy emits the existing panel builder's explicit per-key
`count_complete`/`evidence_id` quality table. Sparse `ride_count_accepted` remains
nullable for lineage; `ride_count` contains the certified positive or zero label.
The decision table carries `policy_version` and the exact certification reason.
The builder's generic `certified_complete` status is qualified by that evidence:
it means this **accepted-record source scope**, not physical-feed completeness.

No TFT architecture, feature order or tensor shape changes. Windows retain the
336-hour encoder and 24-hour elapsed horizon, with calendar features derived in
America/New_York from UTC ordering. Encoder count masks now include certified
zeros. Operation masks remain false. Targets and label-quality diagnostics never
become decoder features.

The existing **retrospective replay assumption** is preserved: eventual approved
counts are assumed available at bucket end. Real publication timestamps remain
unknown. Operational mode still requires actual publication timestamps and does
not become usable merely because this retrospective panel is fully labeled.
Existing model checkpoints and their old positive-only preprocessing remain
historical diagnostics; they are not silently updated. A later training phase
must fit new scalers on unique permitted training labels, including certified
zeros, without July/evaluation fitting.

## Small development verification

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/validate_accepted_recorded_labels.py \
  --previous-review outputs/station-label-review/dev-20261015-verified/report.json \
  --cohort-report outputs/station-dataset-prototype/dev-20261015-verified/report.json \
  --output-dir outputs/accepted-recorded-labels/new-review
```

The script uses the same two frozen development IDs and seven original origins:
January 15, February 1, March 10, April 1, May 1, June 1 and July 1, 2024.
Only seven 720-row slices are built; the complete station dataset is not
materialized. Earlier dates remain retrospective label/window diagnostics
because the January–June cohort was known only at June end. July remains the
development holdout structure, not a source for cohort selection or fitting.

Reports go to a **new** gitignored folder and contain certificates, compressed
station/hour decisions, positive source lineage and separate rejection
diagnostics. Existing reports are never overwritten. DuckDB queries retain
512 MB and disable spill; report storage is capped at 2 MiB. This is read-only
historical input verification, not a reconciliation rerun or new ingestion.

The measured run is saved under
`outputs/accepted-recorded-labels/dev-20261018/`. It reproduced the original
5,040 station-hour keys with **4,388 positive labels, 652 source-scoped zeros
and zero masked labels (100% accepted-record coverage)**, compared with 729
labels under the retained old policy. All 4,388 positive keys agree exactly
with canonical counts and validated-row lineage. The dense grid contains
76,133 accepted starts in total; all 652 zeros have no accepted row in the
complete selected enumeration.

All **336/336 forecast-target positions** are now observed: 288 positive and
48 zero targets. All 14 station-origin windows have 336 observable encoder
hours and 24 observable target hours under retrospective replay. This is
accepted-record window eligibility, not proof of model skill or an operational
backtest. Operating status remains unknown for all 5,040 station-hours.

Separate diagnostics retain **6,643 unique overlapping rejected records**:
6,635 missing start-station IDs and 8 invalid IDs. These counts are citywide
per reviewed interval, not multiplied by the two stations. No rejected ride
was assigned to a station, made valid or added to the target.

The probe took **87.821 seconds** including source/checkpoint hash verification
and bounded filtered enumerations. Peak process RSS was **136,577,024 bytes
(about 130.3 MiB)**; the four initial diagnostic files total 612,837 bytes.
No count scalers were fitted and no model training took place. The broader
regression run passed **149 tests in 32.92 seconds**, including 17 new policy/CLI
tests and the existing TFT/station/historical synthetic regressions. No full ML
suite or real-data reconciliation was run.

Focused artifact-free tests:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_accepted_recorded_labels.py \
  tests/test_accepted_recorded_probe.py tests/test_station_history.py \
  tests/test_station_dataset.py tests/test_station_labels.py
```

These cover positive/zero certificates, rejected unknown stations, unlocalizable
rejected times, missing archives/inventories and unfinalized boundaries,
provenance/checkpoint failures, unseen IDs, changed inputs/cohort identities,
count disagreement, nonzero duplicate refusal, deterministic decisions, grid
budgets, UTC/DST windows and future-target invariance. The CLI regression checks
counts by exact station/hour keys even when station-code order differs from
lexical panel order.

## Remaining requirements

Fully observed accepted-record labels do not establish forecast skill or full
physical-demand observation. Final work still needs the remaining historical
sources and January 2026 boundary archive, the final training-only 100-station
cohort, versioned full-period coverage and timestamp review, training-only
preprocessing, chronological training/validation and fair paired evaluation.
Independent operating-status history is necessary only if reliable operating
availability labels are required. No model training is part of this phase.
