# Station label evidence and development policy

The prototype originally masked every real label because no hourly quality
decision accompanied the sparse canonical counts. A processed archive, an active
station, and a positive aggregate are different kinds of evidence. None alone
establishes a complete observed station-hour or operating schedule.

`features/station_labels.py` now adjudicates **source-scoped positive labels**.
This is a deliberately narrow development policy, not an all-hour demand
benchmark certificate. It leaves the final [TFT design](tft_station_forecasting_design.md)
and its zero-acceptance requirements intact.

## Available evidence and its limits

`StationHistory` verifies the completed temporary reconciliation audit, exact
manifest identity, independent duplicate checkpoint, daily hourly partitions and
their checksums, source totals, local processed-input hashes and offline,
generation-pinned publication receipts. The label adapter checks the processed
inputs again before and after its queries. It does not contact GCS or claim a
new live cloud verification.

January–July 2024 accepted-start months have their adjacent source archives.
The current checkpoint contains ten archives, including August for July's
boundary contribution. The label review reads **all ten** validated inputs for
the requested station/hour keys, including any contribution whose source month
differs from its local start month. It reads all rejection audits too. It does
not rerun source validation, duplicate construction or canonical reconciliation.

For relevant/contributing archives, recorded ZIP member inventories must match
CSV hashes and schema-audit membership. Input = accepted + rejected,
accepted-count conservation and timestamp-epoch checks must pass. The existing
completed global audit must prove zero duplicate IDs; nonzero duplicates fail
closed in this new adapter until an exclusion-aware recount is implemented.
The historical reconciliation implementation itself retains its duplicate rules.

The older February 2025 import has original CSV hashes and verified offline
backup receipts, but no ZIP-member/schema inventory. It is explicitly recorded
as such, never upgraded to complete archive evidence. It contributes no selected
starts or rejected-start hours to these 2024 slices. Such a source blocks a
review if it is required by the requested months or contributes selected starts
or rejection uncertainty. Missing/changed local artifacts or mismatched canonical
counts also fail closed.

This establishes **internal completeness of the enumerated archived accepted
records**, conditional on the pinned source set. It cannot prove the publisher
included every physical ride, rule out omitted upstream records or future late
archives, or reconstruct publication times. “Finalized” local-month counts are
the existing accepted-record boundary convention, not an unlimited lateness
guarantee. Positive trip records prove activity at an instant, not full-hour
operation. No verified operating-status history is established by these inputs.

## Exact policy: `archived-accepted-positive-v1`

The target is starts accepted under the frozen historical validation rules in
the pinned reconciled source set. It is not latent unmet demand or all physical
starts. The policy's positive labels remain conditional on that source scope.

1. Require verified completed source/checkpoint provenance and finalized local
   months, including adjacent boundary sources.
2. Recount selected station/hour starts directly from immutable validated rides,
   grouped also by source archive and exact original CSV path. Compare **every
   sparse positive hour** exactly against canonical hourly counts. Never edit
   counts to conceal a mismatch. Exact IDs and UTC ordering are preserved.
3. Parse every rejected start with the existing ETL timestamp parser. Any
   rejected start blocks **all selected stations** in its possible UTC hour,
   conservatively avoiding guessed station attribution. A naive fall-back start
   blocks both folds. Invalid, nonexistent or otherwise unlocalizable starts
   block the entire requested slice; there is no end-time imputation. This
   intentionally overmasks even rejections with an apparently usable station ID.
4. Admit a positive label only when the recount agrees and its hour is not
   blocked. Pass a per-key quality decision to the unchanged panel builder.
5. **Certify no zero labels.** A missing row remains null/mask 0, even in an
   otherwise clean hour. Zero certification requires the design's separate
   reviewed interval-enumeration decision; this adapter does not manufacture one.
6. Keep station availability null and its known mask false without independent
   evidence. Unknown operation does not invalidate a separately certified count
   zero, and known closure cannot certify a zero from an incomplete source.

This asymmetric positive-only policy is a conservative development subset. It
does not mean a positive row proves physical-source completeness while a zero
does not. Positive *archived* counts have direct row lineage; the additional
all-hour coverage/zero policy remains unresolved. The mathematical absence of
accepted rows in a fixed file set can be checked, but is not promoted to a
complete historical zero-demand observation here.

Quality evidence contains the versioned policy, exact interval and station IDs,
manifest/audit/source identities and hashes, member records, source accepted and
rejected totals, offline backup revisions, blocked UTC hours, reason counts,
and a digest of the per-hour/source/member lineage. Each admitted key has a
deterministic evidence hash linked to that review. The compressed decision and
lineage CSVs allow inspection without materializing a large panel. The builder's
`count_complete`/`certified_complete` fields refer to the **declared target source
scope**, not physical-feed or operational completeness.

## Replay and leakage

Counts and coverage are retrospective: eventually approved hourly observations
are assumed accessible at bucket end. Actual publication timestamps remain
unknown. This is not an operational midnight backtest. The existing optional
1/3-hour latency and operational timestamp checks remain available.

Only pre-origin encoder observations enter input features. Target counts,
quality masks, rejection evidence and realized status never become decoder
features. The real probe changes future labels and masks and verifies all input
arrays remain identical. Frozen January–June candidate membership is reused;
earlier probes are label/calendar diagnostics, not causal forecasts from a cohort
that would already have been known then. July information never reselects IDs.

## Small real-data verification

Run offline in the existing environment, choosing a new report directory:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/validate_station_labels.py \
  --audit outputs/historical/reconciliation-audits/33c5f2f2b402d79758d920a769edf41cb622cd812095d2932265725735c8e31f.json \
  --cohort-report outputs/station-dataset-prototype/dev-20261015-verified/report.json \
  --output-dir outputs/station-label-review/new-review
```

Two previously selected IDs (`6140.05`, `5905.14`) are probed at January 15,
February 1, March 10, April 1, May 1, June 1 and July 1 NYC midnights. Each slice
has 336 encoder + 24 target elapsed hours per station. The nonoverlapping slices
total **5,040 station-hours**, not the full January–July panel.

The measured review admits **729 positive labels (14.464%)**, **zero certified
zeros**, and masks **4,311 hours (85.536%)**. Mask reasons are 3,906 rejected-start
hours with uncertain station assignment and 405 otherwise absent rows lacking a
zero certificate. Those reason categories are disjoint; some rejected-start
hours also have no sparse station row. Operating status remains unknown for
all 5,040 hours. All sparse positive counts, including withheld ones, match the
validated-row recount exactly.

There are 14 structurally valid, deterministic windows with some encoder and
target labels. None has all 24 trusted targets. July 1 has 58/66 trusted encoder
hours and 5/6 trusted target hours for the two IDs respectively. Encoder observed
arrays are `(336, 5)`, encoder calendar arrays `(336, 11)`, decoder calendar
arrays `(24, 11)` and target/mask arrays `(24, 1)`. Spring-forward omits local
02:00 yet still returns 24 elapsed targets; fall-back is covered synthetically.
No count scaler was fitted and no model was trained.

The final report is `outputs/station-label-review/dev-20261015-verified/report.json`,
with compressed per-hour decisions/lineage and `verification.json`. Across the
probe, **4,388 sparse positive station-hours** match the original validated-row
counts exactly; only 729 pass the additional conservative quality filter. The
grid has 652 absent sparse hours, including 247 already masked by rejected-start
uncertainty. A fresh-process repeat reproduces decisions, evidence and keyed
lineage records; parallel SQL may serialize lineage rows in a different order.
Peak review-process RSS was **136,527,872 bytes (~130.2 MiB)**, the largest panel
181,857 bytes, and final reports including verification total 498,380 bytes.
Free space was 26,499,457,024 bytes before and 26,501,771,264 after the probe;
this small host-wide increase is not attributed to the process. Queries retain
the 512 MB DuckDB ceiling with spill disabled.

**114 relevant tests passed**, including 16 new label-policy tests, existing
station/window tests and historical preparation/reconciliation regressions.
Synthetic checks cover rejection uncertainty, invalid/nonexistent/ambiguous
timestamps, nanosecond boundary starts, exact suffixed IDs, empty stations,
canonical mismatch, changed inputs, missing source evidence, nonzero duplicates,
legacy noncontributing source metadata, deterministic decisions and future-target
invariance. Syntax and whitespace checks passed. All 20 accepted/rejected input
hashes and the manifest still match, and all 10 persistent view bindings resolve.
The full ML suite was not run.

## Readiness and remaining requirements

A **small masked-loss TFT implementation smoke test** can now consume real
positive observations, after fitting diagnostic scalers only on unique trusted
training hours. It must skip wholly unlabeled samples and use explicit masks
without NaN multiplication. The current subset is strongly selected toward
hours without unassigned rejected trips and positive demand. Its losses cannot
be interpreted as all-hour demand accuracy or a fair benchmark. Strict complete
24-target evaluation currently has zero eligible windows in this probe.

For useful all-hour training, document and approve the archive-enumeration and
lateness assumptions for the accepted-record target, establish reviewed UTC
coverage intervals, and adopt an explicit conservative treatment of unassigned
rejected starts. Upstream completeness evidence and station-ID corrections or
an approved exclusion policy are needed to improve that coverage; raw trip
activity alone cannot supply them. Do not silently relax masks based on desired
coverage. Historical station-status intervals would be needed to label operating
availability reliably; a live status query cannot recover January–July history.
Status data is independent of, and not a substitute for, count enumeration.

Final training also needs the remaining historical sources plus January 2026
boundary, the final January 2024–June 2025 training-only cohort/scalers, audited
zeros and DST/rejection handling, full coverage reporting and shared evaluation
keys. No final two-year training readiness is claimed.
