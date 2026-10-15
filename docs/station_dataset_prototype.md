# Station-level dataset development prototype

This implements preparation contracts from [the TFT design](tft_station_forecasting_design.md), not a TFT model or final training dataset. The final project still uses January 2024–June 2025 for station selection/training, July–September 2025 validation and October–December 2025 testing. Those dates and the final design remain unchanged.

## Local source and scope

`features/station_history.py` reads the **completed temporary reconciliation audit and its existing daily station-hour Parquets**. Older persistent canonical snapshots do not cover the whole requested period, so they are not the source for this prototype. Nothing is reconciled again. It verifies audit/specification hashes, exact current manifest identity, offline publication receipts, local accepted/rejected inputs, checkpoint identity/hash/counts, disjoint daily partitions and source/adjacent-source membership. Read-only queries verify the requested local-month totals against the completed audit. This establishes accepted-record coverage for the selected source checkpoint, not complete physical demand, station operation or trusted zeros.

No source ZIP/CSV restoration, cloud SDK calls or historical processing occurs. Queries use an in-memory DuckDB connection at **512 MB**, two threads, with disk spill disabled. Changed/missing inputs or mismatched audit provenance fail closed; use a later complete audit explicitly when more months are ingested.

The development script selects stations using **January 1–July 1, 2024, NYC local time, end exclusive**. July counts never enter selection; July's source archive is legitimately needed to reconcile June boundary starts. A station needs at least **15 active local dates in every training month**, with no gap exceeding **14 local dates**, including leading/trailing gaps. Activity means at least one accepted start and does not certify operation. Eligible stations rank by accepted training starts descending, then exact station ID lexically ascending. Codes 1–100 follow that ranking; 0 remains reserved. Fewer than 100 eligible stations raises `InsufficientStations` with candidate statistics, without relaxed thresholds. This is a retrospective development activity screen, **not the final quality-approved station cohort**.

## Panel and quality contract

`features/station_dataset.py` builds explicitly bounded UTC grids for frozen IDs only. It preserves nullable integer `ride_count_accepted` separately from trusted `ride_count`, label masks, quality reasons/evidence, actual publication timestamps and independent nullable operating status. Local calendar columns include cyclical hour/weekday/month, weekend, DST, UTC offset and fold. UTC ordering preserves repeated local hours. A shared `time_idx` can be anchored at the full dataset start.

**Default quality is unknown, even for positive accepted counts.** A completed archive or reconciliation is not an hourly count-quality certificate. To mark a label trusted, pass an explicit per-station-hour quality table with `count_complete=True`, a nonempty reviewed `evidence_id`, and (where known) `coverage_status_available_at_utc`. The caller's evidence review must satisfy the original design's source membership/checksum, boundary, rejection/timestamp and duplicate requirements. The builder verifies structure, not the substantive truth of a human quality decision.

A certified complete hour with no sparse count becomes a trusted zero. An uncertified absent hour stays null; known closure alone cannot create a label. A trusted zero may coexist with unknown operating status. No recent-activity heuristic, interpolation, backward fill or inferred operating schedule is used. Explicit recorded zero counts can also be supplied, but still need quality approval. Operational count availability for such a zero requires its actual observation timestamp.

An optional status table carries `station_available` and `status_available_at_utc`. No future or late-published status enters an encoder. No station coordinates, borough, holiday or weather data is invented.

## Windows and preprocessing

Origins are **NYC calendar midnights**, converted to UTC. The encoder is 336 elapsed hours ending at issuance and the target is the next 24 elapsed hours. `origin_manifest` records warm-up and target-boundary exclusions; `iter_windows` yields samples lazily in origin/station-code order. Missing observations remain grid rows. No pre-January history padding is manufactured.

For batch size B, stacking sample arrays gives:

| Array | Shape |
|---|---|
| Static station code | `(B, 1)` |
| Encoder known calendar/relative position | `(B, 336, 11)` |
| Encoder observed count/mask/age/status/status mask | `(B, 336, 5)` |
| Decoder known calendar/relative position | `(B, 24, 11)` |
| Target and target mask, separately | `(B, 24, 1)` each |

Unknown encoder numbers are finite zero placeholders with false masks; real zero labels retain true masks. Age since the last accessible count is capped at 336. Unknown targets stay nullable in the panel and receive zero placeholders only in the separate target array, always paired with a false mask. Future demand, quality masks and realized operating status are never decoder features. `evaluation_keys()` returns station/origin UTC/target UTC/lead 1–24; overlapping DST forecast decisions remain distinct.

Replay defaults to the explicitly declared **retrospective assumption** that finalized quality-approved hourly counts become available at bucket end. Optional fixed latency masks the last L hours without moving origins or targets. Actual publication columns remain unknown if unavailable; replay assumptions never fabricate actual timestamps. Operational mode requires both count and coverage publication times at/before issuance. It supports one supplied version per hour, not a complete operational revision/as-of store.

`fit_count_scalers` fits per-station mean and population standard deviation from unique trusted training hours, with scale 1 for constant series. It fails if a station has no trusted training labels. Evaluation windows reject scalers fitted after their split start. Training windows must explicitly declare `split_role='training'`; scaler fitting must remain within that training split. Targets remain in original ride units. With no scalers, arrays are explicitly **raw-unit development diagnostics**, never presented as fitted model preprocessing.

## Small real-data probe

Run from the repository root, using existing dependencies and a **new** report directory:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/prototype_station_dataset.py \
  --audit outputs/historical/reconciliation-audits/33c5f2f2b402d79758d920a769edf41cb622cd812095d2932265725735c8e31f.json \
  --output-dir outputs/station-dataset-prototype/new-development-run
```

The script verifies January–July accepted-record totals, calculates January–June candidate statistics, and prepares two stations at March 10 and July 1 origins. Only 1,440 station-hour rows/four windows are inspected; no panel or tensors are persisted. Candidate statistics and a JSON report are capped together at 1 MiB, and existing output directories are never overwritten. The March probe checks actual spring calendar/count alignment; because the cohort is selected at June end, it **is not a causal March forecasting evaluation**. July is the development holdout structure.

The first measured run (`outputs/station-dataset-prototype/dev-20261015/report.json`) examined **2,333 stations**, found **2,069 eligible**, and selected 100 development candidates. Small probes used `6140.05` and `5905.14`. Process peak RSS was **203,440,128 bytes (~194 MiB)**; the largest in-memory panel was **204,582 bytes**. Generated reports totalled **134,220 bytes**. These are preparation/resource observations, not forecasting scores.

A second fresh-process run of the final implementation (`outputs/station-dataset-prototype/dev-20261015-verified/report.json`) reproduced identical candidate statistics and four probe results. It recorded the training dates January 1–June 30 and **18,795,989 accepted training starts**, with source/code hashes, runtime versions and timezone-rule provenance. Peak process RSS was **220,020,736 bytes (~210 MiB)** and its reports totalled **135,031 bytes**. Both small runs are retained; no prior artifact was overwritten.

**No real hourly quality certificates were supplied in these original prototype runs. Therefore their real trusted-label and input masks are all false, and those windows are not training-ready or eligible for the primary evaluation.** Positive accepted counts are retained for count/station/calendar diagnostics. No real scaler was fitted. Synthetic tests with explicit coverage certificates exercise nonempty encoders, positive/zero targets, independent status, leakage controls and fitted scaling. The subsequent [label evidence review](station_training_labels.md) adds an explicit, source-scoped positive-only development policy in separate reports; it does not certify zeros or establish an all-hour benchmark.

## Tests and remaining work

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_station_dataset.py tests/test_station_history.py tests/test_station_prototype_cli.py
```

Core tests are artifact-free. Source-adapter tests reuse existing tiny synthetic reconciliation fixtures with mocked cloud receipts; they do not reconcile real history or contact GCS. Existing environment dependencies are sufficient. Test spring/fall clocks synthetically; the finalized real subset only covers spring.

Before final TFT training: finish the 24-month sources plus January 2026 boundary, review count-source/rejection/DST quality and trusted zeros, create/freeze the final training-only cohort and scalers, implement complete operational versioning if desired, and freeze fair paired evaluation keys/coverage. TFT architecture, training, loss, model adapters and benchmark scoring remain outside this phase. Preserve all existing citywide experiments and historical snapshots.
