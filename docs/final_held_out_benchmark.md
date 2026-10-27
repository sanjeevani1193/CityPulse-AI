# Final held-out 100-station forecasting benchmark

This evaluation freezes the existing six-epoch TFT (`epoch-006.pt`) and the
selected CPU XGBoost 3.2.0 model (300 trees, depth 6, learning rate 0.05, seed 42).
It does not train, tune, refit preprocessing or alter their validation scores.
The evaluation driver is `evaluation/station_benchmark.py`; the training drivers
continue to reject test evaluation.

## Frozen contract before scoring

Use the immutable `accepted_recorded_starts_v1` station100 dataset, exactly the
100 stations selected from January 2024–June 2025 training information, and its
training-only fitted preprocessing. Both frozen experiment reports must match
the dataset identity and explicitly record no earlier test evaluation. All
existing model, preprocessing and dataset hashes must match generation-pinned
GCS receipts before the first test panel is read. This establishes evidence from
saved records and inspected code, rather than an unverifiable claim about every
possible action outside the recorded workflow.

The held-out interval is October 1, 2025 at NYC-local midnight through January 1,
2026 at NYC-local midnight, end exclusive. All four approaches share 92 origins
per station, 9,200 station-origin windows and **220,800 targets**. The contract
is computed from frozen station IDs and the calendar, independent of outcomes.
Saved eligibility flags must match it completely; inconsistencies fail before
scoring, rather than silently dropping targets for one model.

On November 2 the 24-elapsed-hour horizon contains both 01:00 hours with distinct
UTC offsets and ends with the 22:00 EST bucket. The 23:00 EST bucket
(`2025-11-03T04:00:00Z`) precedes the next midnight issuance and is not forecast.
Thus **100 station-hours are intentionally unscored** out of 220,900 panel rows.
No duplicate keys, 25-hour horizons, calendar-day padding or retrospective
extra forecast is introduced. The final scored bucket starts at
`2026-01-01T04:00:00Z`, corresponding to December 31, 23:00 EST.

## Causal inputs and common scoring

Each origin uses only the preceding 336 elapsed hours. A previous test-period
observation can become a later origin's historical input after its bucket ends;
this is a rolling-origin evaluation, not one forecast issued for the entire
quarter. The existing retrospective assumption is retained: eventual certified
accepted recorded counts are available at bucket end. This is not a real-time
publication-latency claim. Operating availability remains independently unknown
without status evidence; future quality diagnostics are not model inputs.

TFT receives the existing static station code, encoder observations/masks and
encoder/decoder calendar features; decoder actuals are never passed to the model.
Its training scalers are loaded without fitting. The original XGBoost 48-column
schema, raw-count lags and past summaries are reused unchanged. Station status
is gated by its publication time. Lag-24 and lag-168 counts must be observed and
available by issuance. No weather or future observed demand is introduced.

One station frame is cached at a time. TFT inference creates one shared keyed
prediction file and a small memory-mapped causal feature matrix. XGBoost runs
in its original separate CPU environment on those exact rows, preventing
population drift. Each phase writes a create-only start/completion marker;
partial failures require inspection and never trigger automatic re-evaluation.
The two models are inferred once, with raw and clipped scores calculated from
those saved forecasts. Recovery and metric verification do not rerun inference.

The predeclared postprocessing is `max(0, prediction)` in original count units,
applied to the TFT's three quantiles independently and to XGBoost's point output.
Report raw and clipped MAE/RMSE/WAPE separately; baselines are already nonnegative.
WAPE is aggregate absolute error divided by total actual starts, expressed as a
percentage. TFT additionally reports each quantile's pinball loss, their mean,
and inclusive empirical q10–q90 coverage (nominal 80%). Zero accepted starts are
valid labels; they do not imply zero physical demand or an operating station.

## Reproducibility, tracking and recovery

`evaluation-plan.json` fixes keys, identity, periods, preprocessing and clipping
before panel access. `test-predictions.parquet` contains every shared key, actual,
raw/clipped model forecast and both baselines. `experiment.json` contains metrics;
`validation-versus-test.csv` compares against frozen validation measurements,
with `test - validation` gaps. `integrity.json` records population, dates, unique
keys, clipping and preservation checks. Source hashes and the recorded Git base
identify any evaluation overlay without falsely claiming it is already committed.

Run small synthetic contract/scoring tests without real data or cloud access:

```bash
python -m pytest -q tests/test_station_benchmark.py
```

The full evaluation is an explicitly authorized, once-only cloud operation,
not a command to run during ordinary CI. Use the existing isolated TFT and
XGBoost runtimes. Preserve the 512 MB DuckDB limit, 2 GiB process RSS guard,
zero swap/host-pressure protection, minimum 15 GiB disk reserve, bounded
workspace and four-hour server-side VM STOP. Stop after any failed prerequisite
or safeguard; do not retry inference automatically or delete protected artifacts.

Publish reports, per-target predictions, code/configuration and a recovery ZIP
to the existing GCS bucket. The existing create-only publisher verifies exact
object generations, full SHA-256 readback and Parquet schema/row counts. Restore
the ZIP into a new directory and verify member hashes before using it. To recover
scores, read the saved predictions/reports; do not run another test inference.

Log results to a separate idempotent final-test MLflow run keyed by frozen model,
dataset, prediction and test-key identities. Historical training/validation runs
and measurements remain unchanged. MLflow stores small reports and GCS references,
not duplicated model weights or dataset panels. Final FastAPI/Docker integration
is a separate, pending phase and is not performed by this benchmark.

## Verified results — 2026-10-26

Scoring completed once on the common 220,800 targets. No fitting or tuning was
performed. Both models and protected dataset/preprocessing identities remained
unchanged. The existing VM stopped automatically and its retained disk is ready.

| Forecast | Test MAE | Test RMSE | Test WAPE (%) |
| --- | ---: | ---: | ---: |
| XGBoost raw | 3.3744 | 5.2640 | 37.212 |
| **XGBoost clipped** | **3.3663** | **5.2630** | **37.123** |
| TFT raw | 4.0330 | 6.0420 | 44.474 |
| TFT clipped | 3.7696 | 5.9449 | 41.570 |
| Lag-24 | 4.6071 | 7.4208 | 50.806 |
| Lag-168 | 4.7665 | 7.7200 | 52.563 |

XGBoost has **10.70% lower clipped test MAE** than TFT on these paired keys.
This is a descriptive comparison, not a statistical significance claim or a
matched-search-budget architecture ablation. No further search followed testing.

| Forecast | Validation MAE → test MAE | Validation RMSE → test RMSE | Validation WAPE → test WAPE (%) |
| --- | --- | --- | --- |
| XGBoost clipped | 4.1028 → 3.3663 | 6.4225 → 5.2630 | 30.008 → 37.123 |
| TFT clipped | 4.7391 → 3.7696 | 7.4922 → 5.9449 | 34.662 → 41.570 |
| Lag-24 | 6.0211 → 4.6071 | 9.5619 → 7.4208 | 44.038 → 50.806 |
| Lag-168 | 5.2934 → 4.7665 | 8.3116 → 7.7200 | 38.716 → 52.563 |

Absolute errors fall on the test quarter, while normalized errors rise:
XGBoost WAPE increases by 7.116 percentage points and TFT by 6.908 points.
Therefore lower test MAE alone does not demonstrate improved relative accuracy
across quarters. The CSV preserves unrounded raw/clipped scores and all gaps.
The calendar, target definition and retrospective availability limitations still
apply; these scores do not measure lost demand or operational data latency.

TFT uncertainty (original ride-count units):

| Mode | q10 pinball | q50 pinball | q90 pinball | Mean pinball | q10–q90 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 0.794913 | 2.016476 | 1.069434 | 1.293607 | 76.556% |
| Clipped | 0.718650 | 1.884786 | 1.020873 | 1.208103 | 79.495% |

Nominal coverage is 80%; raw negative q50 forecasts represent 15.910% of targets.
Clipping was fixed before scoring and was not selected using these test results.

The full held-out target-key digest is
`658573cfdd2aa24de99a29f82478ed008368fdf33af91b702e288a3171867cc7`.
Artifacts were published under
`gs://citypulse-ai-511315-data/benchmarks/station100/final-test-20261026/`.
`test-predictions.parquet` is 9,852,214 bytes, generation `1793048835875310`,
SHA-256 `ccbd2ca6989f85437078bed249a36507c60dfc627d9e2a0139d424e42902cc30`.
The 9,795,088-byte recovery ZIP is generation `1793048837343577`, SHA-256
`da34a4f1db614af515697dad025d6f3b5d0b9c5563ff8e18165a753e53a6aa6d`.
All six primary result/recovery objects passed full readback verification on
the VM; generations, sizes and SHA metadata were independently checked again
without downloading predictions to the Mac. The publication report is separately
checksum-verified. Recovery ZIP members were hash-verified after extraction.

MLflow uses a new final-test run, `0202f941aa2b4eaa90bc465cd9a7cef5`, in
`CityPulse final 100-station held-out benchmark`. It contains 38 measured test
metrics. Repeated metadata import adds no run or metric steps; the original TFT
and XGBoost training/validation run parameters, metrics and histories are preserved.

Verification: **31 local focused tests passed**, with four training/checkpoint
tests deliberately deselected; **five native Linux synthetic contract tests
passed before scoring**. Subsequent local preflight failure tests use synthetic
metadata only. No new dependency installation was needed. Observed scoring peak
RSS was 727.94 MiB, with zero swap in sampled reports. TFT phase time was 246.018
seconds and XGBoost phase time 6.005 seconds (including loading/exports).
Scoring workspace peak was about 281.26 MiB under the reused monitor, which also
counts pre-existing protected source-month files; this is not all new disk use.
Free VM disk was approximately 29.32 GiB before and 29.23 GiB after publication.
The VM was running for about 9.89 minutes. These are measured scoring-phase
resources, not a universal memory bound or a claim about all host processes.

### Remaining deployment work

Final FastAPI/Docker integration remains pending. Existing citywide adapters do
not implement this station-level 48-feature XGBoost schema or the final TFT's
station/encoder/quantile contract. A separate phase must add trusted artifact
loading, exact station vocabulary and masks, causal 336-hour feature preparation,
frozen clipping, response metadata and saved-reference parity tests, followed by
container resource checks. Operational count-publication latency also needs an
explicit policy before claiming a real-time forecasting service. None of that
integration was started here.
