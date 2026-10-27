# Final 100-station XGBoost validation benchmark

This document records the training/validation milestone. The later authorized
[final held-out benchmark](final_held_out_benchmark.md) is complete; final-model
deployment integration remains pending. Statements about held-out status below
describe the earlier milestone.

Use the immutable station100 dataset and its training-selected station vocabulary.
This is a pooled direct, horizon-conditioned model: one row for each station,
NYC-local midnight issuance and lead 1–24. No recursive forecast updates or
within-day actuals are used. UTC ordering and 24 elapsed hours match the TFT,
including the local 23/25-hour DST days.

Training is January 2024–June 2025 (53,300 windows, 1,279,200 trusted target rows).
Validation is July–September 2025 (9,200 windows, 220,800 targets). The test split
is prohibited by the benchmark entry points. Opening the existing dataset verifies
all artifact hashes, including test artifact bytes, but test outcomes are never
loaded for training, features, validation, selection or metrics.

## Features and preprocessing

The fixed 48-column schema is defined in `station_xgboost.FEATURES`:

- Exact station IDs map to the existing frozen training-selected codes. Codes
  are native categorical features, not a numerical ordering of stations.
- Horizon step and the TFT's eleven known-future calendar/relative-position
  features are known at issuance. All local calendar values use New York time.
- Target-aligned elapsed-hour lags 24, 48, 72, 168, 192 and 336, each with an
  observation indicator. Every source bucket ends by issuance.
- Mean, population standard deviation, minimum, maximum and observed fraction
  over the preceding 6, 24, 168 and 336 elapsed hours. These summaries end at
  issuance, never at a future target hour.
- Past availability evidence: known fraction, latest known flag, and latest
  station status. Publication timestamps must not exceed issuance. Unknown status
  stays missing; it does not mean a closed station.

There are no fitted numerical scalers. Counts remain in original units. Unknown
count/status features use NaN plus explicit masks; certified accepted-record zeros
remain zero. The existing accepted_recorded_starts_v1 target and retrospective
bucket-end availability assumption are unchanged. Rejected-record diagnostics
are not future-known inputs. Targets are added only after feature construction.
The 336-hour history is summarized, rather than flattened into thousands of
columns; this is a documented model-specific representation of the same inputs.

## Bounded search, selection and fairness

Exactly three configurations are predefined, in order:

1. depth 4, squared-error objective;
2. depth 6, squared-error objective;
3. depth 6, absolute-error objective (median-oriented, like TFT q50).

Each uses CPU histogram trees, seed 42, learning rate 0.05, max_bin 64, two threads,
full row/column sampling, at most 300 boosting rounds and early stopping after
30 rounds without improved **clipped validation MAE**. Earliest configuration
wins exact ties. Quantile cut construction uses training data only; validation
QuantileDMatrix references the training matrix. The selected model is never
refitted on validation. Do not expand the grid after observing scores in this run.
The original preliminary citywide XGBoost model is not reused as trained weights.

For both raw and clipped (`max(0,prediction)`) outputs, report original-unit MAE,
RMSE and WAPE (%), and save prediction keys and actuals. Verify the validation
station/origin/target/lead digest and lag baselines against the finalized TFT.
TFT remains fixed at epoch 6. XGBoost is a point model; do not invent quantile
intervals or claim matched hyperparameter-search compute budgets. Validation
model selection has optimism; final conclusions require a later authorized,
once-only evaluation of the still-held-out test set with frozen specifications.

## Resources and recovery

Only one station-hour frame is cached while feature memmaps are written. The
training matrix is approximately 234 MiB float32; validation approximately 40 MiB,
plus labels and small key files. QuantileDMatrix bounds histogram input memory.
Retain the 2 GiB RSS guard, zero swap/host pressure checks, 15 GiB disk reserve,
512 MB DuckDB setting and four-hour server-side VM STOP. Stages run sequentially;
no automatic retry follows a safety/integrity failure. Three candidates are
trained sequentially; model selection and publication must finish within the
budget, with explicit time reserve.

The VM wrapper must verify Free Trial evidence and dataset hashes before launch,
use isolated CPU-only XGBoost dependencies, record source hashes and the base Git
revision, and arm an independent publication/shutdown finalizer. The CLI alone
does not establish these protections. No new VM, cloud service or paid activation.

Save selected UBJ trees (including categorical splits), raw-feature preprocessing
schema/vocabulary, search history/configuration, validation prediction Parquet,
comparison metrics and checksum receipts to the existing private GCS bucket.
Do not publish reproducible feature memmaps as a second copy of the dataset.
Fresh-process reload must reproduce saved validation predictions. Use a separate
idempotent MLflow run; log parameters, scores and cloud references, not datasets
or weights. Do not modify the fixed TFT checkpoint or any historical inputs.

## Completed benchmark (2026-10-26)

The predefined search completed on the existing Linux CPU VM. All three
candidates reached the 300-round cap; early stopping was not triggered.

| Depth | Objective | Rounds | Clipped validation MAE | Fit seconds |
| --- | --- | ---: | ---: | ---: |
| 4 | squared error | 300 | 4.2175 | 130.638 |
| **6** | **squared error** | **300** | **4.1028** | **214.535** |
| 6 | absolute error | 300 | 4.1371 | 427.533 |

The selected model uses the shared settings above, native categorical station
codes and no fitted numerical scaling. End-to-end feature preparation, search
and validation export took 1,024.867 seconds (17.08 minutes). The selected UBJ
model is 2,502,679 bytes. Recorded peak process RSS was 1.621 GiB, swap use was
zero, and the guarded workspace peaked near 526 MiB (including pre-existing
processed inputs). Free VM disk after publication was approximately 29.32 GiB.
These observations do not guarantee future workloads will fit the same resources.

| Forecast | Validation MAE | RMSE | WAPE (%) |
| --- | ---: | ---: | ---: |
| XGBoost raw | 4.1050 | 6.4226 | 30.024 |
| XGBoost clipped | **4.1028** | **6.4225** | **30.008** |
| TFT epoch 6 raw | 4.9205 | 7.5498 | 35.988 |
| TFT epoch 6 clipped | 4.7391 | 7.4922 | 34.662 |
| Lag-24 | 6.0211 | 9.5619 | 44.038 |
| Lag-168 | 5.2934 | 8.3116 | 38.716 |

All rows use the same **220,800 validation targets**. Unrounded clipped MAEs
are 4.102756140323995 (XGBoost) and 4.7391398695856335 (TFT), giving
`100 * (TFT - XGBoost) / TFT = 13.43%` lower validation MAE. Raw XGBoost negative
forecasts account for 0.5534% of targets. This is a validation-selected benchmark;
**final held-out testing and final-model deployment integration remain pending**.
The existing citywide API models are separate artifacts, not this station model.

The frozen dataset manifest SHA-256 is
`a6747adb2485276e2e7aa735dc24b1a10048e9b8526928c489a9472a532ba952`;
paired validation-key SHA-256 is
`48821491137ff4c4912493b5f1111f032539b9b16d2a6e88b3444af2df03dbb2`.
Both identity and lag-baseline equality were verified against the finalized TFT
report before fitting. The test split was not evaluated.

## Reproducibility and verified artifacts

The recorded runtime is Python 3.11.2, CPU-only `xgboost-cpu==3.2.0`,
NumPy 2.4.6 and SciPy 1.17.1 in an isolated environment. XGBoost 3.4.1 requires
a newer Python than this VM's Python 3.11; it was not installed there. Existing
ETL/TFT environments were preserved. The base Git revision was
`1b6063ce1b7ecfb52f99ba0f1df1e092890eca96`; the source overlay was recorded by
SHA-256 before training. The later Git commit saves that implementation; it does
not retroactively change the training provenance. The module SHA-256 used was
`6288574745d9c38e4f43f16d085cb9ff82d7b920e7ec3644e6a282b6f5079f68`.

For a separately authorized reproduction, supply verified dataset and finalized
TFT reference locations; use a new output directory. The following is a training
command, not a documentation-only check, and should not be run during a pause:

```bash
PYTHONPATH=src python scripts/train_station_xgboost.py \
  --dataset "$STATION_DATASET" --tft-reference "$TFT_EXPERIMENT_JSON" \
  --output-dir "$NEW_BENCHMARK_OUTPUT" --max-seconds 9000
```

The CLI requires the TFT reference. It checks dataset/window/key identities and
baseline scores before fitting. It does not provision a VM, restore inputs or
establish the external runtime/resource guards. Full synthetic training/reload
tests are available in `tests/test_station_xgboost.py`; for a no-training,
no-inference review use:

```bash
python -m pytest -q tests/test_station_xgboost.py -k 'not native_tiny_training'
```

On the existing macOS environment, use the documented OpenMP launch setting:
`DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs"`.
This does not alter Python environments. Those local checks used XGBoost 3.4.1;
the model's Linux serialization/reload checks used the recorded 3.2.0 runtime.

Artifacts are stored under:
`gs://citypulse-ai-511315-data/models/station-xgboost/station100-20261026/`.
The selected `model.ubj` SHA-256 is
`f7a64cc25e38893b0a566dab2957034fd81f8db6bdf2c712abace79015451b2a`.
Nine objects were verified, including model/preprocessing, experiment and
validation reports, keyed validation predictions, verification receipts,
`xgboost-recovery.zip` and the MLflow receipt. Generation-pinned receipts and
full checksum readback established integrity. The 3,622,070-byte recovery bundle
was restore-tested; a fresh native process reproduced all 147 saved probe
predictions exactly. No large artifacts are tracked by Git.

To recover, use the recorded object generations in `publication-result.json`,
verify SHA-256 before opening, extract the recovery ZIP into a new directory,
and verify its members and source-overlay hashes against the recorded base
revision. Restore the matching runtime and preprocessing vocabulary rather than
refitting. Keep the source dataset recoverable from its separate receipts.

MLflow experiment `CityPulse final 100-station forecasting benchmarks`, run
`4b552d74206f41d7aa2d22fc7fcd3a3d`, contains 46 measured metrics and recorded
configuration/source/dataset identities. Re-import preserved the same run and
added zero duplicate metrics. Reports/cloud references, not weights or datasets,
were logged. Open the [existing MLflow UI](local_mlflow_tracking.md) and select
this experiment; the TFT remains in its separate training experiment/run.
The VM was stopped after verification; the retained disk remains intact.
