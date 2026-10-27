# Final 100-station MLP: frozen development specification

This is a **new model initialized from seed 42**, not a reuse of the preliminary
citywide weights or scalers. The original `forecasting/mlp.py` and its artifacts
remain unchanged. `forecasting/station_mlp.py` reuses its `DemandMLP` hidden layers
and the station XGBoost causal feature builder. No LSTM, TFT or XGBoost fitting
is part of this run. Previously disclosed final test scores do not determine
this configuration; this is a validation-only follow-up comparison.

## Specification fixed before training

- One pooled network: **194 → 32 ReLU → 16 ReLU → 1**, **6,785 parameters**.
- Inputs: 100 one-hot station identities, 47 numerical features (the XGBoost
  schema except native categorical station code), and 47 independent missingness
  flags. Expand only the current minibatch, not the entire training matrix.
- Numerical features retain the existing horizon, calendar, lags 24/48/72/168/
  192/336, past 6/24/168/336-hour summaries and historical availability evidence.
- Fit population feature means/standard deviations only on finite **training
  target rows**, in bounded chunks. An all-missing or constant column uses scale
  1; an all-missing column uses mean 0. Missing numerical values become zero
  after standardization and carry a true missing flag. Observed zeros remain
  real counts; no zero target is invented. Unknown operation is not a closure.
- Target normalization: one population mean/standard deviation over training
  target rows only, with scale 1 for a constant target. Predictions are restored
  to original ride-count units before metrics/clipping.
- Adam, learning rate **0.001**, MSE on standardized targets, gradient norm clip
  **1**, batch size **2,048**, two CPU threads and deterministic PyTorch algorithms.
- At most **30 complete epochs**, one fixed configuration, no search. Each epoch
  visits every training row once, using the permutation seeded with `42 + epoch`.
- Select the completed checkpoint with the lowest **clipped validation MAE**;
  exact ties retain the earlier epoch. Stop after five epochs without an
  improvement of at least **0.0001** against the patience reference.
- A separate 9,000-second training-process budget reserves time for verification
  and publication. Do not start an epoch without the estimated epoch time plus
  600 seconds remaining. Discard an incomplete epoch from selection; retain
  previous completed checkpoints. A runtime stop is not convergence.
- Save each completed model/optimizer/RNG/history checkpoint and publish with
  generation/SHA-256 verification through the existing guarded VM wrapper.

Training MSE is the sample-weighted average of minibatch losses while weights
change; it is not a post-epoch frozen-state full-training metric. Validation
runs without gradients and is streamed in bounded inference batches. The output
head predicts one target row at a time; 24 horizon-conditioned rows form each
24-elapsed-hour forecast, with no recursive actual-demand updates.

## Frozen dataset, causal inputs and validation population

Use the existing 100-station vocabulary and January 2024–June 2025 training
windows: **53,300 windows / 1,279,200 targets**. Validate only on July–September
2025: **9,200 windows / 220,800 targets**, with exactly the frozen TFT validation
key digest and lag-24/lag-168 baseline scores checked before optimizer creation.
Fit no statistics on validation, and do not refit after selecting the checkpoint.

Each origin is NYC-local midnight. All lag/summary demand comes from the prior
336 elapsed hours; genuinely future-known calendar values describe the next
24 elapsed hours. Historical station status is additionally gated by publication
time. UTC ordering and the existing DST contract are retained. The label policy
remains `accepted_recorded_starts_v1`, with the existing retrospective assumption
that certified counts become available at bucket end, not actual publication time.

The development reader verifies the trusted full manifest identity and the
required training/validation panel, preprocessing and index receipts. It **never
opens the test-quarter panels, even to hash their bytes**. Reading the frozen
window index exposes keys/eligibility metadata, not test target values. A small
optional reader-factory parameter in XGBoost's materializer permits reuse without
changing its default behavior or any feature calculation. Both training APIs
continue to reject the test split. No October–December 2025 targets or final test
prediction files are used by this development pipeline.

Raw and `max(0, prediction)` validation MAE, RMSE and pooled percentage WAPE are
reported separately. Compare against the **frozen validation** TFT/XGBoost and
lag baselines, not their revealed test outcomes. MLP is a point forecaster;
no quantile uncertainty performance is claimed. Different objectives and search
budgets mean this is not a controlled architecture-only ablation.

## Reproduction, resource safety and recovery

The existing CPU PyTorch runtime is reused without altering model/ETL environments.
The CLI alone does not establish cloud safeguards:

```bash
PYTHONPATH=src python scripts/train_station_mlp.py \
  --dataset "$STATION_DATASET" --tft-reference "$FROZEN_TFT_VALIDATION_REPORT" \
  --output-dir "$NEW_MLP_RUN" --max-seconds 9000
```

Run only after current Free Trial confirmation, resource checks and the existing
four-hour server-side VM STOP. Retain the 2 GiB process RSS guard, zero swap,
normal host pressure, minimum 15 GiB free disk, bounded workspace and 512 MB
DuckDB limit. Stop and preserve diagnostics on safety/integrity failure; do not
automatically retry or delete protected data.

Raw feature arrays are float32 memmaps; one station frame and one encoded
minibatch are held at a time. The scalar validation prediction vector and keyed
metadata are small. Preserve preprocessing, configuration, epoch history, best
checkpoint and immutable receipts in the existing private GCS bucket. Restore a
verified recovery bundle and compare fresh-process predictions on saved
validation probes. Recovery must not require repeating training or opening test
outcomes. MLflow uses a separate idempotent station-MLP validation run, leaving
all earlier training/validation/test measurements unchanged.

Tests: `tests/test_station_mlp.py` covers training-only moments, missing flags,
station encoding, 100-station architecture, finite gradients, no test-panel
verification, synthetic complete-epoch training, selected-checkpoint reload and
fail-closed paired-reference checks. Existing citywide MLP regressions remain
separate. No final test evaluation or serving integration is part of this phase.

## Verified completed run — October 27, 2026

The single predefined configuration completed **15 full epochs** and stopped by
its five-epoch patience rule. Every epoch visited **53,300 training windows /
1,279,200 target rows**. Epoch **10** had the lowest clipped validation MAE and
is frozen as **`epoch-010.pt`**. There was no architecture change, search, refit
on validation or test-target access. Training MSE decreased from 0.320953 at
epoch 1 to 0.236935 at epoch 15; checkpoint selection still uses validation MAE.

| Forecast | Validation MAE | RMSE | WAPE (%) |
| --- | ---: | ---: | ---: |
| MLP raw | 4.1411 | 6.4527 | 30.288 |
| **MLP clipped** | **4.1213** | **6.4504** | **30.143** |
| XGBoost clipped, frozen comparator | 4.1028 | 6.4225 | 30.008 |
| TFT clipped, frozen comparator | 4.7391 | 7.4922 | 34.662 |
| Lag-24 | 6.0211 | 9.5619 | 44.038 |
| Lag-168 | 5.2934 | 8.3116 | 38.716 |

All rows share **220,800 validation targets** with paired-key SHA-256
`48821491137ff4c4912493b5f1111f032539b9b16d2a6e88b3444af2df03dbb2`.
MLP clipped MAE is 0.45% higher than XGBoost and 13.04% lower than TFT on
validation. These are descriptive comparisons, not significance claims or an
equal-compute architecture ablation. Final MLP test scores do not exist.

Measured model-development time was **405.026 seconds**, including feature
preparation, preprocessing, completed-epoch publication callbacks and selected
validation export. The completed train/validation epoch timers totalled
**189.347 seconds**, averaging **12.623 seconds** (range 7.825–15.174); checkpoint
publication occurs outside individual epoch timers. The guarded training and
verification phase took **421.192 seconds**. Peak process RSS was **1,084.43 MiB**
(about 1.06 GiB), with zero swap in sampled reports and no triggered safeguard.
The monitor's workspace peak was about 529 MiB, including pre-existing protected
source-month files. VM free disk after publication was **28.84 GiB**. These are
observations for this run, not universal runtime/memory guarantees.

The initial native preflight stopped before training because the minimal TFT
environment lacked scikit-learn. A reviewed dependency dry run then stopped on
an additional transitive requirement; no packages or epochs were installed/run
in that step. The complete dependency closure was subsequently installed only
in a new isolated MLP environment. ETL and TFT package-version snapshots remained
unchanged. The MLP environment reuses existing CPU PyTorch **2.14.1+cpu**, NumPy
**2.4.6** and SciPy **1.17.1**, and adds scikit-learn **1.9.1**, joblib **1.6.0**,
narwhals **2.27.1**, threadpoolctl **3.7.0** and cloudpickle **3.1.2**. The reviewed
package downloads totalled **10,153,338 bytes**; the earlier dry run also retrieved
wheel metadata/files. No large data or model downloads went to the Mac.
Full environment freeze and environment-verification evidence are in recovery.

**32 local focused regression tests passed**, including original MLP, station
XGBoost and final-dataset tests. After the final probe-check refinement, its five
MLP-specific tests passed again. **25 native Linux tests passed** in the corrected
MLP environment before real training. Early failed collections are retained in
the run's setup logs and are not counted as successful tests.

All **15 completed epoch checkpoints** were uploaded and checksum/generation-
verified. Selected checkpoint:

`gs://citypulse-ai-511315-data/models/station-mlp/station100-20261027/epoch-010.pt`

- Generation: `1793101068103877`; size: **97,752 bytes**.
- SHA-256: `8bf098d0f3eca06f3fb569931433233c6fb221aff48017ff8d34b5acf414f63f`.

Preprocessing, frozen pretraining specification, configuration, history,
validation predictions/comparison and recovery are in the same GCS prefix.
`station-mlp-recovery.zip` is **4,429,179 bytes**, generation `1793101154309172`,
SHA-256 `b98f0b6c3d7a29a4258b9165dd68822c5c001a585abc8735e56d011b0a9506b4`.
The selected checkpoint and restored recovery bundle each passed fresh-process
exact parity on **100 saved validation probe rows**. Probe results also agree
with full validation batching within the declared float32 tolerance
(`rtol=1e-6`, `atol=1e-5`). Original per-target validation metrics reproduce
exactly using the same batching after checkpoint reload.

MLflow has a separate station-MLP run, **`3f26e822801949efaa5c72099132ed69`**, in
`CityPulse final 100-station forecasting benchmarks`. It records epoch histories,
selected validation scores and comparison/provenance references without copying
weights or datasets. Idempotent import preserves that run and metric steps;
existing citywide, TFT, XGBoost and final-test runs are not modified.

At the training milestone, the existing VM was stopped and its protected 100 GiB
disk retained. The original citywide MLP, TFT/XGBoost checkpoints and historical
data remained intact. The MLP was frozen for a separately authorized one-time
follow-up test, with held-out targets untouched during development. Earlier test
results had already been disclosed, so a later extension could not be described
as a newly untouched blind test. That training task did not start LSTM or serving
integration, or commit/push; subsequent milestones are recorded below.

## Subsequent frozen-model follow-up

The authorized one-time MLP/LSTM test extension is complete. See the
[six-method follow-up benchmark](six_model_followup_benchmark.md) for all six
methods, raw/clipped scores, shared-key checks, disclosure limitations and recovery.
The training/validation history and original four-method measurements above are
preserved. Final station-level FastAPI/Docker integration remains pending.
