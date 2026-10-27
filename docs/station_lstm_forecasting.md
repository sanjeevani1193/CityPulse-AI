# Final station LSTM: fixed development specification

This is a new pooled TensorFlow/Keras model, not 100 independent fits and not
restored citywide weights. The original `forecasting/lstm.py`, its 48-hour model
and checkpoints remain unchanged. The station adapter retains its CPU LSTM(32),
Adam/MSE approach and validation-loss checkpoint selection. It adds the frozen
station vocabulary, a longer encoder, explicit masks and known-future calendar.

## Configuration fixed before real-data training

- One configuration, no hyperparameter search. Seed 42, CPU TensorFlow 2.21.0,
  Keras 3.15.1; deterministic operations, two intra-op and one inter-op threads.
- Historical input `(batch, 336, 16)`: the same 11 historical calendar channels
  and five observed/mask channels used by TFT. Counts use the existing per-station
  training-only scalers. The five channels distinguish scaled counts, observation
  masks, age since last observation, operating status and known-status masks.
- Station codes feed an 8-dimensional embedding (101 vocabulary slots including
  unused slot 0). Repeat this embedding over the encoder and concatenate with
  history, then encode using **LSTM(32)**. There is no attention or TFT machinery.
- Repeat the encoder context and embedding for 24 leads, concatenate the existing
  `(batch,24,11)` known-future calendar, then shared Dense(16, ReLU) and Dense(1).
  Output `(batch,24)` is in station-standardized units; inverse-transform before
  reporting ride-count metrics. **8,953 parameters** for 100 stations.
- Adam 0.001, global gradient norm clip 1, batch size 64, no dropout.
- At most **20 complete epochs**. Select the lowest validation masked MSE in
  station-standardized units; exact ties keep the earlier epoch. Early stopping
  after **four epochs** without improvement exceeding **0.0001** relative to the
  patience reference. This objective differs from the MLP's clipped-MAE selection.
- One reproducible station-block shuffle and within-station window shuffle per
  epoch (seed 42 + epoch); each training window is visited exactly once. Keep
  only one station panel and one sequence batch in memory. No full tensor cache.
- A training-only probe updates a disposable fresh model. Discard its weights and
  reset the seed before the actual fit. Gate full training on measured throughput,
  memory, disk and time, without consulting validation scores to change settings.
- A 10,000-second training-process budget reserves 900 seconds before another
  epoch/verification; estimate the next complete epoch conservatively. Runtime
  interruption is partial training, not convergence. Save completed epochs only.

## Data and leakage contract

Reuse the checksum-verified final station100 dataset, its exact 100 training-only
selected stations, training-fitted preprocessing and approved window index:
53,300 train windows / 1,279,200 targets (Jan 2024–Jun 2025) and 9,200 validation
windows / 220,800 targets (Jul–Sep 2025). The train/validation reader never opens
or even hashes October–December test panels. It reads index metadata only.
Verify the frozen pre-test TFT dataset hash and paired validation-key digest
before optimization. Recheck paired lag-24/168 scores after validation.

The encoder ends strictly before NYC-local midnight issuance. Calendar channels
are genuinely known for the next 24 **elapsed UTC hours**, including DST.
Availability evidence is gated by publication time at each origin. No target
count, rejected-record diagnostic or future availability enters inputs. The
`accepted_recorded_starts_v1` target and retrospective bucket-end availability
assumption are unchanged; this is not an operational publication-latency replay.
Unknown values carry zero placeholders only alongside independent masks; missing
labels are never asserted to be genuine zero demand. Only trusted labels enter
masked loss, normalized by trusted target count. Empty batches skip updates.

Compared with TFT this model consumes the same raw historical and calendar
channels, but has a single summary encoder and no variable selection, gating,
quantiles or attention. Compared with XGBoost/MLP it receives the complete
historical sequence rather than explicit lag/rolling features. Information,
objectives and compute are therefore documented rather than claimed identical.
Prior disclosed test scores do not select these settings. Do not evaluate the
held-out split as part of this development task.

## Recovery and operation

A `.keras` checkpoint saves the model and Adam state after every completed epoch.
Its JSON sidecar binds model SHA-256, dataset/scaler/source/config identities,
full history, patience state, best global checkpoint and optimizer iterations.
Resume only matching state. With no dropout and epoch-derived sample ordering,
restart requires no unsaved random sample iterator. Rerunning a fresh fit into
an existing directory fails safely. Never inherit original citywide weights.

```bash
PYTHONPATH=src python scripts/train_station_lstm.py \
  --dataset "$STATION_DATASET" --output-dir "$NEW_PROBE" --probe
PYTHONPATH=src python scripts/train_station_lstm.py \
  --dataset "$STATION_DATASET" --output-dir "$NEW_LSTM_RUN" \
  --reference "$FROZEN_TFT_VALIDATION_REPORT"
# Only after verifying completed-epoch sidecar/hash and remaining runtime:
PYTHONPATH=src python scripts/train_station_lstm.py \
  --dataset "$STATION_DATASET" --output-dir "$EXISTING_LSTM_RUN" \
  --reference "$FROZEN_TFT_VALIDATION_REPORT" --resume epoch-001.keras
```

These commands do not establish cloud safeguards on their own. Use the existing
VM, current Free Trial confirmation, four-hour server-side STOP, 2 GiB process
RSS, zero-swap/host-pressure and minimum 15 GiB disk safeguards. Never increase
DuckDB's 512 MB limit or alter ETL/model environments. Install TensorFlow only in
a separate reviewed Linux environment. Stop on a failed probe or safety check.
Publish completed checkpoints, config, preprocessing, histories and validation
predictions using SHA-256/generation verification and restore-test recovery.
MLflow is a separate idempotent validation-only run; do not modify previous runs.

Synthetic regression tests: `tests/test_station_lstm.py`. They cover shape and
station identity, frozen preprocessing, masked/empty loss, finite optimizer
updates, model/optimizer reload parity, canonical-window equality, spring DST,
future-target isolation, test-panel exclusion, paired keys and interrupted resume.

## Measured completed run (27 October 2026)

All 20 complete epochs finished. Training stopped at the fixed epoch cap; early
stopping did not fire. Recovery found 16 completed epochs, followed by epochs
17–19 and a verified process-budget pause. A final continuation completed only
epoch 20. No completed epoch was repeated. The last continuation kept its server
deadline earlier than the original four-hour batch deadline.

The selected checkpoint is **`epoch-018.keras`**, with validation
station-standardized masked MSE **0.45499836**. Selection
retains the original loss rule, rather than choosing the lowest ride-count MAE.
The frozen model is ready for a separately authorized one-time test evaluation.
**No October–December 2025 test targets were opened or evaluated in this task.**
This is a validation-only follow-up; earlier models’ test results had already
been disclosed and were not used to choose this configuration.

| Model / postprocessing | Validation MAE | RMSE | WAPE (%) |
|---|---:|---:|---:|
| LSTM_raw | 4.5676 | 7.0628 | 33.407 |
| LSTM_clipped | 4.5272 | 7.0553 | 33.112 |
| MLP_clipped | 4.1213 | 6.4504 | 30.143 |
| XGBoost_clipped | 4.1028 | 6.4225 | 30.008 |
| TFT_clipped | 4.7391 | 7.4922 | 34.662 |
| lag_24 | 6.0211 | 9.5619 | 44.038 |
| lag_168 | 5.2934 | 8.3116 | 38.716 |

All comparisons use the same 9,200 forecast windows and **220,800** eligible
validation targets. Nonnegative clipping is `max(0, prediction)` after inverse
scaling. Raw and clipped scores remain separate.

| Epoch | Training masked MSE | Validation masked MSE | Clipped MAE |
|---:|---:|---:|---:|
| 1 | 0.519937 | 0.719553 | 6.0580 |
| 2 | 0.430720 | 0.650123 | 5.4992 |
| 3 | 0.395604 | 0.587294 | 5.2903 |
| 4 | 0.371149 | 0.599090 | 5.1904 |
| 5 | 0.355902 | 0.547581 | 5.1871 |
| 6 | 0.343349 | 0.528710 | 4.8727 |
| 7 | 0.331492 | 0.528223 | 4.8101 |
| 8 | 0.323599 | 0.511489 | 4.8094 |
| 9 | 0.316318 | 0.504189 | 4.8424 |
| 10 | 0.310808 | 0.500955 | 4.7389 |
| 11 | 0.306441 | 0.495333 | 4.7857 |
| 12 | 0.301624 | 0.479660 | 4.7420 |
| 13 | 0.297996 | 0.479692 | 4.6526 |
| 14 | 0.294364 | 0.489084 | 4.5831 |
| 15 | 0.292679 | 0.474831 | 4.7669 |
| 16 | 0.288283 | 0.471555 | 4.5864 |
| 17 | 0.286051 | 0.470212 | 4.5670 |
| 18 | 0.283398 | 0.454998 | 4.5272 |
| 19 | 0.281313 | 0.476947 | 4.5337 |
| 20 | 0.279353 | 0.465703 | 4.5454 |

Completed-epoch training plus validation time: 182.07 minutes.
Training and selected-validation exports across segments: 203.85 minutes.
Peak measured process RSS: 1.296 GiB;
peak monitored workspace: 251.03 MiB
(includes protected prior source files). Zero swap in every recorded sample;
no resource guard triggered. Free VM disk after publication: 27.332 GiB.
TensorFlow CPU 2.21.0 / Keras 3.15.1 ran in a separate Python 3.11 Linux
environment. Historical ETL environments were preserved. The isolated runtime
occupied approximately 1.38 GiB; initial reviewed wheel downloads were 300 MiB.

All 20 checkpoint/sidecar pairs have immutable SHA-256 and generation receipts.
The recovery ZIP passed full checksum verification, member verification and a
fresh-process exact 64-window prediction/optimizer reload check on the VM.
Model weights, panels and prediction files were not downloaded to the Mac.
MLflow run: `d19a135eed554398b2525b327955014b` in `CityPulse final 100-station forecasting benchmarks`.
An idempotent second import added zero metric steps and kept the same run ID.
The run export, selected-checkpoint record, measured report and verification
receipts are under the existing private GCS prefix:
`models/station-lstm/station100-20261027/finalization/`.
Selected checkpoint SHA-256: `81dde6d7faf480e7e870fb2234a0208828f1f6cb857b95a18c589913d4c531ec`.
Use the generation-pinned finalization manifest and recovery receipts to restore
the frozen source overlay, preprocessing, model and environment. `verify_model.py`
in recovery verifies the saved validation probe in a fresh compatible process;
it does not authorize a test evaluation.

Verification: 70 relevant native Linux synthetic/data/label tests passed
(24 Keras/NumPy deprecation warnings, no unexpected skips). Fresh implementation
syntax and whitespace checks passed. No full project test suite is claimed.
The original citywide LSTM, TFT, station MLP and XGBoost artifacts are preserved.
At the training milestone, no serving integration or held-out LSTM evaluation
had been performed. The subsequent authorized test phase is recorded below.

## Subsequent frozen-model follow-up

The authorized one-time MLP/LSTM test extension is complete. See the
[six-method follow-up benchmark](six_model_followup_benchmark.md) for all six
methods, raw/clipped scores, shared-key checks, disclosure limitations and recovery.
The training/validation history and original four-method measurements above are
preserved. Final station-level FastAPI/Docker integration remains pending.
