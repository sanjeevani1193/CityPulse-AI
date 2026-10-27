# Six-method station100 follow-up benchmark

This phase adds the frozen station MLP and LSTM to the saved XGBoost, TFT,
lag-24 and lag-168 test benchmark. **The earlier four-method test results were
already disclosed before MLP/LSTM development. This is a follow-up extension,
not a fully blind six-model experiment.** No test score may change architecture,
preprocessing, checkpoint selection or clipping.

## Frozen evaluation contract

- Dataset: `accepted_recorded_starts_v1`, the existing 100 training-selected
  stations and training-only scalers. No historical regeneration or fitting.
- MLP: `epoch-010.pt`, SHA-256
  `8bf098d0f3eca06f3fb569931433233c6fb221aff48017ff8d34b5acf414f63f`.
  Keep its validation-clipped-MAE selection, 194 → 32 ReLU → 16 ReLU → 1,
  one-hot station codes, 47 standardized numeric features and 47 missing flags.
- LSTM: `epoch-018.keras`, SHA-256
  `81dde6d7faf480e7e870fb2234a0208828f1f6cb857b95a18c589913d4c531ec`.
  Keep validation station-standardized MSE selection across 20 completed epochs,
  8,953 parameters, station embedding8, LSTM32 and future-calendar Dense16 head.
  Reuse the exact training-only per-station count scalers and sequence masks.
- History: 336 elapsed UTC hours strictly before NYC-local midnight issuance.
  Forecast: the next 24 elapsed hours. Future demand and retrospective quality
  diagnostics never enter predictors. Earlier test observations may enter a later
  origin's history after their buckets end, under the existing retrospective
  bucket-end count-availability assumption; this is not a latency replay.
- Train: Jan 2024–Jun 2025; validation: Jul–Sep 2025; test: Oct–Dec 2025.
  Exactly 9,200 common test windows / 220,800 targets, digest
  `658573cfdd2aa24de99a29f82478ed008368fdf33af91b702e288a3171867cc7`.
- November 2 contains both 01:00 hours with distinct UTC offsets. Its horizon
  ends at 22:00 EST; `2025-11-03T04:00:00Z` (23:00 EST) is unscored. Thus 100
  station-hours are unscored, without changing any method's target population.
- Postprocessing: inverse scaling first, then `max(0, prediction)`. Preserve
  raw and clipped MAE, RMSE and aggregate WAPE separately. Certified accepted
  zeros do not establish zero physical demand or known station availability.

The evaluation module pins model, preprocessing, original validation report and
saved benchmark hashes. It checks cohort/scaler fitting periods and calendar keys
before decoding test panels, saves `evaluation-plan.json`, and requires matching
source hashes. Live GCS generations and local SHA-256 must match before inference.
The operator publishes the plan before either test inference phase starts.

## Execution and recovery

`evaluation/station_followup.py` runs frameworks in separate existing CPU
interpreters. Training entry points still reject test requests. One station is
read at a time. The MLP reuses `station_xgboost.feature_block` and
`station_mlp.predict`; the LSTM reuses `DevelopmentData.batches` and calls the
frozen model with `training=False`. Neither optimizer nor scaler is fitted.

Each model writes a create-only start marker before test reads and a completion
marker bound to prediction/key hashes. A partial inference must be inspected;
never automatically retry it. Summarization reads saved forecasts only. It
requires identical keys, actuals and clipping, copies the four existing method
measurements, and exports a six-row clipped leaderboard, raw/clipped metrics,
validation-versus-test gaps and combined per-target predictions.

```bash
# Synthetic checks; each command uses its existing appropriate environment.
python -m pytest -q tests/test_station_followup.py tests/test_station_followup_mlp.py tests/test_station_benchmark.py
python -m pytest -q tests/test_station_followup_lstm.py

# Authorized cloud operation only. ARTIFACTS_JSON is a trusted operator mapping
# of the frozen artifact names to restored local paths, never an API input.
PYTHONPATH=src python scripts/evaluate_station_followup.py \
  --dataset "$STATION_DATASET" --artifacts-json "$ARTIFACTS_JSON" \
  --output-dir "$NEW_EVALUATION" --phase freeze
# After publishing/verifying the frozen plan and compatibility probes:
# --phase mlp in the existing PyTorch interpreter
# --phase lstm in the isolated TensorFlow interpreter
# --phase summarize in the existing data interpreter
```

These commands do not provision safeguards. The operator must use current
user-confirmed Free Trial evidence, the existing VM/disk/IAM, a four-hour
server-side STOP, 2 GiB process RSS, zero swap/host-pressure checks, a bounded
workspace and at least 15 GiB disk reserve. DuckDB stays at 512 MB. Stop the VM
on success or failure. No installation, billing upgrade or new infrastructure
is required for this phase.

Publish predictions, immutable identity/configuration reports, source archive
and a recovery ZIP to the existing private bucket using generation-pinned
SHA-256 readback and Parquet row/schema checks. Recovery verifies ZIP members
and recomputes scores from saved forecasts, never reruns the models. Checkpoints
remain in their already verified model prefixes rather than being duplicated.
Log a separate idempotent follow-up MLflow run; preserve historical runs and
record that earlier test results were disclosed. Final serving integration
remains a separate pending phase.

## Verified results — 27 October 2026

| Method | Test MAE | RMSE | WAPE (%) |
|---|---:|---:|---:|
| XGBoost | 3.3663 | 5.2630 | 37.123 |
| MLP epoch 10 | 3.4015 | 5.3244 | 37.511 |
| TFT epoch 6 | 3.7696 | 5.9449 | 41.570 |
| LSTM epoch 18 | 3.8010 | 5.8195 | 41.917 |
| Lag-24 | 4.6071 | 7.4208 | 50.806 |
| Lag-168 | 4.7665 | 7.7200 | 52.563 |

## Raw and clipped scores

| Method | MAE | RMSE | WAPE (%) |
|---|---:|---:|---:|
| TFT_raw | 4.0330 | 6.0420 | 44.474 |
| TFT_clipped | 3.7696 | 5.9449 | 41.570 |
| XGBoost_raw | 3.3744 | 5.2640 | 37.212 |
| XGBoost_clipped | 3.3663 | 5.2630 | 37.123 |
| lag24 | 4.6071 | 7.4208 | 50.806 |
| lag168 | 4.7665 | 7.7200 | 52.563 |
| MLP_raw | 3.4397 | 5.3291 | 37.933 |
| MLP_clipped | 3.4015 | 5.3244 | 37.511 |
| LSTM_raw | 3.8641 | 5.8336 | 42.612 |
| LSTM_clipped | 3.8010 | 5.8195 | 41.917 |

## Validation versus test

| Method | MAE validation → test | RMSE validation → test | WAPE validation → test (%) |
|---|---:|---:|---:|
| XGBoost | 4.1028 → 3.3663 | 6.4225 → 5.2630 | 30.0075 → 37.1231 |
| MLP epoch 10 | 4.1213 → 3.4015 | 6.4504 → 5.3244 | 30.1429 → 37.5112 |
| TFT epoch 6 | 4.7391 → 3.7696 | 7.4922 → 5.9449 | 34.6620 → 41.5699 |
| LSTM epoch 18 | 4.5272 → 3.8010 | 7.0553 → 5.8195 | 33.1119 → 41.9171 |
| Lag-24 | 6.0211 → 4.6071 | 9.5619 → 7.4208 | 44.0380 → 50.8063 |
| Lag-168 | 5.2934 → 4.7665 | 8.3116 → 7.7200 | 38.7158 → 52.5634 |

MAE/RMSE and WAPE describe different aspects of quarter-to-quarter generalization. Demand levels change seasonally; lower absolute error alone does not establish lower relative error.

## Integrity and provenance

Exactly 100 training-selected stations, 92 midnight origins per station, 9,200 station-origin windows and 220,800 identical eligible targets.
NYC-local test period: October 1 2025 midnight through January 1 2026 midnight, end exclusive.
The encoder uses 336 elapsed UTC hours and the forecast horizon uses 24. November 2 contains both local 01:00 offsets; its 23:00 EST bucket is unscored (100 station-hours).
First scored bucket: `2025-10-01T04:00:00Z`; last: `2026-01-01T04:00:00Z`.
Key digest: `658573cfdd2aa24de99a29f82478ed008368fdf33af91b702e288a3171867cc7`.
All model inputs use only counts available before each issuance and known calendar; train-only cohort, vocabulary and normalization remain frozen.
Targets are accepted recorded starts with certified source-scoped zeros, not latent physical demand. Unknown operating availability stays independently masked.
The fixed rule is inverse scaling then max(0, prediction). Rolling origins use eventual certified counts at bucket end under the original retrospective assumption, not real publication latency.
Native saved-validation probes matched exactly before test reads; actual target arrays and keys matched the saved benchmark. Protected checkpoints, preprocessing, historical manifest and original benchmark files remained unchanged.
Prediction SHA-256: `4f726d274a1d3090078f7663cc591e2d560d06d332056d168c6c38289f95accd`.
Recovery ZIP: `gs://citypulse-ai-511315-data/benchmarks/station100/six-method-followup-20261027/six-method-recovery.zip`, generation `1793135245751963`.
All published files passed full generation-pinned SHA readback; prediction Parquet row/schema checks and recovery ZIP member hashes passed. Restored forecast metrics were recomputed without model inference.
The evaluation was prepared from repository base `9a5fb26` with a pre-test SHA-bound overlay. The retained VM checkout and model-adapter source hashes are recorded separately in `preflight-verification.json`; reproducibility does not rely solely on its Git HEAD. The Python evaluation source matches the archived overlay. `evaluation-plan.json` binds the frozen artifacts and keys.
Publication recovery reused the original plan metadata and existing integrity report after two operator-wrapper errors; only saved-artifact verification/publication was repeated, never inference. No forecast, metric definition or model selection changed.
No full datasets, weights or per-target prediction files were downloaded to the Mac. Model weights are referenced in existing GCS model prefixes rather than duplicated.

## Observed resources

| Phase | Seconds | Peak process RSS (MiB) | Peak monitored workspace (MiB) | Minimum free disk (GiB) |
|---|---:|---:|---:|---:|
| preflight | 8.756 | 243.93 | 235.34 | 27.322 |
| mlp-compatibility | 7.329 | 204.17 | 235.36 | 27.322 |
| lstm-compatibility | 7.155 | 211.70 | 235.36 | 27.322 |
| mlp | 110.389 | 812.17 | 240.92 | 27.317 |
| lstm | 558.758 | 949.62 | 246.26 | 27.312 |
| publication-finalization | 112.274 | 368.11 | 312.49 | 27.247 |

Free disk after recovery publication: 27.230 GiB.
Recorded phase samples show zero swap and no guard failure. Workspace figures include protected prior-month files and are scoped observations, not a universal peak guarantee; the 1 GiB workspace cap remained enforced through publication.
Use the existing CPU environments, 512 MB DuckDB limit, 2 GiB process RSS, zero swap, minimum 15 GiB free disk and four-hour server STOP. No new infrastructure, installation or billing activation occurred.
Free Trial evidence: user confirmed October 27 2026, €257 available and €0.00 out-of-pocket. The CLI does not expose real-time remaining trial credit.

## Remaining work

Final station-level FastAPI/Docker integration and operational observation-latency policy remain pending. No serving integration was started. No further test-driven tuning is authorized by this evaluation.

MLflow follow-up run `2282aeb14a1d4634b0f0bee8358e457c` contains 90 metric values. A second import added zero values and retained its run ID; row fingerprints verified that all ten earlier runs were unchanged. The run export and finalization metadata are checksum-verified under the benchmark prefix’s `finalization/` directory.

Phase times include data loading, preparation and exports; they are not API latency
measurements. Compatibility RSS describes the supervising process, while the
MLP/LSTM rows measure the inference processes themselves. The VM is confirmed
stopped; its existing 100 GiB balanced disk is retained.

Combined predictions: 17,359,043 bytes, generation `1793135240190936`.
Recovery ZIP: 17,295,304 bytes.
Finalization manifest generation: `1793135623533156`; SHA-256:
`f6f53a65938cfcb5b7fc23aaff5b6deaed6bb28b4b43a17ee8914a8b72556283`.
