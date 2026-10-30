# Final 100-station TFT training

**Current status:** the [six-method benchmark](six_model_followup_benchmark.md)
and [Forecasting v1 native/container serving verification](forecasting_v1.md)
are complete. The training/validation measurements below are preserved from the
original phase; historical held-out/pending statements are not current status.

## Finalized six-epoch experiment

The native PyTorch TFT completed six full epochs on the frozen 100-station
dataset. Each epoch visited all 53,300 training windows (1,279,200 trusted target
positions). Validation used 9,200 windows and 220,800 targets. Training covers
January 2024–June 2025; validation covers July–September 2025. October–December
2025 remains held out: **final testing and final-model deployment integration
are pending**. No test outcomes were used for training or checkpoint selection.

The selected checkpoint is **`epoch-006.pt`**, with the lowest standardized masked
validation pinball loss, **0.1562411265**. The six-epoch budget is finalized;
the original ten-epoch ceiling and patience-three stopping criterion were not
reached. This is not a convergence claim or an instruction to resume training.

| Epoch | Training standardized pinball | Validation standardized pinball | Training + validation seconds |
| --- | ---: | ---: | ---: |
| 1 | 0.144327091 | 0.176327097 | 2818.498 |
| 2 | 0.135084683 | 0.178040008 | 2632.394 |
| 3 | 0.133693688 | 0.168893157 | 2522.462 |
| 4 | 0.132038960 | 0.180204174 | 2595.338 |
| 5 | 0.128662177 | 0.164480040 | 3050.184 |
| 6 | 0.124674028 | 0.156241126 | 2701.040 |

These values come from the finalized experiment/history artifacts, not a new
training run. Epochs 1–3 preceded the bounded continuation for epochs 4–6.

## Selected-checkpoint validation

All scores below use the same 220,800 eligible station/origin/target/lead keys.
Point scores use q50 in original ride-count units. Clipping is the predeclared
`max(0, inverse_scaled_quantile)` rule applied separately to each quantile;
this monotone floor preserves ordering. It changes neither training nor selection.

| Forecast | MAE | RMSE | WAPE (%) | Mean quantile pinball | q10–q90 coverage (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TFT raw | 4.9205 | 7.5498 | 35.988 | 1.584481 | 78.112 |
| TFT clipped | **4.7391** | 7.4922 | 34.662 | 1.526593 | 79.886 |
| Lag-24 | 6.0211 | 9.5619 | 44.038 | — | — |
| Lag-168 | 5.2934 | 8.3116 | 38.716 | — | — |

Raw q10/q50/q90 pinball losses are 0.987379, 2.460242, 1.305822;
clipped values are 0.937233, 2.369570, 1.272976. Raw negative median forecasts
represent 10.635% of targets. Baselines are point forecasts without fitted
intervals. See the [XGBoost comparison](station_xgboost_benchmark.md) for the
separate bounded search on identical validation keys; no test comparison exists.

## Architecture, data and reproducibility

`forecasting/tft_training.py` uses the existing TFT unchanged: 100 station codes,
hidden size 16, two attention heads, dropout 0.1, 336 elapsed encoder hours,
24 elapsed decoder hours, ordered q10/q50/q90 outputs and **62,511 parameters**.
Station embeddings/static context, variable selection, gated residual networks,
LSTM encoder/decoder and interpretable temporal attention remain implemented.
Forecast issuance is NYC-local midnight; UTC elapsed-hour ordering handles DST.
The `accepted_recorded_starts_v1` policy describes recorded accepted starts,
not unmet physical demand or independently observed station operation.

Station selection and fitted preprocessing use training data only. Adam uses
learning rate 0.001, gradient norm clipping 1, batch size 32, seed 42 and two CPU
threads with deterministic algorithms. Seeded station blocks and within-station
origin shuffling permit one station frame to be cached; full training tensors
are not materialized. Masked pinball is normalized by trusted targets. Completed
checkpoints preserve model, optimizer, RNG, history, configuration and global best.

The recorded code revision is `1b6063ce1b7ecfb52f99ba0f1df1e092890eca96`.
The Linux runtime used Python 3.11 and `torch==2.14.1+cpu` in an isolated training
environment, preserving the ETL environment. No GPU packages are needed.
Dataset manifest SHA-256:
`a6747adb2485276e2e7aa735dc24b1a10048e9b8526928c489a9472a532ba952`.
Paired validation-key SHA-256:
`48821491137ff4c4912493b5f1111f032539b9b16d2a6e88b3444af2df03dbb2`.

For reproduction, use the recorded revision, verified dataset, frozen preprocessing,
configuration and runtime from the recovery bundle. `scripts/train_station_tft.py`
is the training entry point; inspect its arguments before a separately authorized
run. Do not restart completed epochs or use the final test split. Synthetic
training/forward tests exist in `tests/test_tft.py` and `tests/test_tft_training.py`;
a review-only check can select metric/control tests without running a model:

```bash
python -m pytest -q tests/test_tft_training.py -k \
  'masked_original_unit_metrics or empty_metrics or test_split_forbidden or resume_driver or session_cap_preserves or nonnegative_clipping'
```

Any future authorized VM run must retain the four-hour server-side STOP, 2 GiB
process RSS guard, zero-swap/host-pressure checks, 15 GiB free disk reserve and
512 MB DuckDB setting. The training CLI alone does not establish these guards.
Stop the VM on completion or failure; retained disk/GCS storage remains billable
against trial credits. Never automatically upgrade billing or delete protected data.

## MLflow and verified recovery

The existing MLflow run `0f58b4ec8ea9411db10fb8c16a95891f` in
`CityPulse final 100-station TFT training` contains reconciled epoch 1–6 loss
histories, with zero duplicate metric steps. Repeated reconciliation created no
new run or metrics. Immutable initial parameters remain intact; current selection
and completion are recorded through step metrics and tags. Only small reports
and GCS references are logged; weights/datasets are not duplicated in MLflow.
Use the existing [local MLflow UI instructions](local_mlflow_tracking.md).

Selected checkpoint:
`gs://citypulse-ai-511315-data/models/tft/station100-20261025/epochs/epoch-006.pt`

- Generation: `1792960124653611`; size: 1,192,212 bytes.
- SHA-256: `5396d276b1d05f84ffcd969767b4319faa659cb2107c022e3fd87d70c7fe12c6`.

Final reports and recovery:
`gs://citypulse-ai-511315-data/models/tft/station100-20261025/finalized-six-epochs-20261026/`

The eight published objects were generation- and checksum-verified. The
`training-recovery.zip` generation is `1793023020071808`, with SHA-256
`06fbb9dc9fa32f400fd83b493cea651ee4476c68198ede383b698e2e094502ee`.
The bundle includes source/configuration, fitted preprocessing, reports, checkpoint
and receipts. Verify its pinned generation and SHA-256 before extracting into a
new recovery directory; retain original receipts and verify member checksums.
A fresh process restored model/optimizer state and reproduced the saved synthetic
probe exactly with output shape `(1, 24, 3)` in the checked runtime. This is a
restoration check, not a claim of cross-platform bitwise equality or new accuracy.
Validation summaries were exported from saved metrics without rerunning inference.
No final test predictions were generated. Artifacts remain gitignored; the VM
was stopped after the verified run.
