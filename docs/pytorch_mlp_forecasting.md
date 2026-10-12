# PyTorch fundamentals and first citywide MLP

This is an educational development experiment using the same 1,440 January–February 2024 hourly observations and 4,006,382 accepted ride starts as XGBoost. The February test period has already been inspected in prior experiments. These results are exploratory comparisons, not an untouched final benchmark. The final January 2024–December 2025 dataset plan and project scope are unchanged.

## Environment

The existing `.venv` remains Python 3.14.7 on macOS 14.5 ARM64. An official PyTorch **2.14.1** `cp314/macosx_14_0_arm64` wheel was available and installed without changing any pre-existing dependency versions. `pip check` reports no broken requirements. PyTorch autograd and an existing XGBoost tiny fit both passed in the same process. The model uses CPU explicitly, float32, one CPU thread and deterministic algorithms; no MPS/CUDA configuration is needed.

See [PyTorch installation guidance](https://pytorch.org/get-started/locally/) and [PyTorch fundamentals](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).

```bash
.venv/bin/python -m pip install -e '.[neural,forecasting,notebooks,dev]'
# Existing macOS OpenMP workaround for XGBoost; run from the project root.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
.venv/bin/python -m pytest -q
.venv/bin/python -m jupyter lab
```

Other measured versions: scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6. Existing XGBoost remains 3.4.1. Same-environment seeded runs are reproducible; exact results across platforms or library versions are not promised.

## Notebooks

`notebooks/05_pytorch_fundamentals.ipynb` contains eight executed code cells with small synthetic examples: tensor shapes/dtypes/devices, NumPy memory sharing, autograd, weights/biases, linear layers/ReLU, forward propagation, MSE, backward gradients, SGD/Adam updates, epochs/batches/DataLoader, train/eval modes, no-grad inference and scaling. A worked SGD update moves a weight from 1 to 1.8 and bias from 0 to 0.4 using visible gradients −8 and −4. It also executes a tiny Adam training loop and plots its loss. Execution took **22.96 seconds**, including kernel startup.

`notebooks/06_pytorch_mlp_forecasting.ipynb` contains eight executed code cells and three plots: training versus validation loss, actual versus predicted test demand, and a 24-hour March 1 forecast. It executes one bounded training experiment, evaluates the restored best state, verifies checkpoint restoration, and records measurements. Execution took **16.25 seconds**, including the existing data loader and kernel startup. Neither notebook had a code-cell error.

## Reused data and forecasting contract

The existing `forecasting/experiment.py` loader verifies local source hashes, duplicate handling, coverage and reconciliation. It includes adjacent-archive starts belonging to the January/February local window. Existing datasets and notebooks are not rewritten.

The new module reuses `FEATURES`, `training_rows`, `forecast_features` and `evaluate_model` from the existing XGBoost experiment. Importing these helpers does not fit XGBoost. Evaluation reuses the existing baseline metric implementation and next-day prediction contract. No baseline or feature definitions are duplicated.

The nine features are hour, weekday, weekend flag, hour sine/cosine, weekday sine/cosine, demand lag 24 and demand lag 168. Each NYC-local midnight forecast issues all **24 elapsed-hour** predictions together. Calendar values are known ahead; demand source timestamps must precede issuance. Elapsed-hour offsets and target grid use UTC; calendar values use America/New_York. There is no DST transition in this window. Missing lag observations fail rather than being imputed from future data.

Training targets are January only. The first 168 January hours lack historical lags and are dropped; **576 January 8–31 rows** remain. Feature and target StandardScalers are fitted exclusively on these January rows. Validation/test transforms reuse those statistics, and predictions are inverse-transformed to rides/hour. Earlier completed validation/test days can supply later forecasts' lag observations, but no actual inside the forecast block is used. The assumption that the preceding hour's demand is finalized at midnight needs revisiting before operational deployment.

## Architecture and training

The `torch.nn.Module` architecture is **9 → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)**. Each row produces one hourly value with shape `(batch, 1)`; forecasting a day uses 24 rows. The model has 865 learned weights/biases. ReLU hidden layers supply nonlinearity; the output stays linear because the standardized target may be negative. The existing forecasting adapter applies the same predeclared zero clipping to raw-demand predictions as XGBoost.

One fixed training configuration was used: Adam learning rate **0.001**, MSELoss, batch size **64**, seed **42**, maximum **200 epochs**, patience **20**, minimum improvement **0.00001** for resetting patience. DataLoader shuffles January training rows with a fixed generator; it never mixes chronological splits. Shuffling already-constructed causal training rows is safe for this independent-row feedforward model.

Backpropagation and optimizer steps use January batches only. Validation runs without gradients and selects the epoch with lowest standardized-target MSE. Patience controls stopping; the state with the absolute lowest observed validation loss is retained even if its improvement is below the patience threshold. There is no hyperparameter search, no test-based tuning and no February refit. This differs from XGBoost's configuration selection by validation MAE.

Training completed **32 epochs**, selected **epoch 12**, and the measured optimization-loop time was **0.227 seconds**. This timer includes mini-batch updates and post-epoch loss evaluations; it excludes data loading, preparation, model/optimizer initialization, checkpoint serialization and notebook startup.

At the selected epoch, full-dataset post-epoch losses were:

| Dataset | MSE in January-standardized target units |
|---|---:|
| January training | 0.273021 |
| February validation | 0.313892 |

These loss values are computed before raw-demand clipping; they are not directly in rides/hour. The January target standard deviation is 2071.027375 rides/hour. By epoch 32, training MSE had fallen to **0.205272**, while validation MSE rose to **0.332167**. This supports using the earlier validation-selected state rather than the last epoch. It is evidence of mild development-split overfitting after epoch 12, not proof of its cause.

## Measured forecast errors

Validation is February 1–14: 14 midnight origins/336 targets. Test is February 15–29: 15 origins/360 targets, including leap day. All models share these hours and the forecast contract. MAE/RMSE use rides/hour; WAPE is pooled absolute error divided by actual demand, expressed as a percentage.

| Split | Model | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| validation | Lag 24 | 908.15 | 1454.00 | 27.90% |
| validation | Lag 168 | 715.63 | 1202.43 | 21.98% |
| test | Lag 24 | 759.85 | 1199.06 | 26.67% |
| test | Lag 168 | 948.01 | 1438.95 | 33.27% |
| validation | XGBoost | 782.44 | 1242.43 | 24.04% |
| test | XGBoost | 649.27 | 994.20 | 22.79% |
| validation | PyTorch MLP | 782.26 | 1160.32 | 24.03% |
| test | PyTorch MLP | 622.37 | 973.49 | 21.84% |

**Validation:** MLP MAE/WAPE are nearly tied with XGBoost and better than lag 24, but worse than lag 168. Its validation RMSE is lower than all three comparators. The 0.18 rides/hour MAE difference from XGBoost is too small to support a meaningful superiority claim from this experiment.

**Test:** the MLP has lower measured MAE, RMSE and WAPE than both baselines and XGBoost. The prior XGBoost results were loaded from its measured experiment record with matching source-manifest provenance; it was not retrained. The test plot still shows missed commute peaks. We make no statistical-significance or final-model claim because this is a short, winter-only, previously inspected development window.

## Checkpoint and reproducibility

All generated files remain under the gitignored directory `outputs/phase3-pytorch-mlp-jan-feb-2024/run-20261012T125502822756Z`:

- `best_model.pt`: best state_dict, ordered feature names, January scaler statistics, selected epoch, settings, training bounds and forecast contract. Loaded with CPU mapping and `weights_only=True`.
- `experiment.json`: measured metrics, versions, source provenance, settings, loss summary and the example forecast.
- `loss_history.csv`: post-epoch training/validation losses.

Each notebook execution creates a new run directory. Training refuses an existing checkpoint path and uses exclusive creation to prevent accidental overwrites. Restoring the checkpoint exactly reproduced predictions in the measured run. The March 1 example predicts 24 hours from completed February history; March actuals are neither loaded nor scored.

## Verification and limitations

Focused tests: **12 passed in 6.92 seconds**. Full suite: **280 passed and 6 subtests passed in 34.28 seconds**. Tests cover architecture and output shape, finite forward passes, gradients and optimizer updates, exact CPU repeatability, best-state restoration, bounded early stopping, train-only scaler fits, validation distribution changes, exclusion of test data from training/selection, midnight issuance, leap day, future-target invariance, completed earlier-day lags, missing-lag rejection, checkpoint preservation, and metrics agreement with tensor MSE.

Only 576 training rows are available for 865 learned parameters. Correlated hourly errors and weather/holiday omissions limit interpretation. Training loss alone is not a forecasting metric; a neural network does not automatically outperform seasonal models. Continue learning from the executed notebooks; further model families or dataset expansion are outside this milestone.
