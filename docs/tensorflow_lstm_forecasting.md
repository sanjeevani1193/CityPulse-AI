# TensorFlow/Keras LSTM development experiment

This educational experiment uses the existing January–February 2024 citywide series: **1,440 observed hourly counts and 4,006,382 accepted ride starts**. The final January 2024–December 2025 dataset plan is unchanged. February's test window has been inspected for previous models; these are exploratory development comparisons, not an untouched final benchmark. No Citi Bike downloads, ETL or cloud operations were run.

## Environment and compatibility

Stable TensorFlow **2.21.0** has macOS ARM wheels for Python 3.10–3.13, not Python 3.14; see [the official release files](https://pypi.org/project/tensorflow/2.21.0/). A metadata-only probe in the existing Python 3.14 environment offered TensorFlow 2.22.0rc0. A release candidate was not installed there.

An already installed ARM64 Python 3.11.4 initially supported TensorFlow's CPU forward pass, but pytest startup segfaulted while importing that interpreter's native `readline` extension, before TensorFlow tests loaded. No model training was performed in that environment. An already installed ARM64 **Python 3.11.15** at `~/anaconda3/envs/bookrag/bin/python` passed the readline check and was used to create **`.venv-tensorflow-311`**. The earlier `.venv-tensorflow` is retained and unused under the no-deletion constraint.

The working isolated environment uses **TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.6**, on macOS 14.5 ARM64. It uses CPU explicitly, one intra/inter-op thread, deterministic operations and seed 42. No new Python version or accelerator plugin was installed. The original Python 3.14.7 `.venv` and its dependency versions remain unchanged; existing notebooks, historical manifest and prior model result records retained their SHA-256 hashes. The isolated environment's `pip check` passed.

To reproduce using the already installed interpreter:

```bash
# The isolated environment already exists; creation is shown for reproducibility.
~/anaconda3/envs/bookrag/bin/python -m venv .venv-tensorflow-311
.venv-tensorflow-311/bin/python -m pip install -e '.[tensorflow,notebooks,dev]'
.venv-tensorflow-311/bin/python -m ipykernel install \
  --prefix "$PWD/.venv-tensorflow-311" --name citypulse-tensorflow \
  --display-name 'CityPulse AI TensorFlow (Python 3.11)'
.venv-tensorflow-311/bin/python -m jupyter lab
.venv-tensorflow-311/bin/python -m pytest -q tests/test_tensorflow_lstm.py
```

Select the **CityPulse AI TensorFlow (Python 3.11)** kernel. Launching Jupyter from this environment exposes its locally installed kernelspec without altering existing kernels. Restart the kernel before Run All: TensorFlow thread/device configuration must happen before runtime initialization. `.gitignore` excludes both `.venv-tensorflow*` environments and all generated experiment outputs.

## Learning notebooks

`notebooks/07_tensorflow_lstm_fundamentals.ipynb` has eight executed code cells: TensorFlow tensors and differentiation; Keras Sequential/Functional APIs; Dense versus SimpleRNN/LSTM shapes; hidden/cell states; a scalar gate calculation; backpropagation-through-time gradients; compile/fit/predict; and a tiny synthetic fit using early stopping and checkpoints. It explains why vanishing/exploding gradients complicate vanilla RNN training and why LSTM gates help without guaranteeing long-term memory. Execution took **40.99 seconds**, including kernel startup.

`notebooks/08_tensorflow_lstm_forecasting.ipynb` has eight executed code cells: the read-only loader and pretraining eligibility table, one bounded fit, learning curves, comparison metrics, actual/prediction plot, checkpoint restoration and a March 1 forecast. Execution took **30.03 seconds**, including kernel startup and source validation. Both notebooks executed without code-cell errors. Synthetic notebook outputs and real experiment outputs use separate new, gitignored run directories.

## Sequence and scaling decisions

Reusable code is in `forecasting/lstm.py`. It reuses the existing citywide loader, split boundaries and pooled metric functions. The loader includes adjacent-source-archive ride starts belonging to the local January/February window and checks source hashes, duplicates and reconciliation. Existing validated data is not rewritten.

Each sequence starts at **America/New_York local midnight**. Its input is the previous **48 observed elapsed hours**, strictly before issuance, with shape **(samples, 48, 1)**. Its target is the next **24 elapsed hourly counts**, shape **(samples, 24)**. UTC defines elapsed-hour grids; NYC defines calendar splits and issuance. There is no DST transition during this experiment. The code rejects blocks whose 24 targets extend beyond the split and rejects missing/nonfinite/negative context or targets rather than filling them.

| Split | Eligible sequences | Origins | Hourly targets |
|---|---:|---|---:|
| Training | **29** | January 3–31 | 696 |
| Validation | **14** | February 1–14 | 336 |
| Test | **15** | February 15–29 | 360 |

These counts were displayed before training. January 1–2 have insufficient in-window history and are explicitly excluded; no December source input is invented. There are exactly 29 training examples, not 696 independent sequences. No hourly sliding origins or February training examples were added.

One univariate StandardScaler fits the **744 observed January hours only**, including training input-only warm-up and training targets. That same mean/scale transforms contexts and targets. It never uses February observations for fitting. Predictions are inverse-transformed into ride counts, then clipped at zero under the same predeclared physical constraint as XGBoost/MLP.

As later validation/test midnights arrive, earlier completed validation/test observations may enter the 48-hour context. No actual count inside the forecast horizon is an input. We assume the preceding hour's count is finalized at midnight; operational reporting latency would require a separate availability policy before deployment.

## Architecture and bounded training

The model is **LSTM(32) → Dense(24)**, with one input feature. The LSTM produces its final 32-unit hidden representation from the 48-step input. The Dense head generates all 24 hourly predictions directly; there is no recursive feedback of predicted/actual future values. The LSTM is stateless across samples, so shuffling already-built January sequences does not carry state between days.

Parameter counts: 4 × 32 × (1 input + 32 recurrent + 1 bias) = **4,352** for the LSTM; 32 × 24 + 24 = **792** for the Dense head; **5,144 learned parameters**, confirmed by `model.count_params()`. After training, Keras summary may also include Adam optimizer-state variables in its total; these are not additional learned network weights.

The single predefined fit used Adam learning rate **0.001**, MSE, batch size **4**, seed **42**, at most **80 epochs**, early-stopping patience **10**, minimum improvement **0.00001**, and best-weight restoration. Validation scaled MSE selects weights; only January batches contribute gradients. No hyperparameter search, test-loss callback or February refit was used.

The run completed **30 epochs** and restored **epoch 20**. `ModelCheckpoint` saved the absolute minimum validation-loss state; after `EarlyStopping` restored its selected weights, the code explicitly reloaded that checkpoint to ensure minimum-loss restoration independent of the patience tolerance. The fit, checkpoint writes/restoration and frozen-state train/validation evaluation took **3.475 seconds**; preparation, loader and notebook startup are outside that timer.

Frozen best-state MSE in January-standardized target units:

| Dataset | Scaled MSE |
|---|---:|
| January training | 0.343098 |
| February validation | 0.430308 |

Keras's training curve averages batch losses while weights change; its validation curve scores the completed epoch state. At epoch 20, batch-average training MSE was 0.349788 and validation MSE was 0.430308. At epoch 30 these were 0.336167 and 0.447486. Continued training-loss improvement with slightly worse validation loss supports the earlier stopping choice. It does not uniquely establish the cause of errors or prove that more capacity would help.

## Measured metrics and comparison

All models share the same validation/test hours and midnight-issued 24-elapsed-hour contract. MAE/RMSE use **rides/hour**, WAPE is pooled absolute error divided by actual demand as a percentage. Existing baselines were evaluated with their existing implementation. XGBoost and MLP measurements were loaded from prior records with matching source-manifest provenance; their models/results were neither retrained nor modified.

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
| validation | Keras LSTM | 965.19 | 1341.21 | 29.65% |
| test | Keras LSTM | 680.16 | 1011.21 | 23.87% |

**Validation:** LSTM MAE/WAPE are worse than all four comparators. Its RMSE is lower than lag 24, but worse than lag 168, XGBoost and MLP. **Test:** LSTM beats both seasonal baselines on all three metrics, but loses to both XGBoost and MLP on all three metrics. We therefore do not claim improvement over the existing trained models. The test plot shows systematic underprediction of several commute peaks.

This is not an equal-input architecture comparison: the LSTM has only a 48-hour univariate history, while XGBoost/MLP have explicit calendar and lag-168 features. Training warm-up also differs (48 hours here versus 168 for engineered-lag models), although evaluation targets match. More recurrent complexity is not automatically more predictive. Only 29 sequences train 5,144 parameters; winter-only coverage, omitted weather/holidays, previously inspected test data and correlated errors limit interpretation. No statistical-significance or final 24-month performance claim is made, and no test-driven tuning followed the results.

## Saved artifacts and verification

New files are under `outputs/phase3-tensorflow-lstm-jan-feb-2024/run-20261012T141848308777Z`:

- `best.weights.h5`: validation-best model weights.
- `experiment.json`: versions, provenance, settings, mean/scale/variance and training-hour count for the January scaler, sequence counts, losses, metrics and example forecast.
- `loss_history.csv`: epoch losses.

Reconstruct the architecture, load weights, and supply the saved scaler statistics for future inference; the executed notebook verifies that reloaded weights with the January scaler reproduce predictions exactly. The example forecasts March 1 using February 28–29 observations; March targets are not loaded or scored. Existing checkpoints are refused and each notebook run has a new output directory.

Final focused verification: **9 tests passed in 19.96 seconds**, with **87 Keras/NumPy warnings**. Tests covered alignment, chronological splits, midnight issuance, missing history, 24-output shapes, January-only scaling, no test fitting, exact same-environment CPU predictions, best-loss checkpoint restoration, completed prior-day context and original-unit metrics. An additional regression uses scripted epoch states/losses to force a best epoch before the final epoch and checks that those earlier weights are restored; its synthetic losses are a test fixture, not model benchmark results.

The original Python 3.14 regression suite passed **280 tests and 6 subtests in 50.42 seconds**, with **one expected skip** of the TensorFlow test module because TensorFlow is deliberately isolated. Keras warnings include its ignored `fit(shuffle=True)` argument for an already-shuffled tf.data Dataset, third-party NumPy `__array__` deprecation warnings, and optimizer-variable loading on a newly created inference model. Inference weights were checked exactly; optimizer state is not used for inference or resumed training in this milestone. Warnings were not hidden or treated as successful evidence of anything beyond passing tests.
