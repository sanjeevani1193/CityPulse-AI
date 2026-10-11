# First XGBoost forecasting experiment

This preliminary citywide experiment reuses the existing January–February 2024 loader: **1,440 continuous hourly observations and 4,006,382 accepted ride starts**. The loader includes adjacent source-archive contributions and verifies source hashes, duplicates and reconciliation. No download, ETL or cloud operation is needed. The final January 2024–December 2025 dataset and project scope remain unchanged.

## Features and forecast contract

`forecasting/xgboost_model.py` separates feature creation, January training-row preparation, next-day prediction, validation selection and evaluation. It reuses `forecasting/experiment.py` for the citywide series and `forecasting/baselines.py` for boundaries and metrics.

Each forecast is issued at **America/New_York local midnight** for the next **24 elapsed hours**. UTC timestamps define the hourly grid and lag offsets; calendar features use NYC local time. There is no DST transition during this experiment. Calendar inputs are hour, weekday, weekend indicator, and sine/cosine encodings of hour and weekday. Cyclical encoding places the end and start of each calendar cycle close together.

Demand lags refer to **target hour minus 24 or 168 elapsed hours**, not issuance minus those offsets. Every lag timestamp is checked to be strictly before issuance. All 24 predictions are issued together; no actual count from within that forecast day is used. Calendar values for tomorrow are already known. Completed observations from earlier validation/test days can supply later forecasts' lags; this is rolling-origin evaluation with a frozen model, not refitting.

The implementation assumes the preceding hour's finalized demand is available immediately at midnight. Real operational reporting delay would require an explicit availability rule before deployment. Missing prediction lags raise an error rather than being imputed. January training rows missing targets or either lag are dropped: **168 warm-up rows dropped, 576 rows retained (January 8–31)**. February data never supplies January training values. Predictions are clipped at zero under a physical constraint declared before evaluation; the same rule applies to both splits. No future demand or weather is a feature.

## Training and selection

XGBoost learns an ensemble of regression trees by adding corrections to a current prediction. Training learns tree thresholds and leaf values; hyperparameters govern how those trees are built. See [XGBoost's official tree/boosting explanation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

All four configurations below were predefined. Each fits the same January rows using squared-error loss. February 1–14 validation MAE selects the configuration; February 15–29 is not provided to the selector. Exact ties keep the earlier configuration. There is no test-based tuning, early stopping or refit on February.

| Configuration | Trees | Maximum depth | Learning rate | Subsample | Validation MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 100 | 2 | 0.05 | 1.0 | 838.89 |
| 1 | 200 | 2 | 0.05 | 0.8 | 827.74 |
| 2 | 150 | 3 | 0.05 | 1.0 | 782.44 |
| 3 | 150 | 3 | 0.1 | 0.8 | 797.91 |

**Selected:** 150 trees, maximum depth 3, learning rate 0.05, subsample 1.0. Fixed settings: `random_state=42`, `n_jobs=1`, `tree_method="hist"`, `objective="reg:squarederror"`, `eval_metric="mae"`, `importance_type="gain"`. Selection uses the reusable MAE calculation, not training loss. The selected already-fitted January model is evaluated once on the test split.

## Measured results

Validation has 14 midnight origins and 336 targets; test has 15 origins and 360 targets, including leap day. Baselines and XGBoost share identical targets and issuance rules. MAE and RMSE are rides/hour. WAPE is 100 times total absolute error divided by total actual demand.

| Split | Model | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| validation | Lag 24 | 908.15 | 1454.00 | 27.90% |
| validation | Lag 168 | 715.63 | 1202.43 | 21.98% |
| test | Lag 24 | 759.85 | 1199.06 | 26.67% |
| test | Lag 168 | 948.01 | 1438.95 | 33.27% |
| validation | XGBoost | 782.44 | 1242.43 | 24.04% |
| test | XGBoost | 649.27 | 994.20 | 22.79% |

**Validation:** XGBoost beats lag 24 but loses to lag 168 on all three metrics. **Test:** XGBoost beats both seasonal baselines on all three metrics. This does not establish general superiority or statistical significance. Stronger validation performance from the weekly baseline supports weekly seasonality as a useful reference. The changing baseline ranking between splits shows that recent daily demand can sometimes be more informative than the previous week. The plotted test week shows that XGBoost still misses some large commute peaks.

## Measured feature importance

Normalized gain credits features for reductions in training loss at tree splits. Correlated hour encodings and lags can share or substitute credit; importance is neither causal nor an independent test-set explanation.

| Feature | Normalized gain |
|---|---:|
| hour_cos | 37.46% |
| lag_168 | 23.66% |
| hour | 10.56% |
| lag_24 | 9.52% |
| hour_sin | 8.65% |
| weekday | 6.13% |
| weekday_cos | 2.21% |
| weekday_sin | 1.80% |
| is_weekend | 0.00% |

The dominant features are cosine hour encoding and last-week demand. Zero gain for the weekend flag means no selected split used that feature; it does not prove that weekends have no effect, since weekday features encode related information.

## Reproduce locally

Install the optional extra in the existing environment:

```bash
.venv/bin/python -m pip install -e '.[forecasting,notebooks,dev]'
```

[XGBoost's macOS installation instructions](https://xgboost.readthedocs.io/en/stable/install.html) require OpenMP. On this machine, XGBoost imported only after its runtime search path included scikit-learn's bundled OpenMP. Homebrew attempted a CMake dependency upgrade; that build was stopped. An installation without dependency upgrades could not find CMake in its build path, and Homebrew disabled `--env=std`. Python and the virtual environment were retained. The following per-process setting was verified by a tiny fit, tests and notebook execution:

```bash
# Run from the project root; this is the existing Python 3.14 macOS environment.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
.venv/bin/python -m pytest -q
.venv/bin/python -m jupyter lab notebooks/04_xgboost_forecasting.ipynb
```

This fallback requires the scikit-learn macOS wheel's `libomp.dylib`; it is specific to this environment. A normally installed system OpenMP runtime does not need this fallback. Restart the kernel from the configured Jupyter process so it inherits the setting.

Measured versions: XGBoost 3.4.1, scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6; existing Python 3.14.7. Seed and one CPU thread improve repeatability; bitwise agreement across other versions/platforms is not guaranteed.

The executed notebook contains an actual/prediction comparison plot, metric table, gain-importance plot, and a 24-hour March 1 forecast using completed February history. March predictions have no loaded actual targets and are not scored. All nine code cells executed without errors in **42.65 seconds**. A measured provenance/results record is saved locally to the ignored `outputs/phase3-xgboost-jan-feb-2024/experiment.json`; existing data outputs are not rewritten.

## Verification and limits

Focused tests: **10 passed in 5.11 seconds**. Full suite: **268 passed and 6 subtests passed in 34.72 seconds**, using the runtime setting above. Tests cover feature/target alignment, January warm-up, no future-target access, exact midnight issuance, leap-day forecasting, missing lags, reproducibility, rejection of test data during selection, January-only fits and no February refit.

Only one winter training month is available, with 576 training rows. Forecast errors are temporally correlated, and weather, holidays, reporting latency and unmet demand are not modelled. These results are development evidence, not final 24-month performance. The next planned experiments must use chronological evaluation on the fixed full historical dataset; this experiment does not authorize further ingestion or model development.
