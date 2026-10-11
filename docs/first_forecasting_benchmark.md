# First citywide forecasting benchmark

This is a preliminary January–February 2024 experiment. The final dataset and
full CityPulse AI scope remain unchanged: January 2024–December 2025, training
through June 2025, validation July–September 2025, and test October–December 2025.
No learned model, weather integration or station cohort is implemented here.

## Inputs and coverage

The read-only loader in `forecasting/experiment.py` uses registered local hourly
Parquet files from January, February and March 2024. It verifies artifact hashes,
monthly source reconciliation and cross-archive IDs, including rejection audits.
Cross-archive duplicates cause a failure: hourly aggregates cannot safely remove
individual duplicated trips, so those inputs would need the existing audited
catalogue. Source artifacts and the live manifest are never modified.

UTC hourly keys are filtered to the half-open NYC-local window January 1–March 1.
January's 31 days and February's 29 days yield 1,440 elapsed hours, with no DST
transition. Missing hourly observations remain unknown by default; forecasts and
metrics refuse missing required lags or labels. This real run needed no zero fill.

| Local start month | Contributions | Accepted starts |
|---|---|---:|
| January 2024 | 1,886,515 from January + 231 from February | 1,886,746 |
| February 2024 | 2,119,401 from February + 235 from March | 2,119,636 |
| Total | 1,440 observed citywide hours | 4,006,382 |

No cross-archive duplicate IDs were found among these three archives. Rejected
rides are excluded. Archive coverage and corrections are still limitations;
these targets represent recorded accepted starts, not unmet demand.

## Forecast issuance and evaluation

January is reference history; no parameters are learned. Validation covers
February 1–14 (14 origins × 24 = 336 targets per model), and test covers February
15–29 (15 origins × 24 = 360 targets per model). Boundaries use America/New_York.

One 24-elapsed-hour forecast block is issued at each local midnight. Lag 24 copies
the previous day's completed block; lag 168 copies the block one week earlier.
Every input timestamp is strictly before issuance. Completed earlier validation
or test days become history for later origins, as in a live rolling forecast;
there are no within-day observation updates. Both lag choices are fixed in
advance, with no test-driven tuning. Counts are assumed available immediately
after each hour completes; real publication delays need a future policy.

Metrics pool all hourly errors within each split. MAE and RMSE are rides/hour.
WAPE is `100 * sum(abs(actual - prediction)) / sum(actual)`, a percentage,
undefined when total actual starts are zero.

| Split | Lag | MAE | RMSE | WAPE (%) |
|---|---:|---:|---:|---:|
| Validation | 24 | 908.146 | 1,454.002 | 27.897 |
| Validation | 168 | 715.628 | 1,202.430 | 21.983 |
| Test | 24 | 759.853 | 1,199.061 | 26.666 |
| Test | 168 | 948.014 | 1,438.950 | 33.269 |

These are measured local-data results, rounded to three decimals. Weekly reuse
performed better on validation but worse on test, so this short experiment does
not establish a universally better seasonal rule. January-only descriptive plots
show hour-of-day and weekday/weekend patterns; those averages are limited
evidence, not a causal explanation. Winter-only observations, holidays, weather,
correlated errors and just 15 test origins limit generalization. There is no
statistical significance or final-model performance claim.

## Reproduce

Use the existing virtual environment and local historical outputs:

```bash
source .venv/bin/activate
python -m pytest -q
jupyter lab notebooks/03_forecasting_baselines.ipynb
```

Select **CityPulse AI (.venv)** and run all cells. If notebook dependencies are
missing, install the existing extra with `python -m pip install -e '.[notebooks,dev]'`.
The executed notebook contains source reconciliation, actual/predicted plots,
metrics and seasonality explanations. It neither downloads data nor contacts GCS.

For programmatic use, call `load_development_citywide`, `seasonal_naive`,
`evaluate_midnights` and `error_metrics` from the forecasting package. The
duplicate checker runs archive-pair joins sequentially: a real-data grouping and
a combined UNION join exceeded 512 MB during development; sequential joins
with fail-fast duplicate checks completed without raising that memory limit.

Continue the fixed historical expansion before final benchmarking. Later work
must account for reporting latency and DST (24 elapsed hours need not equal a
local calendar day), keep model selection chronological, and select the future
100-station modelling cohort using training-period data only.
