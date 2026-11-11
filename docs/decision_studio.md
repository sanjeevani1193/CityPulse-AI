# CityPulse Decision Studio - offline presentation, Milestones 2 and 3

This is a separate Streamlit + Plotly presentation of **saved** CityPulse results.
Overview, Forecast Explorer and Model Comparison are implemented. SQLPilot and
ExperimentLab & Methodology are visibly labelled upcoming, not functional pages.
It is not publicly hosted yet and does not perform live forecasts, SQL execution,
LLM calls or statistical experiments. GCP backup/retirement is a separate task.

The existing six-model API, models, SQLPilot, ExperimentLab, Terraform and cloud
workflows are unchanged. No GCP credentials, cloud SDK, PyTorch or TensorFlow are
needed by this app. No artifact downloads occur at startup.

## Run locally

Use Python 3.14 (the locally tested interpreter); use the same version for later
Community Cloud deployment. Dependencies live beside the app entrypoint to avoid
installing the repository's forecasting extras. The runtime requires only the
four packages in `apps/decision_studio/requirements.txt` and their dependencies.

```sh
python3 -m venv .venv-decision-studio
.venv-decision-studio/bin/python -m pip install --only-binary=:all: \
  -r apps/decision_studio/requirements.txt pytest==9.1.1
.venv-decision-studio/bin/python -m pip check

# Uses the reviewed compact bundle; no private research artifacts required.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-decision-studio/bin/python -m streamlit run \
  apps/decision_studio/streamlit_app.py --server.address=127.0.0.1 --server.port=8502
```

Open `http://127.0.0.1:8502`; Ctrl-C stops the app. The shared existing Streamlit
configuration disables usage telemetry and file watching. Leave other projects
and their environments alone. This environment is local and gitignored.

## Files and data flow

| File/module | Responsibility |
|---|---|
| `apps/decision_studio/streamlit_app.py` | Small entrypoint; resolves `src/` independently of shell `PYTHONPATH` |
| `src/citypulse_ai/decision_studio/bundle.py` | Batch-read verified local predictions; produce create-only, reproducible public bundle |
| `artifacts.py` | Fixed filenames, version/schema/hash checks, exact elapsed-hour sequence validation; fail closed |
| `metrics.py` | Paired error formulas, baseline-relative improvement, selected-model rankings |
| `charts.py` | Plotly figures; receives validated saved summaries, not model objects |
| `ui.py`, `workspaces.py`, `app.py` | Shared presentation, routing, overview/comparison and explicit upcoming pages |
| `forecasts.py`, `explorer.py` | Exact saved sequence selection, descriptive errors and Forecast Explorer UI |
| `apps/decision_studio/data/` | Small, public-safe aggregate results, sample sequences and provenance |
| `tests/test_decision_studio.py` | Artifact, mathematics, DST, exact-source-row and Streamlit tests |
| `scripts/verify_decision_studio.py` | Real loopback HTTP and WebSocket render check; terminates its own process |

**Data flow:** frozen Parquet/JSON/CSV → verified exporter → four small bundle
files → validated loader → metric/ranking functions → Plotly → Streamlit.
Loading, calculation and display stay separate so malformed data fails before
rendering, formulas are independently testable, and UI changes cannot silently
change the benchmark. The app never imports the inference adapters.

## Authentic demo data and provenance

The source is `outputs/forecasting-v1/saved-test-predictions.parquet`:
220,800 paired targets, 20 columns, SHA-256
`4f726d274a1d3090078f7663cc591e2d560d06d332056d168c6c38289f95accd`.
It is private-research input to the exporter, not an app runtime dependency.
The tracked existing `src/citypulse_ai/api/station_artifacts.json` supplies
canonical full-test/validation metrics and the frozen identities. Existing
`outputs/forecasting-v1/diagnostics-final/*.csv` supplies full-test breakdowns;
the exporter verifies their source/key provenance and overall metrics. The final
historical report supplies pipeline totals; its hash is recorded, but its private
paths and full report contents are excluded from the public bundle.

| Important prediction columns | Meaning |
|---|---|
| `station_id` | Exact string ID, never an inferred station name or floating-point ID |
| `origin_utc` | NYC-local-midnight forecast issuance expressed in UTC |
| `target_hour_utc` | UTC start of the elapsed-hour target bucket |
| `horizon_step` | 1–24; target = origin + (step − 1) elapsed hours |
| `actual`, `trusted` | Accepted recorded starts and certification mask, not unconstrained physical demand |
| `xgb_raw/clipped`, `mlp_raw/clipped`, `lstm_raw/clipped` | Saved raw and fixed nonnegative point forecasts |
| `tft_q10/q50/q90_raw/clipped` | Saved TFT quantiles; q50 is the point forecast |
| `lag24`, `lag168` | Saved causal seasonal baselines, already nonnegative |

The exact 20-field schema (including timestamp units) is recorded in
`data/manifest.json`. The public sequence Parquet retains original values/types
and UTC timezone information and removes pandas metadata. It contains no ride IDs,
source paths, model weights, credentials or invented station locations.

**Selection fixed independently of test errors:** sort the original 100 stations
by frozen training-only mean count ascending, then exact station ID. Take
zero-based ranks `[0,16,33,34,50,66,67,83,99]` across the existing lower/middle/higher
activity strata, and origins October 1, November 2 and December 1, 2025. This yields
9 stations × 3 dates × 24 elapsed hours = **648 target rows / 27 complete windows**.
These are illustrative strata/month examples, not a statistically representative
replacement for the evaluation cohort. Forecast Explorer displays only these windows.

The November 2 sequence preserves both 01:00 offsets and ends at 22:00 EST. It
does not fabricate a 25th prediction or fill the unscored 23:00 bucket. Station IDs,
origins and horizons must remain intact if this subset is extended later.

```sh
# Rebuild ONLY from existing authorized local files. No cloud operations.
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-decision-studio/bin/python scripts/build_decision_studio_bundle.py
```

The exporter checks source hashes, reads at most 8,192 source rows per batch,
records source paths **relative to the repository**, schema/counts, transformation,
source Git revision, exporter hash and PyArrow version. Each output has SHA-256
and byte size in `manifest.json`. Identical reruns succeed without rewriting;
different bytes require `--output` pointing to a new directory. A changed PyArrow
version can change encoded Parquet bytes. Checksums detect corruption; they are
not digital signatures. Trust comes from reviewing the bundle and Git history.

## Metrics and scientific boundaries

- **MAE** = mean absolute error, in starts/station-hour. XGBoost test MAE 3.3663
  means about 3.37 starts of average absolute discrepancy, not 3.37% error.
- **RMSE** = square root of mean squared error, same units; emphasizes large misses.
- **WAPE (%)** = 100 × sum absolute error / sum actual. Undefined for zero total
  actual; the metric helper returns `None`, not a fabricated 0%.
- **Baseline improvement (%)** = 100 × (baseline error − model error) / baseline
  error. Negative values mean worse performance than that baseline.

Leaderboard rows always use the frozen **220,800-target** cohort for the selected
validation/test split, not metrics calculated from 648 demo targets. Diagnostics
are explicitly full-test-only, even when validation is selected above them.
Subset comparisons cannot replace the published results: they have different
station/date weights and can give different rankings. No new model selection or
significance claim is made. The test WAPE denominator is lower than validation;
lower test MAE does not establish better relative performance.

Retain the follow-up disclosure: MLP/LSTM development occurred after initial
four-model test scores were known. This was not a fully blind six-model study.
The benchmark assumes recorded counts are available at bucket end; publisher
latency was not replayed. Certified accepted-record zeros do not prove zero
physical demand, publisher completeness or operating availability. TFT's 79.495%
aggregate q10–q90 coverage does not guarantee per-station calibration.

## Verification and remaining work

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-decision-studio/bin/python -m pytest tests/test_decision_studio.py -q
.venv-decision-studio/bin/python scripts/verify_decision_studio.py
git diff --check
```

AppTest checks both completed workspaces, controls, empty selection, upcoming
pages and fail-closed missing-artifact behavior. The loopback smoke test checks
HTTP health/root and renders both pages over real Streamlit WebSockets, then
stops the owned server. Neither test is an LLM evaluation or model inference.
Browser screenshots and real Community Cloud hosting are not claimed.

The source-row equality, exporter reproduction and wrong-source-hash tests need
the retained private local export inputs. On a fresh clone these three checks
explicitly skip with a reason and never download data; the remaining tests use
only the reviewed bundle, tracked contract and temporary synthetic corruptions.

### Measured local verification (November 11, 2026)

- 47 Decision Studio tests passed (zero failures/skips), plus 32 existing
  station-dataset tests passed. These are engineering checks, not new benchmarks.
- Actual local HTTP root/health returned 200; both workspaces rendered over real
  Streamlit WebSockets, including two Plotly charts. The owned server stopped.
- The final exporter completed in about 2.05 seconds, with 146 MiB sampled owned
  RSS. The final app test run took 8.28 seconds; sampled owned RSS peaked at
  225 MiB with no observed host swap growth during that run.
- The local HTTP/WebSocket check took about 2.64 seconds; the server's recorded
  peak RSS was 173 MiB. The supervising check sampled 233 MiB combined owned RSS.
- Existing host warning-level memory pressure remained. An earlier test attempt
  stopped when system-wide swap growth crossed its 256 MiB guard; attribution
  to this app was not established. Arrow-side filtering replaced conversion of
  all source rows into Python objects, preserving output bytes while reducing
  exporter work. Final monitored runs stayed within their resource guards.

These small samples do not establish hosting capacity or safety under concurrent
public traffic. No heavy ML runtime, GCP call or external LLM call was exercised.

### Milestone 2 UI refinement

The initial CSS painted the app/sidebar light but left Streamlit's native theme
unconfigured. A browser/system dark preference could therefore supply white
widget text, a dark header and default red accents over the custom light surfaces.
Global heading/link rules also bypassed component-specific contrast decisions.

The project `.streamlit/config.toml` now defines native main/sidebar colours.
CityPulse intentionally uses **one light canvas / navy sidebar palette** rather
than two different visual designs: light and dark theme sections both specify
the same readable main surfaces and text. Native theme tokens cover widgets,
filters and table headers; scoped CSS handles spacing, KPI cards, header surface,
selected navigation, hover/focus and narrow-width layout. No Emotion-generated
class selectors are used. Native collapse/expand controls remain available.
These project-level visual settings also apply to the standalone SQLPilot UI;
its implementation and security behavior are unchanged.

Main palette: navy `#102D40`, teal `#167E87`, canvas `#F3F7FA`, white cards and
text `#21384B`. Sidebar body/active-item text is pure white. White against teal
has a 4.81:1 contrast ratio; the earlier off-white gave only 4.46:1. Sidebar links
use a lighter teal. Captions have explicit readable foreground colours. Plotly
retains model colours on white surfaces with dark labels, additional room for
bar values/legends, and no automatic Streamlit chart recolouring.

Refinement verification: **50 tests passed, zero failures/skips**. Tests cover
native light/dark configuration, key text-pair contrast, scoped focus/selection
styles, unchanged chart values, both workspaces and the existing artifact/metric
checks. Actual loopback HTTP root/health returned 200; both pages and two Plotly
charts rendered over WebSockets, and the server's emitted light/dark/sidebar
theme configuration matched the intended tokens. The owned server stopped.

An isolated existing Chrome was attempted for desktop/mobile screenshots under
light/dark system preferences. Combined owned RSS exceeded the 512 MiB guard
(about 788 MiB observed), so the guard stopped it before screenshots were saved.
It was not forced or retried. **Pixel-level desktop/mobile, hover, keyboard-focus
and collapse/expand visual sign-off remain manual review items.** No screenshot
verification is claimed. The later lightweight test and HTTP checks stayed within
their resource guards; host memory pressure remained warning-level.

After pulling theme changes, stop your existing local Streamlit process with
Ctrl-C and restart it using the command above. No Mac/browser restart is needed.
The approved Milestone 2 foundation was committed locally as
`8763cc97794918b4170dd79b06527a704c77eb8b`. Milestone 3 implementation is saved
locally; its browser UI review remains pending.

Milestone 3 implements Forecast Explorer using the prepared sequences, as
described below. SQLPilot later needs **newly executed, labelled offline examples**:
its original live report has statuses but no saved generated SQL/results. Never
invent those traces or substitute synthetic fixture answers for real sample data.
ExperimentLab's later workspace will reuse saved synthetic results and caveats.

No deployment workflow was edited. No push/merge is part of this milestone. New
Decision Studio files are outside the existing Cloud Run deployment path filters;
do not modify old forecasting/feature/API paths merely to build presentation code.
Public hosting, any workflow changes and GCP retirement require separate review.

## Milestone 3: historical Forecast Explorer

**Status: implemented and automatically tested; browser UI review is pending.**
Real HTTP/WebSocket checks passed, but they do not replace visual inspection of
the rendered Forecast Explorer. Review the browser UI before further development.

Select an exact station ID (nine available), then an available NYC-midnight
forecast origin (three per station). The widget stores the **UTC ISO origin**;
formatting it in NYC time does not change the key. Select any of the six methods,
switch timestamp labels between NYC offsets and UTC, and optionally display the
saved TFT q10-q90 band. Empty method selection displays actuals only, with a
warning and no fabricated model scores. SQLPilot/ExperimentLab remain placeholders.

`load_bundle()` verifies the same unchanged four-file bundle and returns its
already-validated forecast rows. `available_origins()` and `select_sequence()`
filter exact station/origin identities, order horizons 1-24 and require exactly
one complete trusted sequence. Unknown stations/origins, missing/duplicate hours,
untrusted counts and invalid predictions fail closed. There is no interpolation,
zero-filling, training, new prediction generation or new test-set inference.

For example, on November 2, 2025:

```text
origin UTC: 2025-11-02T04:00:00Z = 00:00 EDT
horizon 2: 2025-11-02T05:00:00Z = 01:00-04:00 (EDT)
horizon 3: 2025-11-02T06:00:00Z = 01:00-05:00 (EST)
horizon 24: 2025-11-03T03:00:00Z = 22:00-05:00 (EST)
```

The chart plots UTC instants on a date axis; explicit labels and tooltips show NYC
offsets or UTC instead of the viewer's browser timezone. Tooltips include horizon,
actual, prediction and signed error. Lower/upper TFT traces form the shaded band;
q50 is the TFT point forecast. Actuals and methods have distinct colours/styles.
The expanded table retains every exact UTC/NYC target label and selected values.

`window_errors()` calculates prediction minus trusted actual for each horizon.
Positive error means overprediction; negative error means underprediction. MAE
is the mean absolute error over these **24** observations, in starts/hour. The
table reports mean signed error, counts of over/under/exact predictions and the
largest absolute miss (earliest horizon wins ties). A horizon plot shows signed
errors around zero. Large errors describe discrepancies, not their causes.

```python
from citypulse_ai.decision_studio.artifacts import load_bundle
from citypulse_ai.decision_studio.forecasts import select_sequence, window_errors

bundle = load_bundle()
station = bundle["manifest"]["subset"]["station_ids"][0]
sequence = select_sequence(bundle["forecasts"], station, "2025-11-02T04:00:00Z")
summary, errors = window_errors(sequence, ["xgboost", "pytorch-tft"])
# 24 certified actuals per method; official 220,800-target metrics are unchanged.
```

The quantiles estimate the conditional distribution of accepted recorded starts:
q10/q50/q90 are the predicted 10th/50th/90th percentiles. The q10-q90 span has
nominal 80% probability mass **if calibrated**. It is not a confidence interval
for an estimated mean/parameter, nor a guarantee that 80% of any selected 24
observations will fall inside. The fixed nonnegative postprocessing is retained.

The app is interactive because widgets select saved data and recompute small
descriptive summaries; trained models are never loaded. The official full-test
scores remain separate in Model Comparison. No ranking/generalization claim may
be inferred from a demonstration window. The 27 windows are not all 9,200 test
station-origin pairs, and station IDs have no invented names or coordinates.

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-decision-studio/bin/python -m pytest \
  tests/test_decision_studio.py tests/test_decision_studio_explorer.py -q
.venv-decision-studio/bin/python scripts/verify_decision_studio.py
```

The real loopback smoke check now renders all three implemented workspaces,
checks 24 distinct forecast timestamps and the TFT band, and stops its owned
server. No heavy browser automation is necessary. Pixel-level visual review and
public hosting remain separate; no deployment is claimed.

Milestone 3 local verification: **75 tests passed, zero failures/skips** (49
existing Decision Studio checks plus 26 Explorer checks), in 14.94 seconds.
Coverage includes all 27 saved windows, invalid/missing selections, malformed
sequences, signed-error mathematics, DST offsets, quantile ordering, chart data,
widget interactions and all completed workspaces. The actual HTTP/WebSocket
smoke check returned 200 for root/health and rendered all three workspaces, with
24 distinct forecast instants and a TFT band. The owned server stopped cleanly.
The final test run sampled about 226 MiB owned peak RSS; the smoke check recorded
about 173 MiB server peak RSS. Existing host memory pressure remained warning-level,
but the monitored runs stayed within unchanged resource guards. No browser
automation was launched for this milestone.
