# CityPulse Decision Studio - offline presentation, Milestones 2-5

This is a separate Streamlit + Plotly presentation of **saved** CityPulse results.
All five workspaces are implemented: Overview, Forecast Explorer, Model Comparison,
curated SQLPilot, and ExperimentLab & Methodology. Milestone 5 awaits browser visual review.
It is not publicly hosted yet and does not perform live forecasts, LLM calls or
statistical experiments. SQLPilot executes only fixed local reference queries.
GCP backup/retirement is a separate task.

The existing six-model API, models, original SQLPilot, ExperimentLab, Terraform and
cloud workflows are unchanged. No GCP credentials, cloud SDK, PyTorch or TensorFlow are
needed by this app. No artifact downloads occur at startup.

## Run locally

Use Python 3.14 (the locally tested interpreter); use the same version for later
Community Cloud deployment. Dependencies live beside the app entrypoint to avoid
installing the repository's forecasting extras. The runtime requires only the
six packages in `apps/decision_studio/requirements.txt` and their dependencies.

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
| `ui.py`, `workspaces.py`, `app.py` | Shared presentation, five-workspace routing and overview/comparison |
| `experiment_data.py`, `experiment_charts.py`, `experiment_workspace.py` | Saved synthetic evidence validation, statistical visualizations and Methodology UI |
| `scripts/build_experiment_demo.py` | Create-only, byte-identical export of existing synthetic research JSON |
| `forecasts.py`, `explorer.py` | Exact saved sequence selection, descriptive errors and Forecast Explorer UI |
| `sql_demo.py`, `sql_demo_worker.py`, `sql_workspace.py` | Fixed query IDs, bounded read-only worker and curated SQL UI |
| `sql_export.py`, `scripts/build_sqlpilot_demo.py` | Create-only projection of the actual analytical sample and reference verification |
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

AppTest checks all five completed workspaces, controls, empty selection and
fail-closed missing-artifact behavior. The loopback smoke test checks
HTTP health/root and renders all five pages over real Streamlit WebSockets, then
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
Milestone 5 below now presents the saved synthetic results and caveats.

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
warning and no fabricated model scores. SQLPilot and ExperimentLab are implemented
in their own subsequent milestones below.

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

## Milestone 4: curated offline SQLPilot

SQLPilot now executes **genuine local read-only SQL** against a projection of the
existing analytical sample. It does not call Groq, run LangGraph/MCP transport,
accept free-form SQL/questions, or connect to GCP. The original agent remains
unchanged and is explained separately in the workspace.

Install the two newly required, existing SQLPilot-pinned dependencies in the
isolated environment, then restart the app using the normal command above:

```sh
.venv-decision-studio/bin/python -m pip install --only-binary=:all: \
  -r apps/decision_studio/requirements.txt
```

No model or cloud SDK is installed. DuckDB 1.5.6 is pinned because the existing
native AST validator deliberately fails closed on unknown parser versions.
`pytz` supports DuckDB's timezone-aware Python retrieval.

### Actual sample and verified export

The original source is
`outputs/historical/samples/2024-01-n1000/citypulse.duckdb`, not the station100
forecasting dataset. Its ETL report records the first 1,000 rows from each of two
January 2024 archive members: 2,000 sampled rows, 1,998 accepted, 2 rejected for
missing start stations, and 1,931 sparse station-hour aggregates. Source archive
months are not exact NYC-local periods: the sample's UTC starts range from
2024-01-01 04:57:44.743 to 2024-02-01 04:35:58.275, corresponding to December 31
through January 31 in NYC. It does not establish complete monthly totals,
physical rental demand, publisher completeness, station availability or genuine
zero counts for absent hours.

`sql_export.py` creates a **new** physical database containing only:

- `hourly_rides`: `start_station_id VARCHAR`, `hour_start TIMESTAMPTZ`,
  `ride_count BIGINT` (1,931 rows).
- `validated_trips`: `started_at/ended_at TIMESTAMPTZ`,
  `start_station_id/end_station_id VARCHAR`, `duration_seconds DOUBLE` (1,998 rows).

Every projected value/type matches the original database. Ride IDs, source paths,
source row numbers, UTC-nanosecond helper columns and the unrelated February view
are excluded. The original database and report are not overwritten. The manifest
records their hashes, projected schema/counts, original CSV member names/hashes,
transformation, and output hashes. It links to retained source evidence; this
milestone does not re-ingest CSVs or independently establish publisher completeness.

Files under `apps/decision_studio/sqlpilot_data/` are `sample.duckdb`,
`queries.json`, `evaluation.json`, and `manifest.json`. They are independent of
the unchanged forecast bundle. Only the exact small projected database is exempted
from `.gitignore`; arbitrary DuckDB databases remain ignored.

```sh
# Optional reproduction with existing local research inputs only. Create-only:
# use a NEW destination; never restore/download sources automatically.
PYTHONPATH=src .venv-decision-studio/bin/python scripts/build_sqlpilot_demo.py \
  --output /path/to/new-sql-demo
```

DuckDB physical file encodings may differ across rebuilds; correctness means
exact projected values/schema and matching query results, not promised
byte-identical reconstructed databases. The shipped files are checksum verified.

### Fixed SQL and real results

The six reference queries exactly match the existing frozen question fixture.
Its `fixture_expected_rows` belong to a **synthetic test fixture** and are never
used as real-sample answers. The exporter ran each query on the original read-only
sample and the projected sample, comparing columns and rows (tight tolerance for
floating results). The public worker compares fresh outputs against those newly
verified references too.

| ID | Real analytical-sample result |
|---|---|
| `total` | 1,998 accepted sampled starts |
| `top_station` | Exact station ID `5779.11`, 54 sampled starts; deterministic ID tie-break |
| `duration` | Mean duration 692.6897217217227 seconds |
| `weekday` | Sunday=0: `0:170, 1:305, 2:280, 3:368, 4:312, 5:301, 6:262`; total 1,998 |
| `station_filter` | `NULL`, because exact ID `001` has no matching hourly rows; not converted to zero |
| `join` | 1,998 accepted sampled trips match exact start station and UTC start hour |

The selector shows the corresponding question and **curated reference SQL**.
Pressing Run executes it; there is no background generation. Source definitions
and all six exact queries are in `decision_studio/sql_demo.py`. For example:

```sql
SELECT sum(ride_count) AS starts FROM hourly_rides
```

returns `[[1998]]`. The weekday query explicitly converts `hour_start` to
`America/New_York` before `date_part('dow', ...)`. Bar charts show true weekday
counts/top-station results, without extrapolating a citywide ranking. Stale
results disappear when another question is selected. Empty rows, SQL NULL and
execution errors have distinct messages and no invented fallback answers.

### Runtime safety and architecture

Public path: **approved query ID -> fixed SQL -> original AST policy -> isolated
read-only DuckDB worker -> checked structured result**. No anonymous SQL editor,
file-path picker, upload or LLM credential is exposed. The fixed ID registry
rejects unknown IDs and query changes even if someone manually re-signs a local
query manifest. Local bundle checksums detect corruption; they are not signatures.
The reviewed Git checkout/database are trusted inputs, not arbitrary visitor files.

`sql_demo_worker.py` reuses `citypulse_ai.sqlpilot.core.worker`: single SELECT,
physical-table/column/function allowlists, no external access/extensions, readonly
connection, 64 MB engine memory, one thread, no spill, 100 output rows and 128 KiB
response cap. A parent wall timeout and worker alarm bound execution to five
seconds; interrupted/timed-out children are killed and reaped, without retries.
The worker resolves the fixed project source path itself and receives a minimal
nonsecret environment. It works without an editable package install or inherited
`PYTHONPATH`. Engine memory is not total RSS; native-engine vulnerabilities remain
outside these application-layer guardrails.

Original agent path: **question -> LangGraph -> hosted LLM -> MCP tools -> guarded
DuckDB -> grounded explanation**, with return/correction edges. The code remains
under `src/citypulse_ai/sqlpilot/`:

- `graph.py` defines state and conditional routing: discover schema, generate SQL
  or clarify, execute, one correction if appropriate, then explain or fail. Normal
  Python could implement this; the graph makes state, budgets and branches explicit.
  No persistent graph checkpointing is claimed by this demonstration.
- `providers.py` sends approved question/schema to Groq and parses candidate SQL
  or clarification. It does not execute SQL or send returned query rows.
- `client.py` uses the official MCP SDK to initialize the local stdio server,
  list named tools and invoke `sqlpilot_schema`/`sqlpilot_query` over JSON-RPC.
  `mcp_server.py` implements the bounded local server/tool subset. MCP standardizes
  tool communication; it is neither the LLM nor the security policy by itself.
- `core.py` validates and executes, using the same restrictions as this demo.
  The original graph permits at most two generation attempts and three tool calls.

The original live report is preserved semantically in `evaluation.json`, with its
source hash recorded. It reports 6/9 correct outcomes, 4/6 correct answerable
results, 5/6 successful answerable executions and median latency 1.946334041 s.
The incorrect top-station result, failed weekday queries and missed clarification
are shown as recorded. Exact generated SQL/results were not saved, so their
precise causes cannot be reconstructed. These new curated outputs must never be
labelled Groq generations, repaired original traces or improved live accuracy.

Offline execution makes the public app inspectable, reproducible and usable
without paid APIs, keys, model weights or GCP. It demonstrates the safe execution
substrate; it is not new natural-language-to-SQL accuracy evidence.

### Local verification

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-decision-studio/bin/python -m pytest tests/test_decision_studio.py \
  tests/test_decision_studio_explorer.py tests/test_decision_studio_sql.py \
  tests/test_sqlpilot.py -q
.venv-decision-studio/bin/python scripts/verify_decision_studio.py
```

The original-source projection/rebuild check explicitly skips with a reason if
retained local research inputs are unavailable; it never downloads them. Other
SQL checks use the small public bundle. Synthetic corruptions/test doubles verify
failure handling only and are not presented as real or live LLM results.

Milestone 4 measured verification: **155 tests passed, zero failures/skips** across
Decision Studio, Forecast Explorer, curated SQLPilot and the existing SQLPilot
core (41.45 seconds). The pre-implementation Explorer regression also passed all
26 checks. Six real curated worker queries matched the original-sample references;
source projection, readonly rejection, external-access blocking, NULL/empty/error
handling, timeout cleanup, stale selection, malformed bundles and all four
implemented workspaces were tested. No genuine LLM evaluation was repeated.

The final real HTTP/WebSocket check returned 200 for root/health, rendered all
four workspaces, clicked the SQL button and decoded its actual Arrow result
`[{"starts":1998}]`. The owned server stopped cleanly. Its peak RSS was about
173 MiB; the final test run sampled about 332 MiB owned peak RSS. Runs stayed
within existing guards despite warning-level host memory pressure. No browser
automation was attempted; the new SQLPilot browser UI review and public hosting
remain pending. Dependency compatibility, source/output hashes, syntax, local
Markdown links and whitespace checks passed. After the user's local functionality
and visual approval, Milestone 4 was committed locally as
`96d380d3354902f59cc93ef10937a00d51d7c440` with 155 tests passing again. No push or
merge was performed.

## Milestone 5: ExperimentLab & Methodology

**Implemented offline; browser visual review and public hosting remain pending.**
The original ExperimentLab engine, report, research files, forecasting results and
SQLPilot bundle are preserved. This workspace displays saved evidence; it does not
run a new experiment. The approved navy/teal/light-canvas theme is reused.

### Frozen study and provenance

The canonical source is
`outputs/experimentlab/release-20261030-final/experiment.json` (23,654 bytes), SHA-256
`fefe5b941ed2a7618b221cf51db9a3c7fccc195cb36ac0016f5fac01135ba527`.
Its checksum matches the original `checksums.json`; the original HTML matches
the tracked [report](experimentlab_report.html), SHA-256
`fd6cf75c44b7237b76589ba01800807ac234c9a62cd4d6eaf641bf40b8c1ce4b`.
`apps/decision_studio/experiment_data/experiment.json` is a **byte-identical** copy.
The adjacent manifest records relative source identities, hashes, sizes, copy
semantics and limitations. The small synthetic records are safe demonstration
artifacts, not a historical-data export. Checksums establish consistency with the
repository's receipts, not an external authenticity signature.

The create-only exporter can reproduce the bundle **only when the retained
canonical local inputs exist**; a fresh clone uses the bundled files directly:

```sh
PYTHONPATH=src .venv-decision-studio/bin/python scripts/build_experiment_demo.py \
  --output outputs/decision-studio-experiment-copy
```

The loader enforces size limits, fixed names, source identities, SHA-256,
non-symlink files, finite results, balanced unique synthetic station units,
disjoint calibration/baseline/post periods, and saved effect/SE/df algebra.
Missing/corrupt evidence produces a safe error, not invented results. The page
loads independently of the forecasting and SQL bundles. The HTML download is
enabled only when its checksum matches.

### What the study tests

A hypothetical redistribution intervention adds +3 starts per station/day in
the simulated treatment arm. Outcome totals come from a fixed-seed synthetic
Poisson process with heterogeneous, stable latent station rates. `seed=2026`
governs simulation; a separate `assignment_seed=42` shuffles 100 SIM station IDs,
assigning 50 to treatment and 50 to control. These are **not** the forecasting
station IDs. The independent unit is a station; the outcome is its mean starts
over 28 local-calendar-days. Days are not treated as 2,800 independent units.
The estimand is the station-average intention-to-treat difference in post means.

| Saved estimator | Effect | SE | 95% Welch CI | Two-sided p |
|---|---:|---:|---|---:|
| Unadjusted sensitivity | +2.8086 | 6.5002 | [-10.0947, 15.7118] | 0.66666 |
| Pre-calibrated CUPED primary | +2.7531 | 0.5208 | [1.7190, 3.7871] | 8.03e-7 |

Welch estimates `mean(T)-mean(C)`, with
`SE = sqrt(s_T²/n_T + s_C²/n_C)` and Satterthwaite degrees of freedom. It allows
unequal arm variances; station independence, randomization, no interference and
a suitable sampling approximation still matter. The frequentist 95% procedure
would cover the fixed true effect in about 95% of repeated studies under its
assumptions. This does not assign 95% probability to the particular realized
interval and is not a prediction interval for individual future observations.

CUPED uses `Y_adjusted = Y - theta*(X-mean(X))`. Frozen theta **1.009154** was
fitted from two earlier nonoverlapping periods before assignment/post outcomes;
the later pre-assignment baseline is X. This app only transforms saved station
records using that coefficient. It never refits it. Earlier calibration unit
records and an exact original implementation hash are absent from the source
JSON: the saved coefficient/fit scope can be preserved, but its original fit
cannot be independently reconstructed here. The **99.358%** effect-estimator
variance reduction depends on this unusually stable synthetic process.

The primary conclusion remains **statistically positive but practical threshold
uncertain**: the interval excludes zero but includes effects smaller than the
predeclared +2 threshold. Neither switching the display estimator nor choosing
the narrower interval changes the pre-specified primary analysis.

### Meaningful controls and planning

- Select CUPED or unadjusted effects; show one or both saved uncertainty intervals.
- Filter treatment/control station records and compare observed versus adjusted
  station means. Adjusted means are statistical quantities, not new ride counts.
- Select saved planning power or MDE; inspect the underlying values and assumptions.
- Expand confidence-interval explanations, pre-period provenance, saved Monte
  Carlo checks, and broader CityPulse methodology; download the original report.

At 50 stations/arm, alpha .05 and practical effect +2, pre-treatment SD planning
gives raw power **6.033%**, MDE **18.7193**, and **4,296 stations/arm** for 80% power;
CUPED gives **98.558%**, MDE **1.3516**, and **24 stations/arm**. These are approximate
equal-variance noncentral-t planning results from frozen pre-treatment SDs, not
post-hoc observed power. Increasing independent sample size generally improves
power and reduces the minimum detectable effect. Saved simulation checks have
only 200 repetitions/scenario; 200/200 detections does not prove perfect power.
No new scenarios, sliders, random draws or simulation batches are introduced.

Methodology expanders connect historical ETL, accepted-record label policy,
train-only station selection/scalers, UTC/DST-aware splits, six forecasting
methods, SQLPilot's LangGraph/MCP architecture and synthetic ExperimentLab.
FastAPI/Docker, Terraform, WIF and GitHub Actions/Cloud Run are described as
**historical verified engineering evidence**, not a claim that GCP is currently
active. The public-facing offline app checks no cloud services.

### Implementation and verification commands

`experiment_data.py` validates evidence and computes only display transformations
and algebra. `experiment_charts.py` plots saved estimates/error bars and planning
values. `experiment_workspace.py` binds selectors to those views. No model,
SciPy or cloud runtime is required; the unchanged six-package runtime file remains
sufficient. SciPy is a **development-only** dependency for independently checking
saved inference against the research engine; no simulations are called.

```sh
.venv-decision-studio/bin/python -m pip install --only-binary=:all: \
  -r apps/decision_studio/requirements-dev.txt
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-decision-studio/bin/python -m pytest tests/test_decision_studio.py \
  tests/test_decision_studio_explorer.py tests/test_decision_studio_sql.py \
  tests/test_decision_studio_experiment.py tests/test_sqlpilot.py -q
PYTHONPATH=src .venv-decision-studio/bin/python -m pytest tests/test_experimentlab.py \
  -k 'welch_matches or power_null or inference_rejects' -q
.venv-decision-studio/bin/python scripts/verify_decision_studio.py
git diff --check
```

Tests compare saved station records with independent Welch equations and existing
SciPy inference, reproduce planning from frozen SDs, check assignment and CUPED
transformations, reject damaged provenance, enforce create-only exports, and
exercise selectors and all five workspaces. The selected original research tests
exclude simulation/regeneration cases intentionally. Real HTTP/WebSocket
verification renders the actual local app and always stops its server; this is
not a pixel-level browser review. Milestone 5 stays uncommitted for visual review;
no public deployment, push, merge, GCP operation or external LLM call is included.

Measured Milestone 5 verification: **187 tests passed, zero failures/skips** in
44.15 seconds across the four Decision Studio suites plus the original SQLPilot
core. **Three original statistical tests passed, 16 simulation-related tests
were deliberately deselected**. The combined test process tree sampled about
334 MiB peak RSS; host-wide swap use rose about 45 MiB during that run, with no
claim that the test caused it. Warning-level host memory pressure remained
unchanged and no guard fired. Actual HTTP root/health returned 200 and all five
workspaces rendered over Streamlit WebSockets. Saved CUPED effect/CI/p-value,
24 forecast points and the real curated SQL result were verified. The server
stopped cleanly after 3.14 seconds, with 181 MiB peak server RSS (264 MiB sampled
combined verifier/server RSS); no host swap growth was observed during that check.
The synthetic bundle adds only **25,049 bytes**. No pixel-level browser sign-off
or current cloud availability is claimed. Original source/research checksums,
protected bundles, dependency compatibility, syntax, local links and whitespace
checks passed.
