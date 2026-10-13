# CityPulse AI

CityPulse AI is a portfolio-grade urban mobility demand forecasting and AI
analytics platform built around NYC Citi Bike trip data and historical weather.
The goal is to demonstrate an end-to-end workflow from validated source data to
reproducible forecasts, interactive analytics, and statistical experimentation.

**Phase 1 is complete:** data engineering, ETL, data-quality review, and initial EDA.
Phase 0, Phase 1.1, and the Phase 1.3 real-data ETL adaptation are implemented:
a Python scaffold and a chunked multi-file CSV pipeline with validation, global
duplicate checks, hourly counts, Parquet output, and a DuckDB table.
Initial EDA in Jupyter includes saved ride counts, missing-value and
duration checks, and an hourly-demand visualization for February 2025. Raw data
stays local; only synthetic fixtures are committed. Phase 2.1 now prepares calendar grids, coverage audits, and chronological splits
from the reviewed warehouse. Weather, model training, APIs, and deployment remain
future work.

## Architecture overview

The planned data flow is:

```text
Citi Bike trips + historical weather
                |
       ETL and validation
                |
        DuckDB warehouse
                |
      Time-aware features
                |
 Forecasting -> evaluation -> MLflow tracking
                |
     FastAPI + Streamlit dashboards

SQLPilot -------> governed warehouse queries
ExperimentLab --> statistical analysis and A/B testing
```

DuckDB will provide the initial SQL warehouse. Forecasting will start with simple
baselines before adding PyTorch and TensorFlow models. SQLPilot will eventually
use a fine-tuned text-to-SQL model with MCP, LangGraph, and vLLM. Docker, CI/CD,
Kubernetes, and cloud hosting belong to later deployment phases. The original
full scope remains unchanged: ARIMA, XGBoost, PyTorch, TensorFlow, LSTM, TFT,
MLflow and model monitoring, QLoRA fine-tuning on H100, LangGraph/MCP/vLLM,
FastAPI, Docker, Kubernetes, GitHub Actions, AWS and Google Cloud integration,
and statistical analysis, A/B testing, and reporting. We implement these
incrementally with measured experiments; this is not a reduced MVP.

## Project structure

```text
.
├── AGENTS.md                 # Guidance for future coding agents
├── LICENSE                   # Existing project license
├── README.md                 # Objective, setup, architecture, and roadmap
├── pyproject.toml            # Packaging, dependencies, and pytest configuration
├── notebooks/
│   ├── 01_data_exploration.ipynb # Existing EDA, preserved
│   └── 02_forecasting_data_preparation.ipynb # Phase 2.1 walkthrough and plots
├── csv_files/                # Local raw CSVs; ignored by Git
├── src/
│   └── citypulse_ai/
│       ├── __init__.py       # Root Python package
│       ├── etl/              # schema.py, trips.py, pipeline.py, cli.py
│       ├── features/         # Time-aware feature preparation
│       ├── forecasting/      # Baselines and future forecasting models
│       ├── evaluation/       # Backtesting and forecast metrics
│       ├── experiments/      # ExperimentLab statistical analysis
│       └── sqlpilot/         # Future text-to-SQL agent
└── tests/
    ├── fixtures/             # Synthetic valid, invalid, empty, and DST CSVs
    ├── test_etl.py            # Validation, aggregation, persistence, and CLI tests
    └── test_imports.py        # Package discovery and import smoke tests
```

The ETL package contains the implemented trip pipeline, and
`features/reporting.py` prepares the reviewed February reporting dataset.
`features/preparation.py` creates forecasting grids and coverage audits, with a
module CLI in `features/cli.py`. The forecasting, evaluation, experiments, and
SQLPilot subpackages remain placeholders. The `src/` layout separates
importable code from repository files.

## Local setup and tests

Use Python 3.11 or newer. From the repository root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The initial runtime dependencies are pandas for tabular transformations, PyArrow
for Parquet storage, and DuckDB for SQL analytics. `pytz` supports DuckDB's Python
retrieval of timezone-aware SQL timestamps. The `dev` extra installs pytest.
ML, serving, and orchestration dependencies will be added when their phases begin.

Pytest discovers tests under `tests/` and includes `src/` on its import path. If
pytest is unavailable, the scaffold smoke tests can also run without installing
third-party dependencies on macOS or Linux:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The unittest fallback runs only the scaffold smoke tests. Use pytest for the ETL
tests, including synthetic DST cases, duplicate handling, Parquet/DuckDB outputs,
and CLI behavior.

## Trip ETL (Phases 1.1 and 1.3)

### Raw data contract and rejection policy

The CSV must have a header containing these columns. Extra columns are ignored.
Identifiers are read as strings to retain leading zeros; whitespace is stripped.
Literal strings such as `NA` remain identifiers rather than inferred nulls.

| Column | Contract |
| --- | --- |
| `ride_id` | Required, nonblank trip identifier; unique across selected files |
| `started_at` | Required ISO-8601 timestamp with seconds |
| `ended_at` | Required ISO-8601 timestamp with seconds; later than the start |
| `start_station_id` | Required, nonblank string with valid ID syntax |
| `end_station_id` | Column required; blank values accepted for ride-start demand |

The real February 2025 CSVs have 13 columns, including ride type, station names,
coordinates, and membership type. The ETL reads only these five columns; it does
not coerce decimal-looking IDs such as `3620.02` into floating-point numbers.
Start IDs must contain alphanumeric segments optionally separated by `.`, `_`,
or `-`. Whitespace is stripped. Blank IDs are missing; case-insensitive `NA`,
`NaN`, `NULL`, `None`, `N/A`, and `<NA>` markers are invalid. This checks syntax,
not membership in an authoritative station catalog. No four-digit numeric
restriction is imposed, preserving leading zeros and alphanumeric IDs.

The reviewed exception allowlist contains exactly **`6569.09_`**. February source
data shows this ID as W 35 St & 9 Ave, approximately 33 metres from the location
recorded for `6569.09`. This supports retaining its 43 otherwise-valid starts,
but does not prove the IDs are identical. Preserve `6569.09_` as a distinct raw
identifier; do not strip underscores or merge it with `6569.09`. Other malformed
IDs remain rejected, and the exception does not bypass timestamp, duration,
missing-value, or duplicate checks. Matching follows the existing whitespace
cleanup, and the allowlist is recorded in each run's validation report.

An end station is not needed to count where a trip started, so missing end IDs
are accepted and counted in the report. A valid `ended_at` remains required to
check elapsed duration. Positive rides over 24 hours are retained for now; an
upper cutoff must be a documented domain decision, not inferred from EDA alone.

Missing columns, a headerless file, or malformed CSV syntax fail the run. A
header-only CSV succeeds and creates typed empty outputs. Rows with missing
values, invalid timestamps, or nonpositive/unrepresentable durations are rejected.
All occurrences of duplicate ride IDs are rejected, even when their other fields
match or one occurrence has another error. Duplicate checking follows whitespace
normalization and covers all selected rows across files and chunk boundaries.
Reasons accumulate per record. Repeating the same input path is an error.
No maximum plausible duration is imposed yet; that needs a domain decision.

Rejected rows retain the original five values, the absolute `source_file` path,
and a per-file `source_row` record number
(header is 1, first data record is 2), and semicolon-separated rejection reasons.
Record numbers are logical CSV records, not physical lines for multiline fields.
A completed run may contain rejected rows, including all rows, and exits zero;
inspect `report.json` for counts. Configuration/read failures exit nonzero.

### Timestamp and hourly-count policy

- Accept `YYYY-MM-DD HH:MM:SS` or a `T` separator, optional fractional seconds
  (up to nine digits, preserved at nanosecond precision), and optional `Z` or
  numeric `±HH:MM` offset.
- Interpret timestamps without an offset in **America/New_York**. The sampled
  real CSVs contain no offsets. The official
  [System Data page](https://citibikenyc.com/system-data) lists timestamp fields
  but does not explicitly specify their time zone, so NYC local time remains a
  documented assumption pending publisher confirmation. February 2025 uses
  UTC−05:00 under this policy. Do not treat that assumption as verified metadata.
- Reject ambiguous fall-back times and nonexistent spring-forward times without
  guessing, shifting, or inferring their UTC offset.
- Accept timestamps with explicit offsets as identified instants. Offsets are
  authoritative; their agreement with NYC civil time is not separately checked.
- Normalize accepted timestamps to UTC. Calculate durations from elapsed UTC
  time, so a spring-forward trip need not have the wall-clock duration.
- Count ride starts by `(start_station_id, hour_start)`, flooring in UTC. The
  half-open interval is `[hour_start, hour_start + 1 hour)`; `ride_count` is int64.
  UTC keeps repeated fall-back local hours distinct. Convert to NYC time for
  display, retaining the offset. Only observed station/hour bins are emitted;
  missing bins are not assumed to have zero demand.

### Run a reproducible local example

After installing the editable package (reinstall after changing entry points):

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
citypulse-etl --input tests/fixtures/trips_valid.csv --output-dir /tmp/citypulse-demo
```

The module entry point also works without reinstalling:

```bash
PYTHONPATH=src python -m citypulse_ai.etl.cli \
  --input tests/fixtures/trips_valid.csv --output-dir /tmp/citypulse-demo
python -m pytest
```

The flow is **CSV header checks → pandas chunks → temporary DuckDB staging →
global duplicate-ID check → chunk validation → chunk hourly counts → final SQL
aggregation → Parquet + DuckDB + audit outputs**. Functions are separated into
`schema.py` (contract), `trips.py` (pure ingestion/validation/aggregation steps),
`pipeline.py` (file persistence), and `cli.py` (argument handling).

Each run writes six data/report files in the selected output directory:

| Output | Purpose |
| --- | --- |
| `hourly_rides.parquet` | Station IDs, timezone-aware UTC hour starts, ride counts |
| `validated_trips.parquet` | Every accepted trip, UTC timestamps, raw station identifiers, duration, source provenance, and exact UTC nanosecond integers |
| `citypulse.duckdb` | Materialized canonical `validated_trips` and `hourly_rides` tables, plus derived `february_2025_hourly_rides` view |
| `february_2025_hourly_rides.parquet` | Observed station-hour counts inside the reviewed NYC February window |
| `rejected_trips.csv` | Rejected source records and their validation reasons |
| `report.json` | Per-file hashes/counts, run counts, rejection reasons, validation/timezone policies, timezone-data hash, processing limits, library versions |

Rerunning replaces these named files/table rather than appending trips. Other
DuckDB tables are preserved. Use a separate directory for each source/run you
want to retain. Writes across the outputs are not atomic; after an I/O
failure, rerun successfully before consuming them. Outputs are sorted by station
and UTC hour. The report contains no run clock, allowing identical reports for
the same input path/bytes and environment. Dependency versions are recorded,
but the project does not yet lock dependencies; preserve the environment for
reproduction across machines. A hash of the local timezone data is recorded.

Only a bounded pandas batch is validated at once (default 50,000 records).
Temporary disk-backed DuckDB tables hold raw ETL columns, global duplicate IDs,
and partial counts; final hourly rows are streamed to Parquet. This avoids a
Python set of millions of IDs and avoids retaining all accepted/rejected rows.
The default DuckDB working-memory limit is 512 MB with disk spill enabled;
it is not a hard limit on total Python/process memory. Allow space on the system
temporary filesystem for staging and spill. Validation uses a readable per-row
timestamp parser, so a full-file run will take longer than a sample.

Accepted trips are streamed to canonical Parquet while validating each batch;
they are not collected into one pandas frame. The DuckDB trip table is copied
from that artifact and does not depend on an external Parquet path. Parquet
preserves nanosecond UTC timestamps. DuckDB `TIMESTAMPTZ` represents microseconds,
so `started_at_utc_ns` and `ended_at_utc_ns` retain the exact original UTC instants
in SQL too. The real February timestamps have millisecond precision and therefore
are also represented exactly in the DuckDB timestamp columns.

### Reviewed February reporting window

Canonical tables retain every accepted trip/count, including January starts.
`features/reporting.py` creates a separate view and Parquet dataset with this
half-open **America/New_York** ride-start window:

```sql
SELECT * FROM hourly_rides
WHERE hour_start >= TIMESTAMPTZ '2025-02-01 00:00:00 America/New_York'
  AND hour_start <  TIMESTAMPTZ '2025-03-01 00:00:00 America/New_York';
```

Both boundaries are specified with their zone, independently of the DuckDB
session timezone. For this month they correspond to February 1 05:00 UTC
inclusive and March 1 05:00 UTC exclusive. The view preserves UTC hour values.
Since the boundaries align with full hours, filtering canonical hourly counts
is equivalent to filtering ride starts before aggregation. No zero-filling,
feature generation, or model training is performed.

The quality review found 273 genuine January 31 raw starts, all ending February 1;
their NYC-to-UTC-to-NYC timestamps round-trip exactly. They belong in canonical
data but not the February ride-start reporting window. Original data and previous
ETL outputs remain intact. Reprocess all three files together into a new directory:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m citypulse_ai.etl.cli \
  --input csv_files/202502-citibike-tripdata_1.csv \
          csv_files/202502-citibike-tripdata_2.csv \
          csv_files/202502-citibike-tripdata_3.csv \
  --chunk-size 50000 --memory-limit 512MB \
  --output-dir outputs/phase13-february-reviewed
```

Query the canonical and derived counts separately:

```sql
SELECT COUNT(*) FROM validated_trips;
SELECT SUM(ride_count) FROM hourly_rides;
SELECT SUM(ride_count) FROM february_2025_hourly_rides;
```

### Reviewed full-run results

The combined reviewed run processed all three raw files into
`outputs/phase13-february-reviewed/`, with 50,000-row chunks, a 512 MB DuckDB
staging limit, and no sample limit. It finished successfully in **345.20 seconds
(5 minutes 45 seconds)** with an empty error log.

| Verification | Actual result |
| --- | ---: |
| Input records | 2,031,257 |
| Canonical accepted trips | 2,030,542 |
| Rejected records | 715 |
| Missing start-station IDs | 673 |
| Other invalid start-station IDs | 42 |
| Duplicate ride IDs | 0 |
| Canonical station-hour rows | 598,855 |
| Canonical sum of ride counts | 2,030,542 |
| February station-hour rows | 598,630 |
| February ride starts | **2,030,269** |
| January 31 trips retained in canonical storage, excluded from the view | 273 |

Input equals accepted plus rejected, and canonical hourly counts equal the
canonical trip count. Both canonical and February hourly datasets have no
duplicate station-hour keys and match their Parquet exports exactly. The raw
IDs remain separate: `6569.09` has 1,354 accepted starts and `6569.09_` has 43.
No source timestamps changed; the SQL timestamp columns match their exact UTC
nanosecond integers for every real accepted trip. Source CSV hashes and all
22 snapshotted previous-output/notebook files were unchanged.

All **56 automated tests passed in 5.50 seconds** before reprocessing. Regression
tests cover the exact reviewed ID, rejection of other malformed IDs, duration
checks despite the exception, January/February/March boundaries, UTC boundary
instants, session-timezone independence, nanosecond preservation, empty outputs,
and count conservation. The new run's `report.json`, `verification.json`,
`run_summary.json`, and logs retain the measured results. Historical outputs
remain available; their older rejection counts describe the earlier policy.

The February data is now ready for a separate station-coverage and missing-hour
policy step. Do not interpret absent rows as zero demand yet. Weather integration,
forecasting features, temporal splits, and models remain later work.

### Verify a bounded real-data run first

```bash
source .venv/bin/activate
PYTHONPATH=src python -m citypulse_ai.etl.cli \
  --input csv_files/202502-citibike-tripdata_1.csv \
  --max-rows-per-file 10000 --chunk-size 2000 \
  --output-dir outputs/phase13-sample
```

This limits validation to the first 10,000 rows of that file. Hashing still reads
the entire source file as a byte stream to record provenance. Duplicate results
cover only selected rows; they do not prove uniqueness in the rest of the file
or across unselected files. `report.json` records the limit explicitly.

The initial Phase 1.3 sample run, before the reviewed exception, read **10,000 rows**, accepted **9,995**, and
rejected **5** for missing start-station IDs. It accepted **26** rows with blank
end-station IDs and produced **9,590** observed station/hour bins. No duplicate
IDs were found within the sample. These are sample results, not monthly totals.

After inspecting the sample audit and checking count conservation, run the first
complete CSV in a separate output directory:

```bash
PYTHONPATH=src python -m citypulse_ai.etl.cli \
  --input csv_files/202502-citibike-tripdata_1.csv \
  --chunk-size 50000 --output-dir outputs/phase13-first-file
```

The API and CLI accept multiple paths in one run, e.g. `--input part1.csv
part2.csv part3.csv`. Inputs are sorted by resolved path for deterministic audits.
One combined run checks IDs across all files; three separate runs do not.
Reruns replace outputs rather than accumulate prior runs. No missing station/hour
bins are zero-filled.

Run these commands from the project root. `PYTHONPATH=src` explicitly imports
the current working-tree package; it also works if the local editable
installation is not being discovered by Python. The existing `.venv` has all
required dependencies, but its editable package path was not discovered during
this verification, so the real-data commands above use this explicit source path.

### Query the warehouse

```bash
python - <<'PY'
import duckdb
with duckdb.connect('/tmp/citypulse-demo/citypulse.duckdb', read_only=True) as db:
    db.execute("SET TimeZone = 'UTC'")
    rows = db.execute('''
        SELECT start_station_id, SUM(ride_count) AS ride_starts
        FROM hourly_rides
        GROUP BY start_station_id
        ORDER BY start_station_id
    ''').fetchall()
    for station, count in rows:
        print(station, count)
PY
```

For the supplied valid fixture, the result is:

```text
001 3
002 1
```

### Before using real Citi Bike data

Verify the chosen files' schema, timestamp precision/time zone, and missing-value
conventions against this contract. Decide how to handle stationless rides and
whether end-station IDs must remain mandatory; define plausible duration bounds
and acceptable rejection thresholds. Choose the source period and provenance
records, confirm usage terms, and review the rejection audit on a sample. For
multiple monthly files, verify global duplicates and output row-count accounting,
measure memory/disk usage on the first file, and plan incremental ingestion.
Define station coverage and zero-fill policies before
turning these observed counts into a forecasting dataset. Resolve ambiguous local
timestamps only with authoritative source information.

## Dataset source and local files

Use the official [Citi Bike System Data](https://citibikenyc.com/system-data)
page and its [trip-history download archive](https://s3.amazonaws.com/tripdata/index.html).
Review the data-use policy linked on that page before using the data. Larger
monthly archives contain multiple CSV parts; keep all parts for that month.

1. Open the official download archive and select the NYC February 2025 trip-data
   ZIP archive (the `202502` period, rather than a Jersey City `JC-` archive).
2. Download and extract the archive locally.
3. Create `csv_files/` in the project root and place the extracted CSV parts there.
   The current analysis uses these three files:

   ```text
   csv_files/202502-citibike-tripdata_1.csv
   csv_files/202502-citibike-tripdata_2.csv
   csv_files/202502-citibike-tripdata_3.csv
   ```

These raw CSVs are excluded from Git and must be downloaded separately after
cloning. `data/raw/`, generated Parquet/DuckDB files, virtual environments, caches,
and local secrets are also ignored. The small synthetic CSVs under
`tests/fixtures/` are committed so tests do not require real data.

## Jupyter EDA setup and saved progress

From the project root, use the same virtual environment as the ETL:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
python -m ipykernel install --sys-prefix --name citypulse-ai \
  --display-name 'CityPulse AI (.venv)'
python -m jupyterlab notebooks/01_data_exploration.ipynb
```

Select **CityPulse AI (.venv)**. The notebook locates `csv_files/` from either the
project root or `notebooks/`. Its first read uses `nrows=5`; later saved EDA cells
scan all three files with pandas chunks or DuckDB and retain their tables and
plot. Opening the notebook displays saved results without rerunning the analysis.
Rerun cells deliberately with your mentor.

The saved results report **2,031,257 rides** across the three parts, no extra
duplicate ride IDs, and **315 rides over 24 hours** under the notebook's current
duration query. These are recorded EDA outputs, not forecasting benchmarks, and
were not recomputed during the repository cleanup. The notebook's exploratory
timestamp queries use naive timestamps; the ETL's explicit UTC/DST validation
policy remains the production contract. Raw EDA counts can differ from validated
ETL counts.

Phase 1 reconciled the exploratory counts with validated ETL counts and reviewed
reporting boundaries and station IDs. Phase 2.1 now documents coverage separately
from station availability. Cross-file duplicate checks are implemented; incremental
ingestion remains future ETL work.

## Phased roadmap

1. **Scaffold (complete):** package layout, dependency metadata, agent guidance,
   documentation, and import smoke tests.
2. **Phase 1 — data engineering, ETL, data quality, and initial EDA (complete):**
   Phase 1.1 initial ETL and Phase 1.3 real
   schema adaptation, chunked multi-file ingestion, global duplicate checks,
   Parquet/DuckDB output, and fixture tests are complete. Complete-file and
   combined-file runs were verified, and reporting-window/station-ID decisions
   now have regression tests and a separate derived reporting dataset.
   Weather ingestion and aligned data contracts follow later.
3. **Features and baseline evaluation:** Phase 2.1 calendar features, coverage
   audits, and temporal splits are complete. Station availability verification,
   historical extension, weather features, seasonal-naive forecasts, and
   reproducible backtesting follow; no models have been trained.
4. **Deep learning:** add PyTorch and TensorFlow forecasting models and compare
   measured results against the baselines.
5. **Tracking and MLOps:** integrate MLflow, reproducible configurations, model
   artifacts, and pipeline monitoring.
6. **Application layer:** expose forecasts and analytics through FastAPI and
   Streamlit.
7. **SQLPilot:** prepare text-to-SQL training/evaluation data, fine-tune a model,
   and integrate MCP, LangGraph, and vLLM with query safeguards.
8. **ExperimentLab:** add A/B test design, statistical analysis, uncertainty
   estimates, and experiment reporting.
9. **Deployment:** containerize, add CI/CD, then introduce Kubernetes and cloud
   infrastructure as requirements justify them.

## Phase 2.1 — forecasting-data preparation

Preparation reads `february_2025_hourly_rides` from the reviewed DuckDB **read-only**.
It does not reread raw CSVs, alter ETL rules, modify the EDA notebook, or change
previous outputs. The canonical warehouse still preserves all 2,030,542 accepted
trips, including the 273 January 31 starts. The February input has **2,030,269**
starts in the half-open NYC calendar window `[2025-02-01, 2025-03-01)`.

### Coverage findings and policy

An absent station-hour is not sufficient evidence of zero demand or station
closure. Trips cannot tell us whether bikes were unavailable, a station was closed,
an identifier changed, or records were missing. The exported station target is
**realized validated ride starts**, not unconstrained demand.

- **Citywide:** create every UTC hour in the NYC-local February window. Only an
  explicit complete-source confirmation permits a missing city hour to become
  zero. The reviewed monthly manifest covers all three files without a sample
  limit, and reconciliation passed. February has all **672 observed citywide
  hours**, so **no citywide zero filling was needed**. Source completeness remains
  an engineering assumption about the supplied archive, not proof that every
  real-world trip was published.
- **Station default:** construct a 672-hour audit grid for every observed ID,
  retaining missing targets as nullable integers (`unknown`). Do not infer
  opening/closing dates from first/last trips. Preserve distinct string IDs,
  including the reviewed `6569.09_` exception.
- **Optional sensitivity rule:** `--station-zero-horizon N` explicitly assumes
  zero only within N hours after a past observed start at a training-seen station,
  and only with complete-source confirmation. It never back-fills before first
  activity, fills long silences beyond N, or fills stations first seen after
  training. Such cells carry `is_zero_filled=True` and
  `coverage_status=assumed_zero_recent_activity`; the original observed count
  stays null. A 24-hour horizon in the notebook illustrates daily recency, not a
  proven operating schedule or a selected optimal threshold. The default run
  below does **not** enable this rule. Any later policy choice must use training
  evidence and validation, never the reserved test set.
- **Activity audit:** observed episodes separated by at least 24 consecutive
  no-record hours and contiguous missing intervals describe recorded activity.
  Missing intervals are explicitly `no_record_unknown_operation`, not confirmed
  inactivity. Full-month first/last activity and coverage statistics are
  retrospective audit metadata, **not predictive features or cohort filters**.

Measured February results:

| Check | Actual result |
| --- | ---: |
| Stations observed during February | 2,237 |
| Stations seen in training (model cohort) | 2,224 |
| Stations first seen after training | 13 |
| Complete station-hour grid | 1,503,264 |
| Observed station-hours | 598,630 |
| Unknown station-hours (default policy) | 904,634 |
| Assumed-zero station-hours (default policy) | 0 |
| Duplicate station-hour keys | 0 |
| Citywide and station target sums | 2,030,269 each |
| No-record intervals | 185,766 |
| Intervals lasting at least 24 hours | 3,051 |
| Stations with at least one such interval | 830 |
| Observed activity episodes | 5,079 |

Median station coverage is **251 of 672 hours**; the range is 1–619, so no station
has starts in every hour. **83 stations** have fewer than 24 observed hours. The
longest no-record interval is **663 hours**. Of the missing station-hours, 49,401
occur before a station's first February observation, 8,067 after its last, and
847,166 between observations. These counts locate uncertainty; they cannot
identify confirmed operationally inactive periods. Historical station
availability, maintenance records, and an explicit identifier-history review
are needed for that distinction.

### Features, splits, and leakage prevention

Store timestamps in UTC and derive `hour` (0–23), `weekday` (Monday=0), and
`is_weekend` from **America/New_York** local time. This February has no DST change;
its window is `[2025-02-01 05:00Z, 2025-03-01 05:00Z)`. Do not hard-code 672 hours
for future months: build UTC instants between timezone-aware local boundaries.
The current preparation module deliberately validates February only.

| Split | NYC local half-open interval | Hours | Ride starts |
| --- | --- | ---: | ---: |
| Training | February 1 through February 21 | 480 | 1,309,480 |
| Validation | February 21 through February 25 | 96 | 286,095 |
| Test | February 25 through March 1 | 96 | 434,694 |

All stations share these chronological boundaries. `training_station` freezes
cohort membership using training observations only. The full grid also includes
13 later-seen IDs for audit; future station models must filter to
`training_station=True`, with a separate cold-start evaluation for those 13 IDs.
No scalers, imputers, lag features, models, or learned station thresholds are
fitted here. Calendar features are known ahead of time. Future target-dependent
features must be shifted and computed using past observations only. Keep unknown
labels masked: zero substitution or evaluating only busy observed hours can
bias performance. A nullable station grid is prepared, but station model training
still requires a reviewed label/availability policy; citywide targets are complete.

### Reproduce and learn

From the project root, with the existing environment:

```bash
PYTHONPATH=src .venv/bin/python -m citypulse_ai.features.cli \
  --warehouse outputs/phase13-february-reviewed/citypulse.duckdb \
  --output-dir outputs/phase21-forecasting \
  --confirm-complete-source

.venv/bin/python -m pytest -q
source .venv/bin/activate
python -m jupyterlab notebooks/02_forecasting_data_preparation.ipynb
```

Omit `--confirm-complete-source` for incomplete or unverified sources; absent
citywide hours then remain unknown. Use a different output directory for any
optional station-zero sensitivity run. No extra runtime dependencies were added.
If needed, install development/notebook dependencies with
`python -m pip install -e '.[dev,notebooks]'`, then register the environment with
`python -m ipykernel install --sys-prefix --name citypulse-ai --display-name 'CityPulse AI (.venv)'`.

Generated local files under `outputs/phase21-forecasting/` (ignored by Git):

- `citywide_hourly.parquet`: complete city calendar, counts, provenance flags,
  local calendar features, and split labels.
- `station_hourly.parquet`: station calendar grid with nullable targets, original
  observations, assumption flags, training membership, features, and splits.
- `station_coverage.parquet`: per-ID coverage, first/last recorded activity,
  totals, training membership, and longest no-record interval.
- `coverage_gaps.parquet`: contiguous no-record intervals with half-open UTC
  bounds and their position relative to observed activity.
- `activity_periods.parquet`: retrospective recorded-activity episodes, not
  operational schedules.
- `report.json`: source warehouse SHA-256, policy configuration, boundaries,
  dependency versions, and actual reconciliation/coverage counts.
- `verification.json`: independent Parquet reconciliation, duplicate-key checks,
  and confirmation that the source warehouse hash was unchanged.

The pipeline reads the much smaller reviewed hourly table and builds a bounded
1.5-million-row February grid in pandas; it never loads two million raw trip rows.
For multi-month scale, prepare partitioned grids rather than expanding every
station over all history in one in-memory table. The notebook reads summaries
and uses a Parquet station filter for its example, with explanations and saved
citywide, coverage, and station plots. Its optional zero demonstration changes
no exported files.

Verification: **75 tests passed, plus 6 subtests**, including 19 new Phase 2.1
cases covering explicit zero policies, future-observation invariance, split and
local-calendar boundaries, count conservation, identifier preservation, invalid
inputs, activity/gap auditing, reproducible exports, and unchanged source bytes.
An independent DuckDB query over the generated Parquet confirmed **zero duplicate
station-hour keys** and both target sums equal **2,030,269**. Source hashing before
and after preparation confirmed the reviewed warehouse was unchanged.

### Limitations and historical extension

February-only data offers 28 winter days and only 20 training days (less than
three weekly cycles). Four-day validation and test intervals do not each cover
an entire week. There is no annual seasonality, summer comparison, or evidence
for robust model ranking. These are provisional learning splits, not a basis for
production-quality model claims.

Phase 2.2A fixes the historical extension at **24 consecutive months, January
2024–December 2025**. The earlier 12–24-month proposal is superseded; retain these
ingestion and evaluation safeguards:

1. Acquire complete monthly NYC archives separately when authorized; retain
   manifests, checksums, row counts, schema versions, and reconciliation audits.
2. Extend reporting windows and process partitioned monthly outputs with
   cross-period ride-ID auditing and DST-safe UTC hourly keys. Keep raw IDs;
   introduce station mappings only through reviewed, versioned evidence.
3. Obtain historical availability/maintenance evidence where possible; current
   station lists alone cannot establish historical opening/closing dates. Review
   missing targets and source outages before adopting any zero policy.
4. Define chronological held-out periods and rolling-origin validation spanning
   full weeks and seasons. Fit cohort criteria, transformations, and learned
   features on each training history only. Do not use future activity or realized
   future weather as forecast inputs.
5. After data preparation is reviewed, proceed to forecasting baselines and
   measured experiments, followed incrementally by the full project roadmap.

**Phase 2.1 checkpoint:** review the preparation notebook and station availability
policy with the mentor. Phase 2.2A historical ingestion is described below. No
models or weather integration were implemented, and no changes were committed
or pushed.


## Phase 2.2A — historical ingestion expansion

The final historical specification is **fixed**: NYC local starts in
`[2024-01-01, 2026-01-01)`, training January 2024–June 2025, validation
July–September 2025, and test October–December 2025. Forecast targets remain
hourly citywide and station ride starts, with a **24-elapsed-hour horizon** and
**100 stable stations selected using training history only**. The full platform
roadmap remains unchanged. The February-only Phase 2.1 split is an educational
checkpoint, not the final historical split.

This phase adds official archive discovery, resumable streaming downloads,
verified extraction, monthly orchestration around the unchanged Phase 1 ETL,
manifest tracking, and global duplicate-audited ride-start-month partitions.
Existing February data is registered by read-only reference. Source partitions
stay immutable; each global reconciliation publishes a new canonical snapshot.
No station cohort selection, zero filling, model training, notebook modification,
weather integration, or deployment occurs in this phase.

See [the historical ingestion guide](docs/historical_ingestion.md) for the
architecture, boundary policy, current measurements, commands, and known gaps.

```bash
# Inspect archive directories and storage requirements without downloading ZIPs.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --start 2024-01-01 --end 2026-01-01 --plan

# Reuse or resume the two authorized pilots; existing verified partitions skip.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month 2024-01 2025-03 --chunk-size 50000 --memory-limit 512MB --reconcile
```

The 24-month ZIP total is **16.387 GiB**; the expanded CSV total is also
**16.387 GiB**. The conservative plan budgets **71.318 GiB plus a 5 GiB reserve**,
before January 2026 boundary input. Current disk space does not support the full
range under that plan. Only January 2024 and March 2025 were downloaded as complete
pilot sources. Expand storage and rerun the plan before processing the remaining
21 modelling archives.

A bounded January 2026 probe found **14 December 2025 starts in 53,277 sampled
rows**. Include that archive as boundary input before declaring December start
coverage complete, without extending the modelling window. This is not its full
boundary count; very long rides and late publication still require review.
Standalone pilot start-month counts remain partial until adjacent sources are
available. Naive timestamps still use the reviewed NYC assumption; authoritative
publisher timezone confirmation and fall-DST ambiguity resolution remain open.

### Phase 2.2B — catalogue schema blocker fixed

The failed union mixed Hive-style January/March paths with February's ordinary
reviewed directory. DuckDB inferred an extra `source_month` column from the former;
the physical Parquet schemas were identical. The catalogue now disables Hive
inference and explicitly selects a fixed schema, preserving string identifiers,
UTC epoch nanoseconds, and source provenance. Missing optional end-station/duration
metadata becomes typed nulls; missing required fields or inconsistent exact epochs
fail explicitly. Archive month and NYC ride-start month remain separate.

**Verification:** 22 focused catalogue tests passed; the complete suite passed
**112 tests plus 6 subtests**. A metadata-only union schema check on the three real
partitions succeeded. No full historical scan, real-data ETL, downloads, or new
snapshot construction ran in Phase 2.2B. Source partitions and unrelated edits
were preserved. This was the Phase 2.2B checkpoint; the subsequent read-only
integration verification is recorded below. See the
[historical ingestion guide](docs/historical_ingestion.md#phase-22b--explicit-catalogue-schema-alignment)
for the exact contract and regression coverage.

## Engineering checkpoint — catalogue reconciliation and GCS

Phase 2.2C executed the explicit 11-column union over the three existing processed
archives in one read-only pass. It reconciled **7,087,613 reported input records**
with **7,084,610 accepted rides** and **3,003 audited rejections**. No duplicate
ride IDs were found within or across these sources, including rejection audits.
The accepted counts are 1,886,925 for January 2024, 2,030,542 for February 2025,
and 3,167,143 for March 2025. UTC precision and source provenance checks passed.
The archive/local-start-month distinction preserved 410 December 2023 starts
from January 2024, 273 January 2025 starts from February, and 410 February starts
from March. These three archives are partial historical coverage; no canonical
monthly snapshots were published. See [the reconciliation results](docs/historical_ingestion.md#actual-pilot-verification).

Python GCS integration now uses ADC for uploads, downloads, existence/metadata
checks, and explicit SHA-256 verification. The optional `cloud` extra installs
`google-cloud-storage`; `citypulse-publish` publishes one existing Parquet under
100 MiB without overwriting objects or local receipts. The 36-byte ADC smoke test
passed, followed by publication and verified download of the existing reviewed
February hourly artifact: **946,546 bytes and 598,630 station-hour rows** at
`gs://citypulse-ai-511315-data/curated/forecasting/2025-02/reviewed-v1/february_2025_hourly_rides.parquet`.
SHA-256, row count, schema including embedded metadata, and cloud provenance
metadata matched. This is the existing February baseline; it does not yet include
the additional February starts found in March's archive or constitute a complete
24-month dataset. See [GCS setup, publication, and verification](docs/gcs.md).

The completed implementation passed **158 automated tests plus 6 subtests**.
Raw data, generated outputs/publication receipts, virtual environments, and ADC
credentials remain outside the Git checkpoint. No forecasting models, weather
integration, or deployment were added.

### Phase 2.2D — historical source publication

The historical CLI now supports explicit `--cloud-publish` and
`--publish-existing-only` modes. Local ETL completion and cloud publication have
separate manifest states. Source-month artifacts use hash-based revisions;
generation-pinned verification and per-file checkpoints support safe retries
without overwrites or deletions. Full source publication remains distinct from
complete ride-start-month coverage and canonical snapshot construction.
See [one-month commands, recovery, and status inspection](docs/historical_ingestion.md#phase-22d--optional-gcs-publication-after-etl).
The implementation passed **187 tests plus 6 subtests** using offline unit and
synthetic integration tests; no real-month uploads, downloads, or ETL runs were
performed in this phase. A real-month upload requires separate authorization and
an explicit file-size ceiling large enough for its validated-trip Parquet.

### Cloud-backed storage retention

Original source ZIPs (or February's exact original CSVs), audit JSONs and immutable
per-month processing-manifest snapshots can now be backed up through an explicit
create-only command. Opt-in cloud resume verifies prior checksums against unchanged
live object generations; catalogue construction can restore its required Parquet
and rejection-audit inputs to a persistent cache. Local-only behavior stays unchanged.
The offline retention report explains recovery requirements and blockers; it has
no deletion mode. See [commands and retention policy](docs/cloud_retention.md).

The implementation passed **235 automated tests plus 6 subtests** using mocked GCS
operations and synthetic files. The real metadata-only dry run identified 52
potential candidates (3.34 GiB), with **zero bytes eligible** under current evidence:
source/manifest backups and February's full historical publication still need
separately authorized verification. No real transfers, historical ETL, cleanup,
commits or pushes were performed during this implementation.

## Local MLP inference API

The existing PyTorch citywide MLP can be served locally with `/health` and
`/predict`, using its saved weights and training-fitted scalers. See
[the FastAPI guide](docs/local_mlp_api.md) for artifact checks, launch commands,
request examples, elapsed-hour/DST semantics and exploratory-model limitations.

## Inference continuous integration

[GitHub Actions CI](docs/github_actions_ci.md) checks repository/Docker
configuration and artifact-free gateway and TensorFlow worker contracts on
pushes to `main` and pull requests targeting `main`. Trained-model parity remains
a separate local check; CI does not train models, build images or deploy services.
