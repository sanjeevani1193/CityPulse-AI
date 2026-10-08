# CityPulse AI

CityPulse AI is a portfolio-grade urban mobility demand forecasting and AI
analytics platform built around NYC Citi Bike trip data and historical weather.
The goal is to demonstrate an end-to-end workflow from validated source data to
reproducible forecasts, interactive analytics, and statistical experimentation.

Phase 0 and Phase 1.1 are implemented: a Python package scaffold and a local CSV
trip ETL pipeline with validation, hourly counts, Parquet output, and a DuckDB
table. Initial EDA in Jupyter includes saved ride counts, missing-value and
duration checks, and an hourly-demand visualization for February 2025. Raw data
stays local; only synthetic fixtures are committed. Weather, forecasting, APIs,
and deployment remain future work.

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
Kubernetes, and cloud hosting belong to later deployment phases.

## Project structure

```text
.
├── AGENTS.md                 # Guidance for future coding agents
├── LICENSE                   # Existing project license
├── README.md                 # Objective, setup, architecture, and roadmap
├── pyproject.toml            # Packaging, dependencies, and pytest configuration
├── notebooks/
│   └── 01_data_exploration.ipynb # Saved exploratory analysis and visualization
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

The ETL package contains the implemented trip pipeline; other subpackages remain
placeholders. The `src/` layout separates importable code from repository files.

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

## Trip ETL (Phase 1.1)

### Raw data contract and rejection policy

The CSV must have a header containing these columns. Extra columns are ignored.
Identifiers are read as strings to retain leading zeros; whitespace is stripped.
Literal strings such as `NA` remain identifiers rather than inferred nulls.

| Column | Contract |
| --- | --- |
| `ride_id` | Required, nonblank trip identifier; unique within the input file |
| `started_at` | Required ISO-8601 timestamp with seconds |
| `ended_at` | Required ISO-8601 timestamp with seconds; later than the start |
| `start_station_id` | Required, nonblank station identifier |
| `end_station_id` | Required, nonblank station identifier |

Missing columns, a headerless file, or malformed CSV syntax fail the run. A
header-only CSV succeeds and creates typed empty outputs. Rows with missing
values, invalid timestamps, or nonpositive/unrepresentable durations are rejected.
All occurrences of duplicate ride IDs are rejected, even when their other fields
match or one occurrence has another error. Duplicate checking follows whitespace
normalization and covers this input file only. Reasons accumulate per record.
No maximum plausible duration is imposed yet; that needs a domain decision.

Rejected rows retain the original five values, a `source_row` record number
(header is 1, first data record is 2), and semicolon-separated rejection reasons.
Record numbers are logical CSV records, not physical lines for multiline fields.
A completed run may contain rejected rows, including all rows, and exits zero;
inspect `report.json` for counts. Configuration/read failures exit nonzero.

### Timestamp and hourly-count policy

- Accept `YYYY-MM-DD HH:MM:SS` or a `T` separator, optional fractional seconds
  (up to six digits), and optional `Z` or numeric `±HH:MM` offset.
- Interpret timestamps without an offset in **America/New_York**.
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

The flow is **CSV → required-column check → normalize/validate rows → UTC hourly
aggregation → Parquet + DuckDB + audit outputs**. Functions are separated into
`schema.py` (contract), `trips.py` (pure ingestion/validation/aggregation steps),
`pipeline.py` (file persistence), and `cli.py` (argument handling).

Each run writes four files in the selected output directory:

| Output | Purpose |
| --- | --- |
| `hourly_rides.parquet` | Station IDs, timezone-aware UTC hour starts, ride counts |
| `citypulse.duckdb` | Materialized `hourly_rides` table, independent of the Parquet file afterward |
| `rejected_trips.csv` | Rejected source records and their validation reasons |
| `report.json` | Input SHA-256/path, row counts, reason counts, timestamp policy, library versions |

Rerunning replaces these named files/table rather than appending trips. Other
DuckDB tables are preserved. Use a separate directory for each source/run you
want to retain. Writes across the four outputs are not atomic; after an I/O
failure, rerun successfully before consuming them. Outputs are sorted by station
and UTC hour. The report contains no run clock, allowing identical reports for
the same input path/bytes and environment. Dependency versions are recorded,
but the project does not yet lock dependencies or system time-zone data; preserve
the environment for reproduction across machines. The pipeline reads one CSV
into memory and does not yet support chunking or cross-file deduplication.

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
multiple monthly files, add cross-file duplicate handling, incremental ingestion,
and memory sizing/chunking. Define station coverage and zero-fill policies before
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

Next, review missing station values and long-duration trips with the mentor,
decide rejection and station/hour coverage policies, and reconcile exploratory
counts with the ETL before creating forecasting features. Cross-file
deduplication and incremental ingestion remain future ETL work.

## Phased roadmap

1. **Scaffold (complete):** package layout, dependency metadata, agent guidance,
   documentation, and import smoke tests.
2. **ETL and warehouse (in progress):** Phase 1.1 trip ingestion, validation, UTC
   hourly counts, Parquet, DuckDB, CLI, and fixture tests are complete. Next add
   real-data readiness controls, then weather ingestion and aligned data contracts.
3. **Features and baseline evaluation:** build calendar/weather features, temporal
   splits, seasonal-naive forecasts, and reproducible backtesting metrics.
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

The next implementation should address the real-data readiness decisions above
using the locally acquired data. Weather integration belongs to a subsequent phase.
