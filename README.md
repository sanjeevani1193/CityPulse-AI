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

The ETL package contains the implemented trip pipeline, and
`features/reporting.py` prepares the reviewed February reporting dataset.
The remaining subpackages are placeholders. The `src/` layout separates
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

Next, review missing station values and long-duration trips with the mentor,
decide rejection and station/hour coverage policies, and reconcile exploratory
counts with the ETL before creating forecasting features. Phase 1.3 now supports
cross-file duplicate checks; incremental ingestion remains future ETL work.

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

Resume with **Phase 2.1: station coverage and missing-hour policy** for the reviewed
February dataset. Decide when an absent station-hour means zero demand versus
unknown/unavailable coverage before creating a complete forecasting grid. No
zero-filling has been performed yet. Feature generation, temporal splits,
weather integration, and model training remain subsequent work.
