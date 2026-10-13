# Local MLflow tracking of existing experiments

MLflow stores experiment parameters, metrics and provenance so you can inspect
and compare runs in a browser. This integration imports existing results; it
does not execute training, evaluate new predictions, register deployable model
versions, or attach MLflow to API requests.

## Isolated setup and commands

The main Python 3.14 environment and the TensorFlow Python 3.11 environment are
preserved. MLflow 3.16.1 runs only in `.venv-mlflow`, using the existing Python
3.11 interpreter. Its resolved dependencies are pinned in `requirements/mlflow.txt`.
Neither serving dependencies, Docker definitions nor inference CI installs MLflow.

The isolated environment is already installed locally. To recreate it later:

```sh
.venv-tensorflow-311/bin/python -m venv .venv-mlflow
.venv-mlflow/bin/python -m pip install --no-cache-dir -r requirements/mlflow.txt
```

From the repository root, prepare the tiny MLP checkpoint metadata (in the
existing PyTorch environment), then import saved JSON results:

```sh
PYTHONPATH=src .venv/bin/python -m citypulse_ai.experiments.mlflow_import --prepare-mlp-metadata
MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=src \
  .venv-mlflow/bin/python -m citypulse_ai.experiments.mlflow_import
```

The preparation command reads the trusted existing checkpoint using CPU and
`weights_only=True`. It extracts feature order and training bounds into a small
JSON sidecar, not weights or fitted tensors. It verifies an existing sidecar and
refuses conflicting content. The tracking importer itself never imports Torch,
TensorFlow, XGBoost or ETL modules. Missing required source metadata/checkpoints
cause an error rather than recovery downloads or fitting.

Open the dashboard with:

```sh
MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 \
  .venv-mlflow/bin/mlflow ui \
  --backend-store-uri "sqlite:///$PWD/outputs/mlflow/tracking.sqlite" \
  --default-artifact-root "$PWD/outputs/mlflow/artifacts" --no-serve-artifacts \
  --host 127.0.0.1 --port 5000 --workers 1
```

Visit **http://127.0.0.1:5000**, select **CityPulse preliminary citywide Jan-Feb
2024**, and compare its five runs. Display the `validation_MAE`, `validation_RMSE`,
`validation_WAPE_percent`, `test_MAE`, `test_RMSE` and `test_WAPE_percent` columns.
Open a run to inspect parameters and provenance tags. Stop the UI with Ctrl+C;
the SQLite store remains on disk. MLflow's documented [local database
setup](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/)
uses the same backend URI for imports and UI access.

## Sources and tracked metadata

`config/mlflow_existing_experiments.json` lists the exact source paths:

- **Lag 24, Lag 168 and XGBoost**: the original XGBoost `experiment.json` contains
  their separate validation/test metric rows. Baselines are imported once,
  despite repeated comparison rows in later experiments. XGBoost's recovered
  `model_manifest.json` supplies verified feature order, raw preprocessing,
  checkpoint reference and checksum. The recovery method is explicitly tagged.
- **PyTorch MLP**: its original run JSON supplies settings, architecture, selected
  epoch and metrics. The small checksum-linked checkpoint sidecar supplies exact
  ordered features and eligible training bounds.
- **Keras LSTM**: its original JSON supplies lookback/horizon, settings, parameter
  count, sequences, epoch and metrics. Its architecture is labelled as coming
  from the existing LSTM documentation, not from an original JSON field.

All five runs are labelled **preliminary**. Chronological periods are January 1
through February 1 exclusive for training, February 1–15 exclusive for validation,
and February 15–March 1 exclusive for test, in America/New_York. Eligible MLP/XGB
training targets start January 8; baselines have no fitting. These periods and
the 24-elapsed-hour/local-midnight contract are explicitly sourced to the
inspected forecasting implementation/docs, whose hashes are retained. Validation
and test metric names remain distinct. MAE/RMSE are ride-count errors; WAPE stays
in the saved percentage units, not a fraction. No metrics are recomputed.

Parameters include recorded hyperparameters, input configuration, training
rows/sequences, selected epochs and checkpoint references where applicable.
Tags retain source JSON hashes, recorded runtime versions and the original data
provenance (including archive and manifest hashes and coverage limitations).
Original training commit IDs are **not recorded**, so they remain unknown.
`import_code_commit` identifies the repository HEAD at import, not the original
training code or an exact snapshot of uncommitted importer changes. Import metric
timestamps are ingestion timestamps, not inferred historical training times.

The imported examples remain exploratory winter citywide experiments with
previously inspected test data. Their results are not the final station-level
benchmark, and processed archive coverage is not proof of complete demand.
Recorded scaler values remain in their original artifacts; MLflow holds
configuration/provenance and references, not copied model serialization.

## Storage and repeated execution

Everything generated lives in gitignored `outputs/mlflow/` or `.venv-mlflow/`.
The importer specifies a local SQLite URI explicitly; an external environment
tracking URI cannot redirect these imports to a cloud store. Telemetry is disabled.
No dataset, checkpoint, source archive or model is logged as an MLflow artifact.
The empty artifact locations reserve a local namespace only.

Each immutable source JSON hash plus model name defines one import key. A file
lock serializes local import processes. Re-running verifies parameters, metrics
and provenance and preserves the same run IDs and one metric-history entry per
metric. A partially imported run resumes missing entries. Changed enrichment
metadata for an existing key causes a clear error instead of silently overwriting
an old run or creating a duplicate. A changed source JSON hash represents a new
source version; existing records are never rewritten. Keep the config, original
JSONs/manifests, referenced checkpoints and MLP sidecar for repeatable import.

## Verification

Run the small fixture and real temporary SQLite tests in the tracking environment:

```sh
MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=src \
  .venv-mlflow/bin/python -m unittest discover -s tests -p test_mlflow_import.py -v
```

These tests never train or require real checkpoints. Real-store tests are an
explicit optional skip in environments without MLflow; use `.venv-mlflow` to run
all of them. They are not added to inference CI's selected test commands.

The original experiment records omit some training code and timing metadata;
this integration preserves those limits rather than guessing. New model training,
automatic tracking callbacks, model registry, remote tracking, dataset artifacts
and deployment automation remain outside this milestone.

Measured local verification: five FINISHED runs were registered, and a second
import reused all five IDs with no new runs or metric-history entries. Every
stored parameter and all six split-specific metrics per run matched the import
plan built from saved sources. Artifact lists were empty. The local dashboard
returned HTTP 200 and its REST endpoint listed all five runs; the verification
server is now stopped. All source/checkpoint and Docker-definition hashes and
both original environment inventories remained unchanged.

The tracking test suite passed **13 tests with no skips**. The artifact-free inference suites passed **67 gateway tests and 15 worker tests**;
static/Compose and whitespace checks also passed.
The isolated tracking environment occupies roughly **628 MiB logical / 677 MiB
allocated** for tracking alone; no model environment was replaced. The local
tracking/verification data initially occupied **0.93 MiB**. Host free space was
**26.19 GiB before isolated setup and 25.47 GiB after verification**. Import test
stores use tiny temporary synthetic metadata; no original data is deleted.
MLflow dependencies passed `pip check`. The dashboard emitted an upstream
Starlette WSGI deprecation warning and cancellation output on Ctrl+C shutdown.
No cloud backend or Docker service was launched.

Audit evidence is retained in gitignored `outputs/mlflow/verification.json`,
`ui-verification.json`, the two import receipts and `preflight/`. No retraining,
new forecasts, checkpoint copies, historical downloads, GCS operations, commits
or pushes occurred.
