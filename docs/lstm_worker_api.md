# TensorFlow LSTM worker and three-model gateway

TensorFlow runs in the verified Python 3.11.15 environment rather than inside
the Python 3.14 MLP/XGBoost process. This preserves each model's library versions,
preprocessing and predictions and keeps TensorFlow imports out of the gateway.
The gateway adds `tensorflow-lstm` without changing the default MLP selection.

## Start and stop

Terminal 1, from the project root:

```sh
PYTHONPATH=src .venv-tensorflow-311/bin/python -m uvicorn \
  citypulse_ai.api.lstm_worker:app --host 127.0.0.1 --port 8001 --workers 1
```

Terminal 2:

```sh
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
PYTHONPATH=src .venv/bin/python -m uvicorn citypulse_ai.api.app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Both listen only on localhost. Stop each with Ctrl+C; neither command removes
data or artifacts. One worker process per service avoids duplicate model loads.

FastAPI 0.143.0 and Uvicorn 0.54.0 plus their dependencies were added only to
the existing TensorFlow environment. All 137 pre-existing package versions were
constrained and verified unchanged; `pip check` passed. The main environment
received no package installation or upgrade. `lstm-worker` in pyproject.toml
records the worker's dependency extra; creating/replacing environments is not
part of this implementation.

## Artifacts and model behavior

The worker requires `best.weights.h5` and `experiment.json` from:

`outputs/phase3-tensorflow-lstm-jan-feb-2024/run-20261012T141848308777Z/`

They contain the trained epoch-20 weights and the January-only scaler statistics
(mean, scale, variance, 744 fitted observations). SHA-256 pins are shared as
metadata through `api/lstm_contract.py`; no weights are committed to Git.
`CITYPULSE_LSTM_ARTIFACT_DIR` may relocate the exact trusted pair, not change
their identity. Caller-supplied paths are forbidden. Missing/corrupt artifacts
yield worker readiness false and prediction 503; there is no fallback training.

The worker constructs the same LSTM(32) → Dense(24), restores weights without
compiling an optimizer, and restores the exact StandardScaler attributes without
fit. It loads once at startup, on CPU with one intra/inter-op thread and
deterministic operations. An inference lock serializes TensorFlow execution.
It transforms a `(48,1)` raw context to float32 `(1,48,1)`, runs with
`training=False`, inverse-transforms `(1,24)` outputs and clips at zero. No nine
calendar/lag features or MLP scaler are applied to the LSTM.

## Requests, validation and forwarding

The gateway accepts `pytorch-mlp` (default), `xgboost` and `tensorflow-lstm`.
Use the [existing request format](local_mlp_api.md) with a top-level `model`
field. MLP/XGBoost still require at least 168 continuous hours. LSTM accepts at
least 48; the gateway validates the entire supplied history, then forwards only
the final 48 records plus origin. The worker requires exactly 48 records and
validates independently. It does not accept a model selector or worker URL.

```sh
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @request-lstm.json
```

To make `request-lstm.json`, use the original guide's complete history generator
and add `request['model']='tensorflow-lstm'`. For a direct worker request,
omit `model` and include exactly the final 48 observations. Synthetic histories
demonstrate the format; real saved-reference parity requires real recorded lags.

Both endpoints require an offset-aware origin representing NYC local midnight.
Every bucket timestamp must be hour-aligned and strictly before origin; the last
must equal origin minus one elapsed hour. Duplicate instants, unordered/gapped
hours, unknown counts, booleans, fractions and negative counts are rejected with
422. Unknown hours are never imputed to zero. Calendar labels use NYC while UTC
defines the 48-hour context and 24-hour horizon. DST skips/repeats local labels
with explicit offsets; it never changes the number of returned forecasts.

The response schema is unchanged: origin, timezone, model name/version, 24 UTC
and NYC-local target starts, offsets and predicted ride counts. The gateway
validates returned lengths, finiteness, nonnegativity, trusted model identity and
the exact requested time grid. Wrong/malformed worker responses cannot silently
become another model's predictions.

## Worker configuration and failure behavior

`CITYPULSE_LSTM_WORKER_URL` defaults to `http://127.0.0.1:8001`. It is server
configuration only and must be a loopback HTTP origin without credentials,
query, fragment or extra path. Request bodies cannot override it. The gateway
disables HTTP redirects and environment proxy lookup. It reuses a single HTTP
client with at most four connections; connect timeout is one second and network/
pool timeout is five seconds. These are per-phase limits, not a total wall-clock
request deadline. There are no automatic retries.

`/models` checks live worker health, including its expected version. The gateway
can start before the worker and becomes LSTM-ready once it is available, without
restart. Worker connection/unready failures return 503, timeouts 504, and invalid
upstream status/payload 502. LSTM failures never substitute MLP predictions or
disable otherwise healthy MLP/XGBoost. `/health` keeps default-MLP semantics;
use `/models` for individual readiness. Only two lightweight metadata modules
and the HTTP client are imported in the gateway, not TensorFlow/Keras.

## Verification and resource limits

- Main environment: **79 gateway/API tests passed**; the existing MLP and XGBoost
  parity/regression checks remain successful. Isolated environment: **15 worker
  tests passed**. No training tests were run. Each suite emitted the existing
  nonblocking Starlette TestClient deprecation warning.
- Real localhost tests: both health endpoints and gateway `/models` returned
  200 with all three ready. Each model returned exactly 24 predictions with
  **zero difference** from its saved March 1 reference and identical repetitions.
  Gateway default MLP and explicit MLP matched. LSTM gateway/direct-worker
  responses matched; both DST transitions were exercised.
- Future inputs returned 422 for all models. After the real worker was stopped,
  its readiness became false and LSTM returned 503 while MLP/XGBoost remained
  200. Mocked tests also cover timeouts, malformed data and wrong worker versions.
- All six trusted model/metadata file hashes remained unchanged. Scaler values
  and model weights were checked unchanged in worker tests.
- During the HTTP/concurrent-request exercise, combined sampled macOS RSS peaked
  at **231,784,448 bytes (~221.05 MiB)**: worker maximum ~111.11 MiB, gateway
  maximum ~109.94 MiB. Fifteen samples were taken at approximately 0.1-second
  intervals. This sums RSS, may double-count shared pages, may miss short peaks,
  and is not USS, a cold-start lifetime peak or a Docker/Linux measurement.
- The user-specified 3.84 GiB Docker VM has not been tested here. A future image
  still needs Linux ARM64 artifact parity and combined startup/concurrency memory
  profiling. No automatic VM-memory increase or Docker cleanup was performed.
- Added serving-package files occupy **13,619,943 logical bytes (~13.0 MiB)** in
  the isolated environment. Host free space was ~30.93 GiB before and ~29.89 GiB
  afterward. That filesystem change includes test/system activity and is not a
  measurement of dependency size or a Docker build budget. No existing packages
  or data were deleted to make room.

## Limitations

The LSTM used only 29 January sequences and 48 univariate historical hours;
MLP/XGBoost have different calendar/weekly-lag inputs. All are exploratory winter
citywide models, with previously inspected test data and no weather/holiday or
station-level final benchmark. Exact original LSTM per-hour held-period arrays
were not saved; only its saved March 1 example and aggregate metrics establish
original parity. Caller-provided observations are assumed finalized at issuance;
timestamp validation alone cannot establish real publication availability.

This is a local educational service without public deployment, authentication,
production admission control or Docker verification. Gateway and worker services
were stopped after smoke verification. No ETL, downloads, GCS, retraining, model
artifact changes, commits or pushes occurred.
