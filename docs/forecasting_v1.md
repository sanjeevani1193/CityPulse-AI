# Station100 inference: Forecasting v1 release candidate

**Status, October 28:** station API implementation, pinned artifact provisioning,
post-hoc diagnostics and lightweight tests are available. Native model restoration
has been exercised for XGBoost, MLP and TFT. Real HTTP verification, native station
LSTM parity, and station Docker startup remain pending because macOS reports
warning-level memory pressure. Docker Desktop is stopped and will be started
manually by the user. This is **not yet a fully verified release**.

## Architecture and preserved functionality

The existing pipeline is ETL → verified GCS/Parquet → frozen station dataset →
training-only preprocessing → separately tracked MLflow experiments → paired
benchmark → inference adapters. Historical ingestion is complete: 89,995,902
accepted recorded starts in January 2024–December 2025, with January 2026 used
only for boundary reconciliation. No ETL or dataset access is needed for inference.

The frozen cohort contains 100 stations selected using January 2024–June 2025
only; see [dataset criteria](final_tft_dataset.md). Training has 53,300 windows;
validation (July–September 2025) and test (October–December 2025) each have 9,200.
Each uses 336 historical elapsed hours and 24 future elapsed hours issued at
America/New_York midnight. The target is `accepted_recorded_starts_v1`, not
physical demand, unmet rentals, publisher completeness or station availability.

The new entry points are `citypulse_ai.api.station_app:app` (gateway) and
`citypulse_ai.api.station_worker:app` (TensorFlow). Existing `api.app` and
`api.lstm_worker` and the original `compose.yaml` preserve the preliminary
citywide services. They have different schemas and checkpoints; do not mix them.

| API identifier | Frozen artifact / algorithm | Processing |
| --- | --- | --- |
| `xgboost` (default) | 300-round depth-6 `model.ubj` | Original 48 raw tabular features, categorical station code |
| `pytorch-mlp` | `epoch-010.pt` | Original fitted training scaler, one-hot stations, explicit missing indicators |
| `tensorflow-lstm` | `epoch-018.keras` | Original station normalization and 336×16 encoder; isolated worker |
| `pytorch-tft` | `epoch-006.pt` | Original station normalization, variable selection, LSTMs, attention, q10/q50/q90 |
| `lag-24` | Historical elapsed lag | Requires all 24 corresponding observed counts |
| `lag-168` | Historical elapsed lag | Requires all 24 corresponding observed counts |

XGBoost loads once at startup; PyTorch models load on first selection and are
cached. TensorFlow never imports in the gateway. Concurrent local model requests
are serialized. Lazy loading reduces startup footprint but is **not** a guarantee
that all cached frameworks fit an 8 GB Mac; use separate process sessions when
memory is constrained. No fitting or training occurs during requests.

## Provision trusted artifacts

Use the existing API environment (PyTorch/XGBoost) and separate Python 3.11
TensorFlow 2.21.0 / Keras 3.15.1 environment. The project `api`, `cloud` and
`lstm-worker` optional dependencies document requirements; Docker/CI use their
existing pinned requirements. Do not install TensorFlow in the gateway environment.
No dependencies were installed during this preparation.

With authorized Application Default Credentials, not checked-in keys:

```sh
PYTHONPATH=src .venv/bin/python scripts/provision_forecasting_v1.py --demo
```

The checked-in `api/station_artifacts.json` pins each object generation, size and
SHA-256. Seven required files total **3,992,134 bytes**, including four frozen
models and fitted preprocessing. No weights or credentials enter Git. Existing
matching files are reused; mismatched files fail without being overwritten.
Every model verifies SHA-256 again before deserialization. Artifact paths are
operator configuration, never request fields. Missing/corrupt models report
unready/503 instead of creating a substitute or retraining.

`--demo` creates a single verified validation example for station `6140.05` at
2025-08-01 midnight. It range-reads only one 22.27 MB validation panel member from
the pinned recovery ZIP, verifies that member's SHA-256/CRC, and extracts a tiny
336-hour history. It does **not** download the full dataset ZIP. Three saved
validation prediction files (approximately 3.4–4.3 MB each) supply reference
forecasts; only newly owned scratch is removed. All downloads stay under ignored
`outputs/`. Private GCS requires authorized access; this repository is not a
public distribution of model weights.

## Start local services when host memory permits

```sh
export PYTHONPATH=src
export CITYPULSE_STATION_ARTIFACT_DIR="$PWD/outputs/forecasting-v1/artifacts"
# Existing documented macOS OpenMP configuration for the gateway:
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs"
.venv/bin/python -m uvicorn citypulse_ai.api.station_app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

In a separate terminal, only if LSTM is needed and memory permits:

```sh
PYTHONPATH=src CITYPULSE_STATION_ARTIFACT_DIR="$PWD/outputs/forecasting-v1/artifacts" \
  TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
  .venv-tensorflow-311/bin/python -m uvicorn citypulse_ai.api.station_worker:app \
  --host 127.0.0.1 --port 8001 --workers 1
```

The gateway's fixed operator-configured `CITYPULSE_LSTM_WORKER_URL` defaults to
localhost:8001. Callers cannot select URLs or artifact paths. Worker timeouts,
invalid identities or outages return bounded errors without affecting other
models. Stop these foreground processes with Ctrl-C. Do not run services under
warning/critical host memory pressure merely to complete a demo.

## Requests, responses and masks

- `GET /health`: default XGBoost readiness (200 or 503).
- `GET /models`: all six methods, versions, artifact availability, loaded state,
  readiness. `verified-not-loaded` is distinct from ready.
- `GET /stations`: the exact frozen station vocabulary.
- `POST /predict`: selected model, default XGBoost.

```sh
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/models
curl --fail http://127.0.0.1:8000/stations
curl --fail -H 'Content-Type: application/json' \
  --data-binary @outputs/forecasting-v1/demo-request.json \
  http://127.0.0.1:8000/predict
```

The complete generated request contains `station_id`, offset-aware midnight
`origin`, optional `model`, and exactly 336 ordered hourly history objects:
`{"timestamp":"2025-07-18T04:00:00+00:00","ride_count":3}` is the first
bucket shape for this example (the actual generated count comes from the source).
A null count means unknown; zero is an explicitly recorded accepted-start count.
Missing, duplicate, future or unordered hour rows are rejected. Known operating
status must include its `status_available_at` timestamp; otherwise status remains
unknown. Optional count `available_at` must be after bucket end and no later than
issuance. Omitting it explicitly adopts the retrospective bucket-end availability
assumption. Caller counts/status are assertions, not independently certified archive
evidence; the API cannot verify physical availability from ride histories.

A successful response has `station_id`, `model`, checksum-derived `model_version`,
`origin`, `timezone`, 24 `target_timestamps_utc`, 24 local timestamps and offsets,
and 24 `predicted_ride_counts`. TFT additionally returns ordered `quantiles`
(`q10`, `q50`, `q90`) and a nominal 80% `prediction_interval`; its point forecast
is q50. Values are inverse-transformed then clipped with `max(0, prediction)`.
Counts remain continuous conditional forecasts, not rounded integers.

Future targets are never supplied to adapters. The original feature/window code
constructs known calendar covariates and masked future labels. DST uses UTC elapsed
ordering: fall-back repeats local 01:00 with distinct offsets, and 24 hours need not
cover an entire 23/25-hour local calendar day. Frozen models reject issuance before
the training period ended. Unknown stations require a future retraining decision,
not an improvised embedding.

When services can run safely, the validation-only smoke client is:

```sh
PYTHONPATH=src .venv/bin/python scripts/verify_station_api.py \
  --output outputs/forecasting-v1/http-verification.json
```

Use `--models xgboost pytorch-mlp pytorch-tft lag-24 lag-168` without the worker;
use `--models tensorflow-lstm` for a worker-only session. It checks 24 outputs,
repeated inference, future-input rejection and saved reference parity where
available (`rtol=1e-5`, `atol=1e-4`). No saved per-target TFT validation export
exists: that parity is not claimed. A separate frozen synthetic restore probe
matched its original normalized-output hash exactly; the real-window original TFT
forward versus HTTP comparison still awaits safe service execution.

## Docker preparation — execution pending

An image packages application code and dependencies; a container runs that image.
The existing Dockerfiles install CPU-serving requirements, use a non-root UID and
health checks, and exclude datasets/credentials through `.dockerignore`.
`compose.station.yaml` reuses both Dockerfiles/images with station entry-point
overrides and read-only artifact bind mounts. The optional `lstm` profile isolates
TensorFlow; only gateway localhost:8000 is exposed. The internal worker hostname is
`lstm-worker`, not the container's localhost. The original citywide Compose remains
available unchanged.

**Do not run these commands until Docker Desktop is manually started with enough
host memory. They are prepared instructions, not evidence of a passed check.**

```sh
docker compose -f compose.station.yaml --profile lstm config --quiet
docker compose -f compose.station.yaml build forecast-gateway
docker compose -f compose.station.yaml build lstm-worker
docker compose -f compose.station.yaml --profile lstm up -d
# Run the validation-only client above, plus worker failure/recovery checks.
docker compose -f compose.station.yaml --profile lstm stop
```

Build sequentially only if required; existing images may lack the new source.
No automatic prune or deletion is performed. Each service retains its existing
1536 MiB memory/swap ceiling and CPU limit. These caps are not proof of combined
host safety. Existing images previously used approximately 7.17 GB; new image/cache
sizes and station container RSS must still be measured. Keep at least 10 GiB host
free, preserve Docker settings, and stop test containers after verification.

## Verification and remaining release work

Measured in this preparation:

- All seven frozen serving files retrieved with pinned generations/SHA-256.
- Native real XGBoost and lag forecasts exactly matched the validation references;
  native MLP maximum absolute difference was 0.00000642 rides.
- Native TFT produced 24×3 quantiles; its separate frozen synthetic restore probe
  reproduced the original hash exactly. This is not a test-set evaluation.
- 23 new artifact-free contract/worker-transform/diagnostics tests and three
  synthetic provisioning tests passed. No test doubles are presented as real
  learned predictions. An additional 44 dataset/history regression tests passed. The two synthetic
  worker tests also passed in the existing Python 3.11 worker environment.
- Static Python, workflow YAML, Dockerfile contracts and Compose safety checks
  passed without starting Docker.
- Real HTTP verification stopped **before starting either service**: macOS kernel
  memory-pressure level 2, about 7.46 GiB swap in use. This host-wide observation
  is not attributed to CityPulse. No service RSS measurement is available.

CI now selects artifact-free station gateway and worker contracts in addition to
existing citywide contracts. Real saved-artifact parity stays local; fresh runners
have no weights or cloud credentials. Remote CI has not yet run these changes.
The TensorFlow runtime regression and full gateway runtime regression are deferred
while host memory pressure remains high; no full-suite success is claimed.

Next: finish native LSTM/HTTP parity and worker failure isolation when memory is
safe; verify station Docker startup, parity and restart behavior after the user
starts Docker; inspect final diffs, then commit/push the verified release. Do not
claim production readiness: authentication, rate limiting, live count publication,
station-status ingestion, weather and hosted deployment remain deferred.

See [saved benchmark diagnostics](forecasting_v1_diagnostics.md) and the existing
[CI guide](github_actions_ci.md) for the separation between testing and deployment.
