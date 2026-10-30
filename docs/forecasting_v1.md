# Station100 inference: verified Forecasting v1

**Status, October 28, 2026:** native frozen-model restoration, saved-validation
parity, real six-method FastAPI HTTP inference and CPU Docker Compose inference
are verified on standard GitHub-hosted ARM64 runners. Worker outage isolation and
restart recovery passed after a small verification-harness fix. This completes the
Forecasting v1 reproducible demonstration; it is not hosted production deployment.
The Mac remains unsuitable for heavy verification while memory pressure is elevated.
See [run evidence and secure cloud access](cloud_release_verification.md).

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
forward versus HTTP comparison subsequently passed on the hosted runner (error 0),
as recorded in the measured verification table below.

## Docker Compose — verified on hosted ARM64

An image packages application code and dependencies; a container runs that image.
The existing Dockerfiles install CPU-serving requirements, use a non-root UID and
health checks, and exclude datasets/credentials through `.dockerignore`.
`compose.station.yaml` reuses both Dockerfiles/images with station entry-point
overrides and read-only artifact bind mounts. The optional `lstm` profile isolates
TensorFlow; only gateway localhost:8000 is exposed. The internal worker hostname is
`lstm-worker`, not the container's localhost. The original citywide Compose remains
available unchanged.

The commands below were exercised on the hosted runner. On an 8 GB Mac, first
check memory pressure and start Docker Desktop manually only when resources are
safe. Hosted verification does not establish local capacity.

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
host safety. The verified images are approximately 1.50 GB (gateway) and 2.24 GB (worker),
uncompressed image sizes. Builds and Python environments also consume disk;
this is not their total cache footprint. Observed combined container memory peaked
at approximately 671 MiB across the two runs, not a guaranteed worst case. Keep at least 10 GiB host
free, preserve Docker settings, and stop test containers after verification.

## Measured verification and limitations

Evidence is split across [the full native/container run](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/40707527557)
and [the successful targeted recovery run](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/40709209185).
The first run passed all model checks but failed at `compose start --wait`;
the exact private command diagnostic was not exported. The harness now starts
normally and polls application readiness for at most 90 seconds. The second run
verified only the affected outage/restart/LSTM parity checks; passed native phases
were deliberately not repeated. No model or frozen preprocessing changed.

| Check | Measured result |
| --- | --- |
| Pinned serving artifacts | Seven files; exact generations, sizes and SHA-256 verified |
| Native and Docker XGBoost, lag-24, lag-168 | 24 outputs each; exact validation/lag reference parity |
| Native and Docker MLP | Maximum saved-reference absolute error 0.000005133 rides |
| Native and Docker LSTM | Maximum saved-reference absolute error 0.000004168 rides |
| Native TFT | 24×3 ordered quantiles; original frozen forward comparison error 0 |
| Docker TFT | 24 point outputs and ordered q10/q50/q90; no saved per-target TFT validation export exists |
| Input validation | Seven live invalid cases rejected with 422; DST/masks/causality covered by regressions |
| Worker outage and recovery | LSTM 503 while other models remain available; restart returns valid LSTM forecasts |
| Framework/contract regressions | 174 passed, zero test failures or skips in the first run |
| Native combined child RSS | Peak sampled 995,024,896 bytes (about 949 MiB) |
| Container memory | Peak sampled across runs 703,175,066 bytes (about 671 MiB) |
| Swap | Zero use observed in both hosted runs |

Synthetic full training-loop tests were intentionally not selected for this
verification task. Three new lightweight monitor/workflow/restart tests also
passed locally. Ordinary artifact-free CI passed for both verification commits;
real weights remain restricted to the manual keyless workflow. Sanitized reports
are downloadable from the run pages for 14 days; source, numerical summaries and
commands remain in Git. Private weights, requests, raw forecasts and service logs
were not uploaded as GitHub artifacts. All temporary services and containers
stopped; no GCP VM or heavy local process was started.

No held-out inference, real-data model training, ETL or benchmark recalculation
occurred. Framework unit tests used small synthetic fixtures only.
The previously established scores and non-blind follow-up qualification remain
unchanged. TFT's original per-target validation export is unavailable: the real
HTTP comparison instead uses an independently assembled original validation window
and original model forward path. Its separate frozen synthetic restore probe had
also matched its original output hash exactly.

Operational publication latency, caller-count certification, live station status,
authentication/rate limiting, weather and hosted deployment remain deferred.
Lazy loading does not guarantee that all cached models fit the user's Mac. The
hosted measurements are samples on 16 GB runners, not worst-case memory proofs.

See [saved benchmark diagnostics](forecasting_v1_diagnostics.md),
[keyless workflow and revocation instructions](cloud_release_verification.md),
and the [CI guide](github_actions_ci.md).
