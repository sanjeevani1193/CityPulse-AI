# Local FastAPI inference for the existing citywide MLP

FastAPI exposes Python functions as HTTP endpoints with validated JSON requests
and automatically generated API documentation. This service wraps the existing
PyTorch citywide forecasting experiment. Training learns weights and fits scalers;
inference restores those fixed artifacts and predicts from new, recorded history.
No weights or preprocessing statistics are fitted by this service.

The service uses FastAPI's [lifespan startup pattern](https://fastapi.tiangolo.com/advanced/events/)
to restore the model once per process, and [typed request bodies](https://fastapi.tiangolo.com/tutorial/body-nested-models/)
to validate inputs. Tests use the documented [lifespan TestClient pattern](https://fastapi.tiangolo.com/advanced/testing-events/).

## Trusted artifacts and preflight

The fixed developer-controlled artifact directory is:

`outputs/phase3-pytorch-mlp-jan-feb-2024/run-20261012T125502822756Z/`

`best_model.pt` contains the selected epoch-12 state, the exact ordered features,
training-fitted feature/target means and scales, and the 24-hour forecast contract.
`experiment.json` provides provenance and recorded reference predictions. The
loader verifies both SHA-256 fingerprints before CPU `weights_only=True` loading.
Changing artifacts requires a reviewed code/configuration update, not a request
parameter. These gitignored files are not packaged into a wheel; launch from this
checkout with its preserved experiment artifacts. Missing/corrupt files produce
an unready service (`503`), never a random-weight fallback. See startup logs.

Feature order is exactly:

```text
hour, weekday, is_weekend, hour_sin, hour_cos,
weekday_sin, weekday_cos, lag_24, lag_168
```

The existing feature builder is reused. Scalers are reconstructed from saved
January training statistics; no `fit` call occurs. The network runs on CPU in
`eval()` and `torch.inference_mode()`, with one PyTorch CPU thread. Output follows
the experiment's nonnegative clipping rule and returns floats, without rounding.
Model version includes the experiment run and checkpoint hash prefix.

Artifact preflight reproduced all 24 stored March 1 predictions exactly, with
maximum absolute difference zero, using the surviving local experiment inputs.
No archives, ETL, cloud access or retraining were needed.

## Launch

Install the serving extra if needed (the existing local environment now has it):

```sh
.venv/bin/python -m pip install -e '.[api,dev]'
# Required macOS runtime setting for the optional XGBoost model at startup.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
PYTHONPATH=src .venv/bin/python -m uvicorn citypulse_ai.api.app:app \
  --host 127.0.0.1 --port 8000
```

Run one worker for this first local service; each worker would load its own copy.
Startup now also attempts to load XGBoost; use the documented per-process OpenMP
setting above. If XGBoost cannot load, MLP remains available. Interactive schemas
are at `http://127.0.0.1:8000/docs`. See [multi-model serving](multimodel_api.md)
for `/models`, explicit selection and configurable trusted artifact directories.

```sh
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @request.json
```

A ready health response is:

```json
{
  "application": "citypulse-ai",
  "application_ready": true,
  "model_ready": true,
  "model_name": "citywide-pytorch-mlp",
  "model_version": "run-20261012T125502822756Z:af2fe71d011d"
}
```

Health checks application/model readiness, not a data store or cloud connection.
An unavailable model gives HTTP 503; invalid request data gives 422.

## Requests and forecast timing

`origin` must include an explicit offset and represent midnight in
`America/New_York`. Equivalent UTC representations are accepted. The service
converts instants into NYC time rather than trusting an offset as a timezone.
History timestamps can use UTC or explicit local offsets. Bare local timestamps
are rejected because they cannot distinguish repeated DST hours.

Each record describes a completed hourly bucket by its **start** timestamp.
`ride_count` must be a finite, nonnegative integer (no booleans, fractions or
unknown values), at most 2^53 − 1. For MLP/XGBoost send 168–8,760 observations in strictly
increasing UTC order, with exactly one record per elapsed hour and no gaps.
History must end at origin minus one elapsed hour. Duplicates are detected by
instant, including two different offset representations of the same instant.
All observations must be strictly before issuance. Unknown hours are never
silently converted into zero demand; explicit zero counts must be recorded data.

For LSTM, send at least 48 continuous hours; see [the worker guide](lstm_worker_api.md).
The MLP/XGBoost 168-hour minimum is a deliberately simple contiguous contract covering both
required lag windows. The feature builder actually consumes target minus 24
and target minus 168 **elapsed** hours. All 48 source timestamps precede issuance;
the horizon never uses actual counts inside the forecast period.

Example request generation, using **synthetic recorded counts** for learning the
HTTP format (these are not experiment measurements):

```sh
.venv/bin/python - <<'PY'
import json
from datetime import datetime, timedelta, timezone
origin = datetime.fromisoformat('2024-03-01T00:00:00-05:00')
hours = [origin.astimezone(timezone.utc) - timedelta(hours=168-i) for i in range(168)]
request = {'origin': origin.isoformat(), 'history': [
    {'timestamp': t.isoformat(), 'ride_count': 100 + i % 24}
    for i, t in enumerate(hours)]}
with open('request.json', 'x') as stream:
    json.dump(request, stream)
PY
```

For the real saved March 1 example, this response excerpt shows the first two
aligned entries; each returned list contains exactly 24 values:

```json
{
  "origin": "2024-03-01T00:00:00-05:00",
  "timezone": "America/New_York",
  "model_name": "citywide-pytorch-mlp",
  "model_version": "run-20261012T125502822756Z:af2fe71d011d",
  "target_timestamps_utc": ["2024-03-01T05:00:00+00:00", "2024-03-01T06:00:00+00:00"],
  "target_timestamps_local": ["2024-03-01T00:00:00-05:00", "2024-03-01T01:00:00-05:00"],
  "utc_offsets": ["-05:00", "-05:00"],
  "predicted_ride_counts": [93.0400390625, 17.976806640625]
}
```

These measured predictions require the real lag history; the synthetic request
above will produce different values.

## DST and model limitations

Targets are origin + 0…23 elapsed hours on a UTC grid, with local timestamps and
offsets derived afterward. On March 10, 2024, hour 02 is skipped locally and the
last bucket starts at March 11 midnight. On November 3, hour 01 repeats with
different offsets and the last bucket starts at 22:00; the 25-hour calendar day's
final 23:00 bucket is outside this 24-hour horizon. Calendar features describe
each actual target's local hour/weekday. Lag offsets remain elapsed-time offsets,
not “same local hour yesterday/last week.”

This is a retrospective, previously inspected January–February development
model: only 576 training rows for 865 parameters, no weather or holiday inputs,
and no final station-level benchmark or uncertainty intervals. DST handling is
tested for timestamp/feature correctness; the training period contains no DST
transition, so this is not evidence of forecast accuracy on DST days.

The API cannot verify source coverage or real observation publication timestamps.
It assumes each supplied count, including the immediately preceding hour, was
finalized and available at issuance. The caller must enforce that availability
assumption; timestamp validation alone cannot establish it. No historical data
is fetched or checked by the service. Large counts/distribution shifts may yield
poor extrapolation. This is a loopback-only educational service without auth,
rate limiting, deployment infrastructure or model registry.

## Tests

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_inference_api.py \
  tests/test_pytorch_mlp.py tests/test_forecasting_baselines.py -q
```

API tests use the existing trusted model, synthetic request histories, and a
read-only comparison against the saved real experiment. The latter needs the
surviving January–March hourly artifacts. Missing artifacts fail the real parity
test rather than silently skipping it. Tests never replace or retrain the model.

## Measured implementation verification

- Relevant API, MLP, baseline, preparation and XGBoost regressions: **92 passed
  in 18.80 seconds**. One Starlette TestClient deprecation warning recommends
  `httpx2`; the installed `httpx` TestClient path still passed. No serving failure
  or artifact blocker remains.
- Real loopback HTTP smoke: `/health` **200**, `/predict` **200**, and an
  observation at issuance **422**. Exactly 24 outputs, zero maximum absolute
  difference from the saved experiment, and identical repeated requests.
- The checkpoint and experiment JSON retained their original SHA-256 hashes.
  The local test server was shut down after verification; use the launch command
  above to start it again.
- Measured serving packages: FastAPI **0.143.0**, Uvicorn **0.54.0**, Pydantic
  **2.14.1**, Starlette **1.8.0**. `pip check` reported no broken requirements.
  Dependencies were added to the existing local environment; no existing package
  was upgraded in that installation.
