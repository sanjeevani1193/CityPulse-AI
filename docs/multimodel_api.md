# Citywide MLP and XGBoost inference

The existing API loads two frozen models and routes LSTM requests to a separate
local TensorFlow worker; see [LSTM integration](lstm_worker_api.md). Requests
without a `model` field still select the PyTorch MLP. Its prediction values,
scalers, response keys, `model_name` and version are unchanged.

Both adapters reuse the same nine-feature builder, NYC-local midnight issuance
validation and 24-elapsed-hour UTC target grid. Calendar features use NYC local
time; lag-24 and lag-168 observations must precede issuance. Neither adapter
fetches data, trains, tunes or fits preprocessing. Validation still rejects
gaps, duplicates, nonnegative-count violations, future observations and unknown
fields. The retrospective availability assumption and exploratory limitations
in [the original API guide](local_mlp_api.md) remain applicable to both models.

| Request identifier | Response model_name | Inference |
| --- | --- | --- |
| `pytorch-mlp` (default) | `citywide-pytorch-mlp` | January-fitted feature scaler → 9/32/16/1 MLP → saved target inverse scaler |
| `xgboost` | `xgboost` | Nine raw features → 150 depth-3 regression trees, no scalers |
| `tensorflow-lstm` | `tensorflow-lstm` | Last 48 observed hours → saved univariate scaler → LSTM(32)/Dense(24) in a separate worker |

Both return floats clipped at zero, following their original evaluation rules.
The MLP learns nonlinear transformations with dense layers; XGBoost combines
learned tree splits and leaf predictions. Their shared features/issuance contract
allows aligned comparisons, but neither is a final station-level benchmark.

## Trusted artifact configuration

Artifacts load once per model per process during application lifespan startup.
Loading errors are isolated and logged; a missing XGBoost artifact or native
runtime cannot break an otherwise ready MLP. Artifact hashes and metadata are
validated before deserialization. File paths never come from request bodies.

Default directories:

- MLP: `outputs/phase3-pytorch-mlp-jan-feb-2024/run-20261012T125502822756Z/`
- XGBoost: `outputs/phase3-xgboost-jan-feb-2024/recovered-83a9eaa90d964c889a03c7b6ec7de09d/`

Server operators may relocate these exact trusted artifacts with
`CITYPULSE_MLP_ARTIFACT_DIR` and `CITYPULSE_XGBOOST_ARTIFACT_DIR`. These settings
change locations, not accepted model identity: checksum pins remain enforced.
MLP requires its original `best_model.pt` and `experiment.json`; XGBoost requires
`model.ubj` and `model_manifest.json`. The XGBoost manifest's old absolute
`artifact_location` is provenance only: the loader uses the configured directory
and fixed `model.ubj` name, preventing stale host paths from controlling loading.

XGBoost's manifest is checksum-pinned and must declare the verified status,
feature order, raw-feature preprocessing, artifact hash and model version.
The loader requires the recorded XGBoost 3.4.1 runtime and checks the deserialized
feature names, nine-column input and 150 boosting rounds. It restores the
manifest's original hyperparameters and one-thread CPU behavior, never calls fit.
MLP restoration and fitted scaler statistics remain unchanged.

The reconstructed XGBoost matched all 24 originally saved March 1 forecasts and
the original validation/test aggregate metrics. The original full held-period
arrays were not saved; those arrays cannot be claimed as original per-hour
references. The recovery run's newly verified 720 predictions are retained for
subsequent checks. The API reads neither the recovery history nor those references
during serving; they are test fixtures only.

## Launch and readiness

Use the existing macOS OpenMP runtime configuration per process:

```sh
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
PYTHONPATH=src .venv/bin/python -m uvicorn citypulse_ai.api.app:app \
  --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models
```

This setting does not install libraries or modify environment files. Missing
native OpenMP makes XGBoost unready; use startup logs to diagnose it. `/health`
retains its original default-MLP semantics: 200 when MLP is ready, 503 otherwise.
Use `/models` to check each model independently, including if XGBoost is usable
while the default MLP is not. LSTM readiness is checked live through the worker's
health endpoint; TFT is not registered.

`GET /models` returns configured model identities/versions and actual startup
readiness, for example:

```json
{
  "default_model": "pytorch-mlp",
  "models": [
    {"id": "pytorch-mlp", "model_name": "citywide-pytorch-mlp",
     "model_version": "run-20261012T125502822756Z:af2fe71d011d", "ready": true},
    {"id": "xgboost", "model_name": "xgboost",
     "model_version": "january-2024-selected-150-trees:98687a1aefb4", "ready": true}
  ]
}
```

A configured but unavailable model remains listed with `ready: false`. Its
listed version describes configured identity, not a successfully loaded model.
Selecting it gives 503; unknown identifiers give 422. An available model is
never silently substituted for a requested unavailable model.

## Selecting predictions

Use at least 168 hours for MLP/XGBoost or 48 for LSTM (up to 8,760 contiguous
hours for any model); longer LSTM histories are validated before forwarding
only the final 48 hours.
Add a top-level `model` field to that request JSON:

```python
request['model'] = 'xgboost'       # raw-feature tree inference
request['model'] = 'pytorch-mlp'   # explicit existing scaled MLP inference
# Omitting model continues to use pytorch-mlp.
```

For example, write the complete request to `request-xgboost.json`, then:

```sh
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @request-xgboost.json
```

The response schema stays the same for both choices: origin, timezone,
`model_name`, `model_version`, 24 UTC targets, 24 NYC-local targets, their offsets
and 24 predicted ride counts. `model_name` identifies the selected model; the
MLP's legacy name is deliberately preserved. Repeated calls do not reload or
update weights. DST changes calendar labels/offsets, not the 24-hour horizon.

## Tests and adding future models

```sh
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_inference_api.py tests/test_multimodel_api.py -q
```

The original MLP regression tests remain unchanged. New tests verify both saved
March 1 references, raw-feature XGBoost inference without scaler calls or fits,
startup-only loading, model listing, invalid choices, model isolation,
timestamp/leakage validation, DST and deterministic predictions. All corruption
fixtures are new temporary copies, leaving original artifacts intact.

To add a model later, first obtain trusted persisted artifacts and reference
predictions. Implement a model-specific adapter exposing `predict`, `model_name`
and `model_version`; validate its own preprocessing/schema/version/checksum
contract. Register its startup loader and listing metadata, explicitly extend
the request's allowed identifiers, and add parity/readiness/issuance tests.
Share timestamp/feature logic only when that model actually uses this contract.
Changing the default or existing model response behavior needs a separate review.

## Measured verification

- **65 tests passed in 12.68 seconds**, covering the existing 28 MLP API tests,
  29 new multi-model tests and 8 no-fit XGBoost feature/issuance regressions.
  Two synthetic configuration-search tests were intentionally deselected; no
  training was performed. One existing Starlette TestClient deprecation warning
  remains nonblocking.
- Local HTTP `/health` and `/models`: **200**, both configured models ready.
  Predictions for both models: **200**, exactly 24 values, identical repeated
  responses and **zero maximum absolute difference** from their original saved
  March 1 examples. Default and explicit MLP responses were identical.
- Future observations and unknown model selection: **422**. The loader tests
  verify **503** for unavailable XGBoost while MLP health/prediction stays **200**.
- Both model files and their experiment/manifest checksums remained unchanged.
  Smoke evidence is retained in the gitignored `outputs/api-smoke/` structure.
  The temporary test server was shut down after verification.
- No model training, scaler refitting, environment installation, Docker work,
  historical ingestion, cloud operations, commits or pushes occurred.
