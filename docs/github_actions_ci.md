# GitHub Actions inference CI

Continuous integration (CI) runs automated checks when code changes. This
workflow runs on pushes to `main` and pull requests targeting `main`. It checks
code and serving contracts; it does not deploy the application, train models,
build/push Docker images, fetch datasets, restore checkpoints or access GCS.

## Jobs and dependencies

| Job | Runner / Python | Checks |
| --- | --- | --- |
| Repository and Docker configuration | Ubuntu 24.04 x86-64 / 3.11 | Changed-file whitespace; AST syntax for application, tests and utility scripts; workflow YAML and expected safety settings; Dockerfile instruction/serving contract checks; `docker compose config --quiet` |
| Artifact-free gateway contracts | Ubuntu 24.04 x86-64 / 3.14 | `tests/ci/test_gateway.py` |
| Artifact-free TensorFlow worker contracts | Ubuntu 24.04 x86-64 / 3.11 | `tests/ci/test_worker.py` |

The workflow uses the official [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) actions, pinned to
verified release commit IDs (v7.0.1 and v7.0.0). Permissions are limited to
`contents: read`; checkout does not persist Git credentials. No secrets are
required. Superseded runs are canceled. Static checks have a five-minute timeout;
test jobs have fifteen minutes. No step uses `continue-on-error`.

CI dependencies live in `ci/`, separate from the ARM64 container requirements.
The gateway uses the official CPU PyTorch index and same-version `xgboost-cpu`.
The worker uses `tensorflow-cpu==2.21.0` and `keras==3.15.1`; it installs no Torch
or XGBoost. Both include DuckDB/PyArrow because the current forecasting imports
reach those libraries even during serving. Uvicorn, notebook, cloud and training
orchestration dependencies are not needed for TestClient checks. Major serving
versions and direct dependencies are pinned; this is not a fully hashed lock of
all transitive packages. `pip check` and explicit runtime imports fail if the
fresh installation is incompatible. Pip caches contain dependency wheels, never
models or data; their keys depend on the job's requirements file.

## Tests that run without trained artifacts

The CI test modules reuse selected existing assertions, without changing the
original local parity suites:

- Gateway: issuance validation, strict counts, duplicate/order/missing/future
  hours, timezone and midnight rules, DST's 24 elapsed hours, default MLP/model
  selection and metadata, nine-feature order and causal lag values, startup
  loading once, worker's 48-hour forwarding, live readiness, timeout/error
  handling, malformed worker output and Compose URL restrictions. Missing and
  corrupt artifact tests exercise the real loaders' fail-closed behavior.
- Worker: the original `LSTMForecaster` performs real TensorFlow tensor
  operations and scaler transform/inverse-transform against an explicit tensor
  test double. Checks cover float32 `(1,48,1)` input, 24 original-unit outputs,
  `training=False`, repeatability, unchanged declared synthetic scaler statistics,
  single startup load, exact 48-hour validation, DST, missing/corrupt artifacts.

Gateway model loaders are replaced with named protocol doubles; worker weights
are replaced with a non-learned tensor operation. Synthetic counts and scaler
statistics are declared fixtures, **not recovered weights, benchmark scores or
reference predictions**. No fit method is called. These tests establish serving
contracts, not trained-model prediction parity. Training/fit calls are forbidden
in the fixtures, and Path-based reads under the checkout's `outputs/` or
`csv_files/` fail.
`CITYPULSE_CI_REQUIRE_NO_SKIPS=1` makes any collected or runtime skip fail the CI
session, so missing required dependencies cannot yield a green skipped suite.

## Tests kept local

The complete `tests/test_inference_api.py` and `tests/test_multimodel_api.py`
suites require saved MLP/XGBoost artifacts; their saved-example checks also need
small historical outputs or the recovered history file. The complete
`tests/test_lstm_worker.py` requires the trusted LSTM weights, experiment metadata
and recovered history. CI reuses only their artifact-independent contract
assertions with fixture loaders. It does not collect their saved-reference or
real-checkpoint restoration tests.

`tests/test_lstm_gateway.py` looks like a mocked HTTP suite, but its failure
isolation assertions still require a healthy MLP by default. The CI wrapper
provides an explicit protocol model for those assertions. Running that file
alone on a fresh runner would not be sufficient.

The full repository test suite is not selected. Some forecasting tests perform
synthetic fitting/search, while historical/cloud tests cover separate ETL and
storage phases. They are outside this inference-only, no-training CI milestone.
No skipped or missing-artifact check is counted as a parity result. The existing
80 gateway/API and 15 saved-worker checks remain separate local regression
commands documented in the [worker guide](lstm_worker_api.md). Real container
parity remains the explicit [Docker verification](docker_inference.md).

## Local commands and measured verification

From the repository root, use the existing environments without installing the
Linux CI requirements into them:

```sh
.venv/bin/python scripts/check_ci.py
docker compose config --quiet
git diff --check
CITYPULSE_CI_REQUIRE_NO_SKIPS=1 \
  DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
  PYTHONPATH=src .venv/bin/python -m pytest tests/ci/test_gateway.py -q
CITYPULSE_CI_REQUIRE_NO_SKIPS=1 PYTHONPATH=src \
  .venv-tensorflow-311/bin/python -m pytest tests/ci/test_worker.py -q
```

The macOS OpenMP variable is local only; it is not set on Linux runners. Static
checks do not import ML runtimes. Dockerfile checks cover this project's simple
instruction subset and safety rules, not Docker's complete grammar. Native
Compose parsing runs separately without checking artifact existence, launching
services or creating Docker resources. The generic YAML validator does not
execute GitHub expressions; no actionlint binary was available locally.

Measured local results: **67 artifact-free gateway tests and 15 artifact-free
worker tests passed**, with zero skips and the existing TestClient deprecation
warning in each suite. Both suites passed again from a temporary source checkout
with no `outputs/` or `csv_files/` directories, using the existing host runtimes.
An intentionally skipped negative probe exited with status 1, verifying the
required-skip guard. Python AST syntax checks covered 66 files; workflow YAML,
Dockerfile contract checks, native Compose configuration and whitespace passed.
No trained-model parity suite, full project suite, fresh Linux dependency install
or remote workflow was run as part of this CI-only task. No dependencies were
installed into the Mac's environments and no Docker resources were changed.

## First remote run and failures

After a separately authorized commit/push or PR, open the repository's **Actions**
tab, choose **Inference CI**, and open the run for the intended commit. Confirm
all three jobs succeed and both test jobs report passed tests with no skips.
Expand the failing step to read its logs. An installation/import failure is a
runtime or package-index problem; a pytest failure shows the precise contract
and assertion that regressed. Fix the cause and rerun the affected job rather
than adding a skip or `continue-on-error`.

The first remote execution still needs confirmation: local checks use macOS
ARM64 environments, not fresh Linux x86-64 installations. Official CPU wheel
metadata was checked for the required versions, but no Linux installation,
remote action execution or hosted-runner timing has been measured in this task.
GitHub-hosted runners supply Docker Compose; missing Compose, native runtime
libraries, unavailable pinned wheels, disabled Actions, runner capacity and
package-index outages should fail clearly in the corresponding step. The cache
can reduce later downloads but will not eliminate a cold initial installation.
A successful CI run neither publishes a service nor replaces artifact-backed
model validation or deployment resource profiling.
