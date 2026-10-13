# Local three-model Docker demonstration

A Docker **image** is a packaged Linux filesystem containing the application,
Python and libraries. A **container** runs that image as an isolated process.
A **Dockerfile** describes how to build the image. **Compose** starts the two
containers, their private network and artifact mounts together.

The gateway uses Python 3.14, CPU-only PyTorch 2.14.1 and XGBoost 3.4.1.
The separate worker uses Python 3.11, TensorFlow 2.21.0 and Keras 3.15.1.
These files target **Linux ARM64**; the pinned PyTorch wheel is not for x86-64.
OpenMP is provided by Debian's libgomp1. No GPU device is exposed, and the
existing inference adapters keep model execution on CPU.

## Build, start and stop

From the repository root, with Docker Desktop running:

```sh
docker compose config --quiet
docker compose build forecast-gateway
docker compose build lstm-worker
docker compose up -d
```

Build sequentially to reduce simultaneous downloads and installation memory.
Check readiness with `docker compose ps` and `docker compose logs`.

```sh
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @request.json
docker compose stop
```

Use the history format in [the API guide](local_mlp_api.md). Set `model` to
`pytorch-mlp`, `xgboost`, or `tensorflow-lstm`; omitting it still selects MLP.
MLP/XGBoost need 168 consecutive observed hours, LSTM needs 48. All observations
precede NYC-local midnight issuance. UTC defines the 24 elapsed target hours;
NYC labels retain DST offsets. Unknown hours are rejected rather than imputed.
No training or scaler fitting occurs.

`stop` stops this project's containers and preserves images and containers.
No pruning or automatic resource deletion is part of this workflow.

## Networking and artifacts

Within a container, localhost refers to that container. The gateway therefore
uses `http://lstm-worker:8001`, resolved by Docker's internal network. The explicit
operator flag `CITYPULSE_LSTM_COMPOSE_NETWORK=1` permits only that additional
hostname. Local non-container launches retain the loopback-only default.
Callers cannot choose worker URLs. The worker network has no outbound access. The gateway also joins a separate
bridge because this Docker Engine requires it for host port publishing. Only
gateway port 8000 is published, bound to host **127.0.0.1**; the worker has no
published host port. The gateway bridge permits outbound access; no cloud
requests are made by the application.

A **bind mount** makes an existing host directory visible to the container.
Compose mounts the original MLP run directory, recovered XGBoost directory and
LSTM run directory read-only under `/artifacts`. Missing directories fail rather
than being created automatically. The trusted artifact SHA-256 checks still run
at startup. These artifacts include the fitted scaler statistics in the saved
MLP checkpoint and LSTM experiment metadata. Neither requests nor containers
can change their identity. Preserve these host directories to run the service.

Both containers run as UID 10001, drop Linux capabilities and use read-only
filesystems except for bounded 64 MiB scratch `/tmp`. `.dockerignore` allows
only application source and container definitions into the build context;
no outputs, weights, datasets, credentials, notebooks or Python environments
are embedded. DuckDB/PyArrow are necessary because existing forecasting helper
imports reach the historical modules; no ETL is run. TensorFlow is absent from
the gateway, and PyTorch/XGBoost are absent from the worker's installed libraries.

A **health check** periodically calls each container's own `/health`. Gateway
health retains default-MLP readiness; `/models` checks all three individually.
LSTM is deliberately not a startup dependency, so a failed worker does not
prevent MLP/XGBoost startup. LSTM network failures return 503, timeouts 504,
and invalid worker responses 502. Start or restart the worker to recover:

```sh
docker compose stop lstm-worker
docker compose start lstm-worker
```

## Resources and limitations

Each service is limited to 1.5 GiB memory and one CPU, with no additional
container swap allowance. The combined 3 GiB limit
leaves approximately 0.84 GiB of the current Docker VM allocation for overhead.
The limits apply to running services, not BuildKit build steps. Binary wheels,
sequential builds and disabled pip caches limit build costs. Maintain at least
10 GiB host free space; check `df -h .`, `docker system df`, and `docker stats`.
Images and build caches remain on disk until explicitly managed by the user.

The requirements pin direct and resolved transitive Python dependencies, and
Dockerfiles pin base-image digests. Debian security packages installed with apt
can still change on a later rebuild; image IDs identify the verified filesystem.
The CPU PyTorch wheel URL includes its SHA-256. These local demo definitions
are not a complete production supply-chain lock.

The models remain preliminary winter citywide experiments. They have different
inputs and are not the final station-level benchmark. Timestamp validation
assumes observations were available at issuance; it cannot establish real
publication latency. Linux results must be compared with saved predictions;
macOS parity alone does not establish container parity.

Run the host-only verification script with `.venv/bin/python docker/verify_inference.py`.
It reads existing compact reference histories, starts this Compose project, checks
all three models and worker restart, then stops its containers. It saves a unique
audit under `outputs/docker-verification/`. It never fits or modifies artifacts.
Cross-platform parity uses `atol=0.001`, `rtol=1e-6`; actual differences are recorded,
not described as exact unless zero.

## Verified local run

Docker Desktop 4.17.0 / Engine 20.10.23 / Compose 2.15.1 ran the images on
Linux ARM64 with 4,123,439,104 bytes (3.84 GiB) VM memory. The final images are:

| Image | Image ID prefix | Uncompressed image size |
| --- | --- | --- |
| `citypulse-forecast-gateway:local` | `396c42b1ee27` | 1,497,364,939 bytes (1.50 GB) |
| `citypulse-lstm-worker:local` | `bafd5de67630` | 2,239,982,230 bytes (2.24 GB) |

Both containers became healthy; gateway `/health` and `/models` returned 200.
All three `/predict` selections returned 200 with exactly 24 predictions.
XGBoost and LSTM matched all 24 saved March 1 reference values exactly. MLP's
maximum Linux/macOS difference was **0.0009765625 rides**, within the declared
float32 tolerance; it is not bit-for-bit cross-platform parity. Repeated and
concurrent Linux requests were identical. Default MLP, both DST transitions,
future-input rejection and non-midnight rejection passed. Stopping the worker
made LSTM unready and returned 503 while MLP/XGBoost continued responding;
restarting the worker restored LSTM predictions. All six artifact hashes stayed
unchanged. Relevant host suites passed: **80 gateway/API tests + 15 worker tests**
with the existing TestClient deprecation warnings. Whitespace checks passed.

The final run sampled combined Docker memory usage up to **603.2 MiB** across
eight samples (six with both containers running); individual sampled maxima were
300.1 MiB gateway and 303.2 MiB worker. An earlier run recorded a combined
**633.0 MiB** over 18 samples. These are Docker's cache-adjusted sampled container
usage, not whole-VM memory or guaranteed lifetime peaks. This kernel does not
expose the probed cgroup peak counter. No OOM was reported; final shutdown exit
codes were zero. Larger histories, sustained concurrency and cold-start peaks
still require broader profiling; these observations do not prove universal
memory safety within the VM allocation.

Host free space changed from **28.80 GiB before any build to 27.09 GiB after
verification**, preserving the 10 GiB reserve. Docker reports **7.168 GB of image
storage** across five retained images, including intermediate builds, and about
**226 kB** of separately reported build cache. Docker layer sharing, compression
and filesystem/system activity mean these figures are not additive measurements
of the host free-space change. No pruning was performed. Final containers are
stopped; their images and artifacts remain available.

Detailed successful final evidence is under
`outputs/docker-verification/7507e03ea6634da08cbceca83ccbf9f6/`.
The earlier attempt's internal-only gateway network could not publish the host
port on this Engine; the gateway-only bridge fixes that without exposing the
worker. The generic XGBoost wheel requested NCCL, so the official same-version
CPU distribution was used and reference-verified. Neither issue required model
changes, Docker settings changes or installation into the host environments.
