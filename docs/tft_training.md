# CPU training of the final station TFT

The dataset implementation is committed separately from this training driver.
`forecasting/tft_training.py` uses the existing native PyTorch TFT unchanged:
100 station codes, hidden size 16, two attention heads, dropout 0.1, 336 encoder
hours, 24 decoder hours and ordered q10/q50/q90 outputs (62,511 parameters).

Only train and validation streams are accepted. Full dataset integrity hashes
include test artifacts, but no test observations are loaded for fitting, model
selection or metrics. Station-specific preprocessing remains frozen from the
verified dataset. Predictions are inverse-scaled without clipping; the report
includes the fraction of negative median forecasts.

Training uses Adam at 0.001, gradient norm clipping at 1, batch size 32, seed 42,
CPU threads 2 and deterministic PyTorch algorithms. Each epoch visits every
training window once, including final short batches. Stations and their origins
are shuffled deterministically in station blocks. This ordering permits one
station's unique-hour frame to be cached, bounding memory and avoiding repeated
Parquet reads; repeated window tensors are never cached. There are no loader
subprocesses. Eight disposable real-training batches estimate throughput; their
weights are discarded and the actual model is reset to the declared seed.

The initial schedule is at most ten complete epochs, with early stopping after
three validation losses fail to improve by 0.0001. Selection uses standardized
masked validation pinball loss. Training and validation losses are weighted by
trusted target count. Original-unit MAE, RMSE, WAPE (%), per-quantile pinball,
mean pinball and inclusive q10–q90 interval coverage are reported on validation.
Lag-24 and lag-168 baselines use exact accepted recorded historical counts on
the identical station/origin/target keys. Their observations end by issuance;
calendar DST behavior is the existing 24-elapsed-hour contract. Baselines are
point forecasts and have no fitted prediction intervals.

Completed epoch checkpoints contain model/optimizer state, torch RNG, epoch,
configuration, dataset checksum, code checksum, losses and best-checkpoint state.
Checkpoint restoration is exact-tested before publication. Each epoch is
published and generation/checksum-verified before the next begins. A later
failed or partial epoch never replaces a completed checkpoint. The run checks
remaining time before starting each epoch; a budget stop is explicitly partial
training, not convergence. CPU throughput measurements are feasibility estimates,
not benchmark accuracy results.

The existing GCP VM uses `.venv-tft`, Python 3.11 and official CPU-only
`torch==2.14.1+cpu`. Its `.pth` references the existing verified ETL data/cloud
libraries read-only. Torch and its dependencies are installed only in the new
venv; before/after ETL `pip freeze` must match. No GPU runtime is needed. Keep the
512 MB DuckDB limit, 2 GiB process RSS guard, zero-swap and host-pressure checks,
15 GiB minimum disk reserve, and four-hour server-side VM STOP. The deployment
wrapper applies these guards; the plain CLI below does not provision or change
VM safeguards. Estimate runtime and refresh the server deadline before resuming.

```bash
cd ~/CityPulse-AI
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv-tft/bin/python -B \
  scripts/train_station_tft.py \
  --dataset outputs/tft-final/station100-20261024b \
  --output-dir outputs/tft-training/station100-20261024 \
  --resume outputs/tft-training/station100-20261024/epoch-003.pt \
  --max-seconds 10800 --batch-size 32 --max-epochs 10 --additional-epochs 3 \
  --gcs-prefix models/tft/station100-20261024/epochs
```

The recorded 2026-10-24 session completed epochs 1–3 and stopped for its runtime budget. Epoch 3 is both the latest and best checkpoint; full training remains incomplete. For a later session, use its latest verified complete epoch instead. Preserve the same code,
dataset, settings and existing run directory; restore the best checkpoint too
if it differs from the latest. The recovery ZIP contains both, preprocessing,
code, runtime/configuration, metrics and cloud receipts. Earlier checkpoints
remain available through their separate pinned receipts. A fresh environment
must verify the recovery ZIP's published generation and SHA-256 before loading.

Validation prediction exports are streamed to Parquet; no test exports exist.
The experiment JSON is the source of truth and can be logged through the already
installed local MLflow environment, recording dataset/checkpoint GCS references
rather than duplicating weights or installing MLflow into the CPU worker. Stop
the VM after completion or failure; retained disk and GCS storage keep consuming
trial credits. Do not delete protected data or upgrade billing automatically.

For a resumed VM session, apply the same guard to the command above. Obtain the
new four-hour termination timestamp from Compute Engine before starting; set
`CITYPULSE_TFT_VM_DEADLINE` to that timezone-aware UTC value. Use a new audit
folder, retain the existing run directory, and replace the completed checkpoint
filename. The wrapper below is reproducible and does not modify the trainer:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv-tft/bin/python -B - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import os, runpy, shutil, sys
from citypulse_ai.etl.monthly_cycle import BoundedRuntime
from citypulse_ai.etl.resource_monitor import snapshot
root = Path('outputs/vm-historical')
work = Path('outputs/tft-training/station100-20261024')
report = Path('outputs/tft-training-audits') / datetime.now(timezone.utc).strftime('resume-%Y%m%dT%H%M%SZ')
deadline = datetime.fromisoformat(os.environ['CITYPULSE_TFT_VM_DEADLINE'])
seconds = min(10800, int((deadline-datetime.now(timezone.utc)).total_seconds()-1200))
assert seconds >= 1800 and shutil.disk_usage(root).free >= 28*1024**3
state = snapshot()
assert state['swap_used_bytes'] == 0 and not state['memory_pressure_unsafe']
report.mkdir(parents=True, exist_ok=False)
sys.argv = ['scripts/train_station_tft.py', '--dataset',
    'outputs/tft-final/station100-20261024b', '--output-dir', str(work),
    '--resume', str(work/'epoch-003.pt'), '--max-seconds', str(seconds),
    '--batch-size', '32', '--max-epochs', '10', '--additional-epochs', '3', '--gcs-prefix',
    'models/tft/station100-20261024/epochs']
with BoundedRuntime('2025-12', root, work, report) as guard:
    runpy.run_path('scripts/train_station_tft.py', run_name='__main__')
    guard.stage('complete')
PY
```

Use a bounded systemd scope and verified recovery publication/shutdown finalizer
as in the recorded run; do not rely on the SSH connection for runtime control.
The finalizer waits for training to finish, verifies successful resource reports,
publishes/restores the recovery bundle once, and halts the existing VM even if
publication fails. It does not retry ingestion/training or delete the disk.

## Bounded continuation and validation clipping

`--additional-epochs 3` resumes epoch 3 for at most epochs 4–6 while preserving
its original ten-epoch configuration, optimizer, RNG, history, global best and
patience counter. A session-cap stop is partial training, not convergence.
The reviewed initial driver SHA-256
`2697566a11e6610a2f6256212e4a4cced778e0515e491eb35efd5ea7b910ecc4`
is explicitly compatible: the continuation changes session stopping and reporting,
not optimization. Unknown driver hashes, changed settings or dataset identities
still fail closed. Each new run records both the previous and current code hashes.

Validation reports raw forecasts and the predeclared alternative
`q_clipped = max(0, q_inverse_scaled)` for each of q10/q50/q90. Applying the same
monotone floor preserves ordering. Report MAE, RMSE, WAPE, pinball and interval
coverage separately on identical trusted keys. Negative frequency refers to raw
median predictions. Clipping affects neither training nor checkpoint selection,
and no test observations are used to choose the rule. Its scientific meaning is
nonnegative accepted recorded starts, not a claim about unmet physical demand.

Continue the existing MLflow run ID on resume. Keep immutable original parameters;
record current epoch/best checkpoint as tags and step-indexed metrics. Before
logging each epoch, compare existing metric history and skip exact matches;
a conflicting value is an integrity failure. Record only small reports and GCS
references, never copy weights or datasets into MLflow. Update after each verified
epoch, including a continuation that later stops for resources or runtime.
