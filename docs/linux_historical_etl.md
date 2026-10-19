# Historical ETL on an approved Linux VM

This is an operating procedure, not authorization to create a VM, enable APIs,
change IAM, transfer data or start ingestion. No such operations were performed
while preparing this code. The intended pilot is Debian 12 / Python 3.11,
`e2-standard-4`, a 100 GiB disk and the existing Frankfurt bucket.

## What changed

`resource_monitor.py` reads macOS kernel pressure and swap counters, or Linux
`/proc/meminfo`, `/proc/self/status` and memory PSI. Linux `VmRSS` and
`ru_maxrss` are KiB; macOS `ps` RSS is KiB but `ru_maxrss` is bytes. Every reported
RSS value is converted to bytes. Linux `MemAvailable` is reported explicitly;
macOS does not invent an available-RAM estimate from a free-page percentage.

The existing 512 MB DuckDB limit, 2 GiB process RSS ceiling, 2 GiB swap-capacity
growth ceiling, workspace/file/spill ceilings, disk margins and default 28 GiB
starting gate remain. Swap **use** growth above 2 GiB now also stops execution.
Both swap measurements are host-wide and cannot identify which process caused
growth. No swap files are changed. The monitor also refuses unsafe initial
pressure before opening an engine and preserves that first sample.

On macOS, kernel pressure levels 2 (warning) and 4 (critical) stop the cycle.
Linux stops if MemAvailable falls below the greater of 256 MiB or 5% of physical
RAM, or memory PSI `full avg10` reaches 10%. These are conservative operating
thresholds, not equivalent OS pressure scales. If PSI is unavailable, the report
says so and still enforces MemAvailable, RSS, swap and disk guards. Missing or
malformed required counters fail closed. Sampling cannot guarantee a limit is
never crossed between samples; an independent VM runtime deadline is required.

## Recovery contract and protected local files

`vm_recovery export` creates a new offline bundle containing the original
historical manifest (including receipt/generation state), completed global audits
and independently checksummed intermediate checkpoint files. It excludes raw
archives/CSVs, datasets, monthly DuckDB files, canonical snapshots, credentials
and models. Stop ingestion/reconciliation writers before export. Sources are
hashed before/after copying; a changing or corrupted bundle is refused and kept.
The printed SHA-256 of `bundle.json` must be recorded independently of the copied
bundle and supplied on restore. It proves transfer identity, not authorship.

Restore creates an owned **separate** VM root. It changes only each entry's
working `output_dir`; original source paths, attempt history, source hashes,
publication revisions, generations and receipt paths remain provenance. The
original manifest remains byte-for-byte in the bundle. Existing VM manifests
that have advanced through an ingestion attempt are retained, not reset. Every
old processed entry must still match its original mapped entry exactly.

The existing `restore_partition` downloads only validated/rejected inputs,
processed JSON audits and immutable source processing manifests. It checks live
generation/provenance metadata and downloads with the recorded generation and
SHA-256. Matching local files are reused. Changed, partial or missing objects
stop recovery; conflicts are never overwritten or deleted. An interrupted
download leaving invalid bytes requires an explicitly chosen new restore root.
The restorer takes the existing single-writer lock and checks a 15 GiB reserve.

Checkpoint identities, stage names, payload checksums and asset hashes are
verified without rewriting paths inside audit payloads. The existing reconciler
decides reuse from the exact input/receipt identity. An old ten-archive checkpoint
does not authorize skipping the new eleven-archive audit after September.

The previous read-only inventory found approximately 2.31 GiB of required cloud
reconciliation inputs, 1.29 GiB of local monthly DuckDB files, 2.47 GiB of local
canonical snapshots and 63 MiB of local reconciliation reports/checkpoints.
Sizes are historical observations, not enforced forecasts. The latter three
groups are not covered by monthly GCS publication receipts. Preserve them on the
Mac. The VM does not need old DuckDB files or persistent canonical catalogs to
perform temporary reconciliation; they must not be silently relocated or deleted.
Rebuilding a new persistent catalog is a separate task. No old persistent view
is rebound: the fresh VM root intentionally has no canonical catalog.

## Commands after VM and transfer approval

Commands below are examples to execute only after the cloud pilot is authorized.
Replace `YOUR_APPROVED_VM` and the trusted bundle digest. Do not copy ADC files or
service-account keys. Use the VM's attached, narrowly scoped service account.

On the VM, obtain the code and install only ETL/cloud/test dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git python3.11-venv
git clone https://github.com/sanjeevani1193/CityPulse-AI.git
cd CityPulse-AI
git checkout 13afe4c0c859a34635f085f74595622555226bab
python3.11 -m venv .venv
```

That commit is the reviewed base, and does not include this uncommitted Linux
preparation. Until a new revision is separately committed/pushed, transfer these
six reviewed files from the Mac, with the repository already cloned on the VM:

```bash
CITYPULSE_VM=YOUR_APPROVED_VM
gcloud compute scp src/citypulse_ai/etl/monthly_cycle.py \
  src/citypulse_ai/etl/resource_monitor.py src/citypulse_ai/etl/vm_recovery.py \
  "$CITYPULSE_VM:~/CityPulse-AI/src/citypulse_ai/etl/" \
  --project=citypulse-ai-511315 --zone=europe-west3-a
gcloud compute scp tests/test_monthly_cycle.py tests/test_resource_monitor.py \
  tests/test_vm_recovery.py "$CITYPULSE_VM:~/CityPulse-AI/tests/" \
  --project=citypulse-ai-511315 --zone=europe-west3-a
```

Alternatively, check out the subsequently approved commit containing these files.
On the VM:

```bash
.venv/bin/python -m pip install -e '.[cloud,dev]' \
  duckdb==1.5.6 pandas==3.0.6 pyarrow==23.0.1 pytz==2026.5 \
  google-cloud-storage==3.17.0
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  tests/test_resource_monitor.py tests/test_vm_recovery.py tests/test_monthly_cycle.py
```

These versions record the existing ETL environment. Linux wheel installation and
native behavior still require this actual VM check; stop on incompatibility.
Do not install the forecasting/neural/TensorFlow/API extras for ETL.

On the Mac, export to a **new** directory and record the printed digest:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  -m citypulse_ai.etl.vm_recovery export \
  --historical-root outputs/historical --bundle /private/tmp/citypulse-recovery
gcloud compute scp --recurse /private/tmp/citypulse-recovery \
  "$CITYPULSE_VM:~/citypulse-recovery" \
  --project=citypulse-ai-511315 --zone=europe-west3-a
```

On the VM, restore into a fresh root, keeping the bundle separate:

```bash
cd ~/CityPulse-AI
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  -m citypulse_ai.etl.vm_recovery restore \
  --bundle "$HOME/citypulse-recovery" --bundle-sha256 TRUSTED_64_HEX_DIGEST \
  --output-root "$PWD/outputs/vm-historical"
```

The restore command does not ingest September, restore old raw archives, upload
objects, or build a canonical snapshot. Rerunning it rechecks live identities
and local checksums without redownloading matching files.

## Resume September and verify

September has no validation-batch checkpoint. Its incomplete validation must
restart; completed historical archives are not reingested. The normal downloader
reuses retained verified ZIPs/CSVs when available. On a fresh VM it retrieves the
official September ZIP and extracts there. Both the recorded publisher identity
and original SHA-256 are enforced **before validation**. A mismatch stops instead
of accepting a changed source. No Mac CSV upload is necessary. If the official
source changed, stop and arrange a separately approved transfer of the retained
verified ZIP and receipts rather than discarding the old source identity.

Use the default 28 GiB gate on the 100 GiB VM, not the Mac-only September budget
exception (which also needs an August resource report not included in this bundle).
With the independent four-hour VM STOP deadline already configured, run on VM:

```bash
set -o pipefail
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 timeout --signal=INT --kill-after=90s 3h \
  .venv/bin/python -u -B -m citypulse_ai.etl.monthly_cycle \
  --month 2024-09 --output-root "$PWD/outputs/vm-historical" \
  --work-root "$PWD/outputs/monthly-work/2024-09"
```

The existing runner performs ingestion, processed/source publication with pinned
verification, temporary reconciliation, cached idempotence and integrity checks
in sequence. Do not run a second parallel publisher/reconciler. On any safety or
integrity failure, inspect `stopped.json`, `resources-progress.json`,
`resources.json` when available, and the working manifest. Do not automatically
retry. Resource monitoring covers host activity, not just CityPulse.

Inspect the new run's `result.json`, ingestion counts/reasons, publication receipts
and reconciliation audit. Require accepted + rejected = source rows, exact global
duplicate results, equal hourly/accepted totals and unique station-hour keys.
Finalize August only with the complete global checks; September stays provisional
until October. The cached rerun must report identical results with SQL forbidden.
The zero persistent views on the VM are intentional and do not verify the Mac's
existing ten bindings; those protected catalogs remain on the Mac.

After success **or interruption**, export a fresh recovery bundle from
`outputs/vm-historical` to a new directory outside it. Record its digest and copy
it off the VM (for example using approved `gcloud compute scp --recurse` back to
a new Mac destination), then verify using `verify_bundle` and that digest. This
is still a required explicit transfer: monthly publication does not automatically
publish the working manifest or global checkpoint tree to GCS. Do not delete the
VM disk until this state and all required outputs have independently verified
recoverable copies. Future create-only GCS bundle publication can reuse existing
storage helpers, but is not implemented or performed by this preparation.

Stop the VM from the Mac:

```bash
gcloud compute instances stop "$CITYPULSE_VM" \
  --project=citypulse-ai-511315 --zone=europe-west3-a
```

Stopping preserves its disk and that disk remains billable. Resource deletion
requires separate approval after recovery verification. Do not configure a
deadline that deletes the only copy of incomplete work. October is the next
cycle only after September is verified; no bulk mode or final TFT training is
part of this procedure.
