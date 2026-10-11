# Python Google Cloud Storage integration

Install the optional client library in the existing virtual environment:

```sh
.venv/bin/python -m pip install -e '.[cloud]'
.venv/bin/python -m pytest tests/test_gcs.py
```

`citypulse_ai.cloud.gcs.GCSStorage` uses Application Default Credentials (ADC).
It creates no keys and embeds no credentials. Unit tests inject a mock client;
importing the package does not authenticate or contact Google Cloud.

```python
from pathlib import Path
from citypulse_ai.cloud.gcs import GCSStorage

store = GCSStorage(
    project="citypulse-ai-511315",
    bucket="citypulse-ai-511315-data",
)
# The bucket's region is configured in GCS; it is not a client parameter.
sha256 = store.upload_file(Path("tiny.txt"), "smoke-tests/example.txt")
assert store.object_exists("smoke-tests/example.txt")
store.download_file(
    "smoke-tests/example.txt", Path("tiny-downloaded.txt"),
    expected_sha256=sha256,
)
```

Uploads use a generation precondition of zero to create new objects without
replacing existing ones. Downloads require an existing parent directory and a
new destination path. Both operations disable SDK automatic checksum cleanup,
which can delete corrupt objects or local files. Instead, downloads explicitly
verify SHA-256 against the caller's trusted digest. A mismatch or transfer failure
raises a clear exception and retains downloaded bytes for inspection. No wrapper
operation deletes files. Hashing reads 1 MiB blocks; transfers use the SDK's
streaming file methods with 60-second request timeouts and retries disabled.

`upload_file` returns the SHA-256 of the source bytes. Do not modify a file during
upload. The source hash alone is not proof of cloud integrity: a successful,
verified download establishes that the downloaded bytes match it.

Authentication, permission, and transfer failures raise `GCSOperationError`
with the underlying exception attached. Missing upload sources raise
`FileNotFoundError`; existing download destinations raise `FileExistsError`;
malformed names or expected digests raise `ValueError`. Hash mismatches raise
`ChecksumMismatchError`. After an uncertain upload, inspect the existing object
before retrying; do not delete or overwrite it automatically.

The approved smoke-test object is
`gs://citypulse-ai-511315-data/smoke-tests/python-adc-check.txt`.
Its cloud object and temporary local files are intentionally retained. No real
Citi Bike data is used or uploaded, and the integration test is not part of the
automated unit suite.

## Publish one existing Parquet artifact

The publishing command accepts a local Parquet path and an **exact object URI**.
It refuses files at or above 100 MiB, inspects the Parquet footer, uploads the
unchanged bytes, then verifies a retained temporary download. It does not run
ETL, alter datasets, change bucket permissions, or make the object public.

```sh
mkdir -p outputs/gcs-publications
PYTHONPATH=src .venv/bin/python -m citypulse_ai.cloud.publish \
  outputs/phase13-february-reviewed/february_2025_hourly_rides.parquet \
  gs://citypulse-ai-511315-data/curated/forecasting/2025-02/reviewed-v1/february_2025_hourly_rides.parquet \
  --project citypulse-ai-511315 \
  --report outputs/gcs-publications/february-2025-hourly-reviewed-v1.json
```

Installing the updated project with `pip install -e '.[cloud]'` also registers
`citypulse-publish`, with the same arguments. The module form above explicitly
sets the `src/` import path and works even if that console script is unavailable.

Object names are supplied by the caller; the command does not append filenames,
choose versions, URL-decode names, or generate monthly datasets. The example
uses `curated/forecasting/<local-start-month>/<reviewed-version>/<filename>`.
`reviewed-v1` identifies this existing February baseline, **not** a complete
24-month forecasting release. In particular, publishing these bytes does not
add the February starts later found in March's archive.

The object retains all embedded Parquet metadata because bytes are unchanged.
Custom GCS metadata records the source path and filename, SHA-256, byte size,
row count, schema fingerprint, UTC publication time, and hashes of adjacent
`report.json` / `verification.json` files when present. The receipt also records
the complete Arrow schema, cloud destination, verification outcomes, and local
download path. The validation reports themselves are not uploaded. Do not put
credentials or sensitive path names into publication metadata.

Cloud writes retain `if_generation_match=0`: an existing object causes failure,
including when its bytes happen to match. Existing receipt paths are also
protected with exclusive creation. Choose a deliberate new version and receipt
name for a later authorized publication. There is no overwrite or delete flag.
[Google's precondition documentation](https://docs.cloud.google.com/storage/docs/request-preconditions)
describes this generation-zero protection.

## Verify a publication

Publication automatically verifies object accessibility through ADC, custom
metadata, SHA-256, Parquet row count, and schema equality including schema
metadata. A successful receipt has `uploaded: true`, `verified: true`,
`metadata_match: true`, and all `checks` set to true. Inspect it with:

```sh
cat outputs/gcs-publications/february-2025-hourly-reviewed-v1.json
```

A later read-only verification can reuse the recorded digest and destination
without uploading again:

```python
import json
from pathlib import Path
import tempfile
from citypulse_ai.cloud.gcs import GCSStorage
from citypulse_ai.cloud.publish import parse_destination, verify_parquet

receipt = json.loads(Path(
    "outputs/gcs-publications/february-2025-hourly-reviewed-v1.json"
).read_text())
bucket, name = parse_destination(receipt["destination"])
store = GCSStorage(receipt["project"], bucket)
copy = Path(tempfile.mkdtemp(prefix="citypulse-gcs-reverify-")) / "copy.parquet"
store.download_file(name, copy, expected_sha256=receipt["sha256"])
print(verify_parquet(Path(receipt["source_path"]), copy))
# Both source and download are retained; this makes no cloud modifications.
```

Failed uploads or verification attempts return a nonzero CLI exit status. If a
receipt was opened, it records the failure and whether upload completed. A cloud
object may remain after a later verification failure, or after an upload whose
response was interrupted. Inspect that state before retrying; no automatic
cleanup occurs. Receipt parents must already exist.

Run focused tests without credentials or cloud requests:

```sh
.venv/bin/python -m pytest tests/test_gcs.py tests/test_gcs_publish.py -q
```

## Completed real verification

The ADC smoke test uploaded and downloaded 36 bytes at
`gs://citypulse-ai-511315-data/smoke-tests/python-adc-check.txt`; SHA-256 matched.
The subsequent real-artifact publication at the example URI above completed:

- Source: `outputs/phase13-february-reviewed/february_2025_hourly_rides.parquet`.
- Size: **946,546 bytes (0.903 MiB)**; **598,630 station-hour rows**.
- SHA-256: `a5058fa781b33205da104ed9ffe332d4673adcfe32bed5fdc5fc3804d1bbcf3b`.
- Verified download: identical SHA-256, row count, and schema including metadata.
- ADC object access, existence, and custom provenance metadata checks passed.
- Schema: `start_station_id: string`, `hour_start: timestamp[us, tz=UTC]`,
  `ride_count: int64`.
- Local receipt: `outputs/gcs-publications/february-2025-hourly-reviewed-v1.json`
  (ignored by Git); downloaded files and cloud artifacts remain intact.

The completed implementation passed **46 focused GCS/publishing tests**, and the
full suite passed **158 tests plus 6 subtests**. These measured cloud results are
from the completed publication, not a cloud request made during the Git checkpoint.
Repeating the example upload would fail its no-overwrite precondition; use the
read-only verification example when checking the already published artifact.

## Historical ingestion integration

The historical CLI now has an explicit `--cloud-publish` stage and a
`--publish-existing-only` mode for already processed source partitions. It reuses
this wrapper and the publisher's verification helpers; it does not change the
single-artifact command's fixed size limit or overwrite behavior.

See [Phase 2.2D commands and recovery policy](historical_ingestion.md#phase-22d--optional-gcs-publication-after-etl)
for one-month processing, publishing without ETL, retry behavior, size ceilings,
and local manifest status inspection. The historical manifest records verified
object generations; downloads pin both the blob generation and the matching
precondition. Reconciliation and canonical snapshot publication remain separate.
This phase performs no real cloud transfers during tests.

Source ZIP/CSV backups, opt-in cloud resume, verified restore, and the offline
retention dry run are documented in [Cloud-backed storage retention](cloud_retention.md).
This adds no automatic deletion and requires separate authorization for real transfers.
