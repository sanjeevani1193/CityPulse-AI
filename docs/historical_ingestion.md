# Historical ingestion: Phase 2.2A

This phase expands ingestion, not model training. The complete platform roadmap
is unchanged. The final forecasting specification is fixed:

| Item | Specification |
| --- | --- |
| Local start window | `[2024-01-01, 2026-01-01)`, America/New_York |
| Training | `[2024-01-01, 2025-07-01)` |
| Validation | `[2025-07-01, 2025-10-01)` |
| Test | `[2025-10-01, 2026-01-01)` |
| Targets | Hourly citywide and station-level validated ride starts |
| Horizon | Next 24 **elapsed** hours |
| Station cohort | 100 stable stations selected from training history only |

The February-only splits in Phase 2.1 remain an educational example; they are
not the final historical split. This ingestion phase records the fixed
specification but does not select the cohort, zero-fill grids, create forecast
samples, train models, or integrate weather. A 24-hour forecast horizon does not
mean a 24-hour maximum trip duration. The reviewed ETL still accepts any strictly
positive, representable duration.

## Official sources and compatibility

The [official System Data page](https://citibikenyc.com/system-data) links to the
[trip-history archive](https://s3.amazonaws.com/tripdata/index.html). Review the
linked data-use policy. The official page says larger months can contain multiple
CSVs; never ingest only the first part and call it a complete archive.

S3 listings verified the modern NYC names `YYYYMM-citibike-tripdata.zip` for all
24 requested months, and January 2026. Jersey City `JC-` archives are not selected.
The archive reader supports CSVs in subdirectories, ignores macOS resource files,
rejects unsafe paths/symlinks/duplicate member names, and fails visibly on an
unsupported archive structure rather than guessing an adapter.

The January 2024 pilot contains **2 CSVs**, March 2025 **4**, and existing February
2025 **3**. Across the 24 archive directories, member counts range from **2 to 6**.
The two pilot ZIPs store their CSVs without compression: a ZIP extension does not
guarantee a storage saving. Each pilot CSV has the same observed 13 columns:

```text
ride_id, rideable_type, started_at, ended_at, start_station_name,
start_station_id, end_station_name, end_station_id, start_lat,
start_lng, end_lat, end_lng, member_casual
```

Both pilot previews have fractional-second timestamps. No required-schema adapter
was needed for these sources. Other months' headers have **not** all been sampled;
each header is checked at ingestion against the existing five-column contract.
Missing columns cause a recorded failure, leaving raw data intact. Extra columns
are recorded in the manifest and ignored by the unchanged trip validator.

Archive modification dates and ETags identify the publisher versions inspected.
Historical archives can be republished; a year/month filename alone is not a
version identifier. Existing February is registered by reference with its known
CSV hashes and output hashes. Its original ZIP SHA-256 is unavailable, and the
manifest says so explicitly.

## How the tool works

1. **Discover and plan:** query the official listing and fetch only ZIP-directory
   ranges. Estimate retained ZIP/CSV/output storage plus staging/spill and a
   default 5 GiB safety reserve. Check the configured filesystems.
2. **Download:** stream 1 MiB blocks to `.part`; record URL, ETag, and expected size.
   A restart uses HTTP `Range` and `If-Match`. Refuse ignored ranges or changed
   versions rather than append the wrong bytes. Verify final byte length and
   record SHA-256. Multipart S3 ETags are identity tokens, **not MD5 checksums**.
   There is no publisher-provided SHA-256 here; HTTPS/object identity and ZIP CRC
   checks support transport verification, while local SHA-256 supports reproducibility.
3. **Extract:** stream one member at a time. Full reads verify ZIP CRCs; publish
   completed CSVs only after verification and record their hashes. Preserve both
   archives and extracted CSVs. Resume completed members by hash. Incomplete
   extraction scratch is regenerated, not appended to an unverified CSV.
4. **Sample then ingest:** call the existing `run_pipeline` with all CSV parts in
   one run, 50,000-row chunks and a 512 MB DuckDB limit. Samples are separate from
   full partitions and never registered as complete sources. The original
   timestamp, duration, station, duplicate, and rejection rules are unchanged.
5. **Verify and publish:** independently query counts, unique station-hour keys,
   and timestamp epoch consistency. Publish a completed source partition by
   directory rename, then atomically update the manifest. Interrupted ETL restarts
   that partition; it does not attempt an unsafe mid-query checkpoint. A run
   interrupted after directory publication can be verified and registered on retry.
6. **Reconcile:** scan every complete registered source, including rejected IDs,
   using disk-backed DuckDB queries. Reject **every occurrence** of a nonblank
   duplicate ride ID, consistent with Phase 1. Emit the additional global rejection
   audit and write new canonical partitions by NYC local **start month**.

The CLI allows one manifest writer at a time through an OS file lock. Lock files
may remain on disk, but the OS releases ownership when the process exits. Direct
Python API callers must serialize mutations of the same output root. Failed
attempts retain error text and status. Verified successful partitions are skipped,
including when a later range command requests sampling. Corrupted completed
outputs cause an error; they are not silently overwritten.

The existing ETL's extra `february_2025_hourly_rides.parquet` and view are retained
inside source partitions for compatibility. They may be empty for other archives;
use the canonical start-month outputs for historical analysis.

## Boundary trips, duplicates, and precision

A publication month is not a ride-start month. Preserve `source_archive_month`,
`source_file`, and `source_row` on canonical trips. Assign `start_month` from the
UTC start instant converted to America/New_York. A trip spanning midnight or
month-end contributes exactly **one start** to its local start hour/month; do
not split its duration into demand counts.

The source reports stay immutable. Cross-archive conflicts are quarantined in the
new canonical rejection audit; they are not silently merged, assigned an arbitrary
survivor, or added twice. Counts can change when new archives reveal conflicts,
so each reconciliation publishes an immutable snapshot rather than rewriting a
previous one. Invalid raw records also participate in duplicate-ID checks; a valid
row cannot escape a conflict simply because its other occurrence was invalid.
Original records remain available in raw files and source rejection audits.

Canonical Parquet reconstructs `timestamp[ns, tz=UTC]` from exact epoch-nanosecond
columns. This avoids DuckDB's microsecond TIMESTAMPTZ conversion losing submicrosecond
precision. Hourly grouping uses UTC instants, so repeated NYC fall-DST clock hours
remain separate elapsed-hour keys. The timezone assumption remains the reviewed
Phase 1 contract: naive timestamps mean NYC local time; ambiguous/nonexistent naive
DST times are rejected. The official page does not explicitly confirm this timezone
assumption; do not claim publisher confirmation or guess offsets.

The January 2026 bounded probe parsed **53,277 rows** from the first 5 MiB of each
CSV and found **14 December 2025 starts** (8 + 6). One example:

```text
ride_id:    7F9EFD979D383745
started_at: 2025-12-31 23:49:15.165
ended_at:   2026-01-01 00:04:53.730
source:     202601-citibike-tripdata_1.csv
```

**Include January 2026 as boundary input** before claiming complete December start
coverage. These are sample counts, not a full-month total or validated December
forecast target. Cross-check with December IDs during global reconciliation. Store
January 2026 starts canonically, but exclude them from `model_window_hourly_rides`,
which uses the fixed modelling window. No complete January 2026 archive was
downloaded in the pilot.

One next-month archive is a practical boundary input, **not a proven completeness
bound**: there is no upper duration cutoff, and delayed publication/corrections can
occur. Review later archives for earlier starts before declaring the final dataset
complete. Similarly, January 2024 still needs February 2024 boundary review, and
March 2025 needs April 2025. Missing archives and boundary-input presence are
explicitly reported; pilot partitions are labelled partial history.

## Files and commands

Run from the project root. Existing dependencies are sufficient; no download SDK
or cloud services were added. Source modules are `archives.py` (transport),
`historical.py` (orchestration), `history_catalog.py` (global audit/partitioning),
and `historical_cli.py` (CLI). Offline synthetic tests cover these behaviours.

```bash
# Metadata-only storage inspection; no monthly archive downloads.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --start 2024-01-01 --end 2026-01-01 --plan

# Small remote boundary probe; does not register a full source partition.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month 2026-01 --probe-boundary

# Controlled samples; this retains complete ZIPs and raw CSVs, but processes
# only the first 1,000 rows PER CSV. Already verified full partitions skip.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month 2024-01 2025-03 --sample-rows-per-file 1000

# Register February once by reference; never overwrite its outputs.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --register-existing 2025-02=outputs/phase13-february-reviewed

# Complete pilot partitions; reuse verified downloads/extractions.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month 2024-01 2025-03 --chunk-size 50000 --memory-limit 512MB --reconcile

# Retry the same ingestion command to resume or skip verified partitions.
# Reconcile alone publishes a NEW canonical snapshot, retaining prior ones.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli --reconcile

.venv/bin/python -m pytest -q
```

Local layout (raw/generated content is ignored by Git):

```text
data/raw/citibike/
  archives/                               # ZIPs, download receipts, .part files
  source_month=YYYY-MM/                    # Retained CSVs and extraction receipt
outputs/historical/
  manifest.json                           # Per-source state, attempts, hashes, reports
  samples/YYYY-MM-n1000/                   # Separate sample outputs
  source_month=YYYY-MM/                    # Original ETL outputs + verification/metrics
  boundary_probe_2026-01.json              # Bounded investigation, not full ingestion
  storage_plan_24_months.json              # Metadata-only planning evidence
  canonical/CURRENT.json                  # Points to last successfully published snapshot
  canonical/snapshots/<id>/
    catalog.duckdb                         # Global audit views and hourly table
    trips/start_month=YYYY-MM/             # UTC-ns trips with archive provenance
    hourly/start_month=YYYY-MM/             # Observed station-hour counts, no zero filling
    canonical_rejections.parquet           # Combined local + global rejection audit
    duplicate_ids.parquet                  # Duplicate IDs and archive/occurrence counts
    report.json                            # Global and month-level counts, missing-source flags
```

Read `canonical/CURRENT.json` to find the current `catalog.duckdb`. SQL example:

```sql
SELECT start_month, SUM(ride_count) AS validated_ride_starts
FROM model_window_hourly_rides
GROUP BY start_month
ORDER BY start_month;
```

## Completing the remaining ingestion later

Do **not** run the full-range commands until storage is expanded. The 24 ZIPs total
**16.387 GiB**, and their CSVs total **16.387 GiB**. The conservative plan requires
**71.318 GiB additional capacity plus 5 GiB reserve**, before adding January 2026.
This budgets retained files, outputs, and staging; it is not a measured peak.
Allow about **80 GiB free or more**, preferably on a larger dedicated volume, and
rerun the plan for the actual configured roots. There is no full-history download
in this task.

With sufficient storage on the existing roots, the range command skips the three
verified sources and processes the remaining 21 modelling archives:

```bash
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --start 2024-01-01 --end 2026-01-01 --chunk-size 50000 --memory-limit 512MB

# Separately sample, then ingest the boundary source. Do not extend --end.
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month 2026-01 --sample-rows-per-file 1000
PYTHONPATH=src .venv/bin/python -m citypulse_ai.etl.historical_cli \
  --month 2026-01 --reconcile
```

Use `--raw-root /path/to/larger-volume/data/raw/citibike` and
`--output-root /path/to/larger-volume/outputs/historical` to configure a larger
volume. For a **new** output root, register the January/March source partitions and
reviewed February output once using `--register-existing MONTH=PATH ...`; this
references them without deleting, relocating, or copying existing data. Each
root has its own manifest; changing only the root does not migrate a manifest.
Do not reduce the safety reserve to force downloads onto a nearly full disk.

Before final feature preparation: review all monthly rejection/DST audits, verify
source and boundary coverage, investigate duplicate conflicts without guessing a
survivor, audit source changes, and document station availability. Only then select
100 stable stations from the fixed training period and prepare 24-elapsed-hour
forecast samples. Keep test-period data out of cohort selection and learned rules.

## Actual pilot verification

Both full pilot ETL runs completed. Phase 2.2C then verified the existing processed
January/February/March partitions read-only, without rerunning raw ETL or publishing
canonical snapshots. Reported input totals came from the existing ETL reports;
accepted Parquet rows and rejected audit rows were counted during verification.

| Source archive | Reported input | Verified accepted | Verified rejected |
| --- | ---: | ---: | ---: |
| January 2024 | 1,888,085 | 1,886,925 | 1,160 |
| February 2025 | 2,031,257 | 2,030,542 | 715 |
| March 2025 | 3,168,271 | 3,167,143 | 1,128 |
| Total | 7,087,613 | 7,084,610 | 3,003 |

The explicit 11-column union executed successfully. All nine source-file counts
reconciled. There were **7,084,610 unique accepted ride IDs**, no within/archive or
cross-archive duplicates (including rejection-audit IDs), and no required-value,
provenance, or UTC epoch mismatches. Fractional seconds were preserved; no actual
pilot timestamps required submicrosecond precision, and exact nanosecond epoch
integers remain available. Original artifact sizes and modification times stayed
unchanged. February's original ZIP hash remains unavailable; recorded CSV hashes
remain available. No source hashes were recomputed during this verification.

| Source archive | Previous local start month | Accepted previous-month starts | Same-month starts |
| --- | --- | ---: | ---: |
| January 2024 | December 2023 | 410 | 1,886,515 |
| February 2025 | January 2025 | 273 | 2,030,269 |
| March 2025 | February 2025 | 410 | 3,166,733 |

Verification took **13.02 seconds** with a 512 MB DuckDB limit, approximately
625 MiB process peak RAM, and a 71 MiB temporary identifier ledger. It scanned
accepted source Parquet once, retained no full trip table, and produced no
canonical snapshots. These counts do **not** prove complete calendar-month
coverage; adjacent-source and final historical coverage reviews remain necessary.

## Phase 2.2B — explicit catalogue schema alignment

The pilot integration failure was caused by **directory inference, not different
physical trip schemas**. January and March files live under `source_month=...`;
DuckDB's automatic Hive partition discovery added a virtual `source_month` column.
February's reviewed directory has no such key. All three physical Parquet schemas
have 10 columns, but the old `SELECT *, <month> AS source_archive_month` returned
12 columns for January/March and 11 for February, making positional `UNION ALL`
fail. Earlier synthetic tests used ordinary directories and missed this layout.

The catalogue now reads Parquet with `hive_partitioning=false`, inspects footer
metadata, and projects the same named columns and SQL types in the same order:

```text
source_file VARCHAR, source_row BIGINT, ride_id VARCHAR,
started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ,
start_station_id VARCHAR, end_station_id VARCHAR,
duration_seconds DOUBLE,
started_at_utc_ns BIGINT, ended_at_utc_ns BIGINT,
source_archive_month VARCHAR
```

`source_archive_month` comes from the registered manifest, not inferred folder
columns. `start_month` remains separately derived from the NYC-local start instant.
Original source-file/row provenance, exact string identifiers, and epoch-nanosecond
integers are preserved. The existing bounded Parquet writer reconstructs UTC
nanosecond timestamps from those exact integers.

Only `end_station_id` and `duration_seconds` are optional catalogue trip metadata:
missing fields become **typed NULL**, never an invented end station or duration.
Required identity/provenance/time fields must exist and have compatible types.
During catalogue execution, missing required values, inconsistent UTC timestamp
and epoch pairs, or nonpositive elapsed durations fail before snapshot publication.
The rejection audit also accepts a missing optional end-station column without
losing duplicate detection or original rejection reasons. These checks do not
change Phase 1 ETL rules or rewrite source partitions.

Column order and extra physical columns no longer change the union schema. Extra
fields remain in the immutable source files; the new `source_schema_alignment`
report lists them explicitly, along with missing optional columns and the disabled
Hive setting. Do not use permissive union-by-name to turn absent required fields
into unexplained nulls.

Verification in this phase: **22 focused catalogue tests passed**, and **112 tests
plus 6 subtests passed** in the complete suite. New tiny fixtures cover actual
Hive/non-Hive paths, reordered/extra fields, missing/null optional fields, required
schema/value errors, exact timestamp precision, identifier preservation, provenance,
boundary counts, and cross-partition duplicates including rejected records.
A metadata-only `DESCRIBE` of the aligned union across all three real Parquet files
bound successfully with the expected 11 columns. No real trip rows were scanned,
no raw ETL or downloads ran, and no historical catalogue snapshot was published.
At the end of Phase 2.2B, real cross-archive counts and boundary reconciliation
were still unverified. The subsequent Phase 2.2C results are recorded in
[Actual pilot verification](#actual-pilot-verification) above.
