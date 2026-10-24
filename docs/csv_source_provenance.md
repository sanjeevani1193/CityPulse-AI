# CSV-only historical provenance supplements

`features/csv_source_inventory.py` supports legacy imports whose **original CSVs**
were backed up but whose original ZIP was not retained. It does not reconstruct
or claim to verify a ZIP, modify historical manifests, rerun ETL, or change
`accepted_recorded_starts_v1`. It supplies missing evidence about the frozen
original CSV source set already declared by the historical source-backup system.

A certificate requires all original CSVs, exact original filenames and SHA-256
hashes, verified generation-pinned source receipts, matching byte sizes, streamed
CSV headers/record widths, and original per-file row counts. All required raw
columns must exist, with no empty or duplicate header names. Values are not cast
or revalidated; exact timestamp and station strings stay in the original files.

A 512 MB DuckDB connection with spill disabled verifies that the immutable
accepted/rejected outputs have their original counts. For each original source
path, their row ordinals must jointly cover every CSV data record exactly once:
count = distinct count = original row count, with range 2 through N+1. This
checks source membership, not new ride acceptance criteria. Hashes are checked
before/after the reads. Existing historical duplicate, boundary, timestamp and
label checks still apply.

The separate JSON certificate binds the original source entry, current frozen
manifest hash, canonical source identity, each CSV hash/schema/count and backup
generation. It explicitly records `original_zip_available=false` and
`zip_inventory_verified=false`. Its scope is the enumerated archived CSV set;
publisher completeness, physical demand and operating availability are not
established. No ZIP checksum, central-directory member or CRC is invented.

The certificate is opt-in and must be supplied with a separately trusted SHA-256,
for example a verified GCS publication receipt. A certificate with changed source
identity, manifest, membership, schemas, row coverage or generations is rejected.
It cannot replace an existing complete archive inventory. Absence of an optional
certificate preserves the previous fail-closed behavior.

```python
history = StationHistory(audit_path, manifest_path,
    csv_inventories={"2025-02": (certificate_path, trusted_sha256)})
# No station selection, cohort or labels are created by this readiness check.
evidence = history.verify_source_inventories(["2025-02"])
```

The accepted-record label adapter accepts verified `source_member_inventory_verified`
for explicit CSV certificates, preserving the original archive checks and native
evidence identities otherwise. `archive_member_inventory_verified`
remains **false** for a CSV-only source, and the certificate digest is recorded
in label provenance. The old positive-only policy keeps its conservative masks;
the accepted-record policy keeps its existing positive/zero decisions and
independent unknown-operation masks. A source certificate does not bypass the
per-station/hour recount, finalized boundary coverage or cohort-membership rules.

Recovery requires the original canonical recovery bundle and original processed
objects, this certificate and its trusted checksum/publication receipt, and the
supporting code. The original CSV objects remain at their existing pinned GCS
locations; do not duplicate them in a new recovery archive. Certificate recovery
is verified against the same frozen manifest/audit; changed source provenance
requires a new reviewed certificate, never an edited historical manifest.

Run the focused synthetic tests without real source data or cloud access:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_csv_source_inventory.py \
  tests/test_station_history.py tests/test_station_labels.py \
  tests/test_accepted_recorded_labels.py tests/test_accepted_recorded_probe.py \
  tests/test_cloud_retention.py tests/test_vm_recovery.py
```

The real February repair restores only the three pinned CSVs to the existing
Linux VM, retains the 512 MB DuckDB and 2 GiB RSS safeguards, publishes small
certificate/readiness/code recovery artifacts, verifies downloaded restoration,
and stops the VM. It does not select stations or build/train the final TFT.
