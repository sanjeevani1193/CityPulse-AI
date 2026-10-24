# Final 100-station TFT dataset

`features/final_station_dataset.py` prepares the fixed January 2024–December 2025
accepted-recorded-starts dataset. It does not train or change the TFT. Run the
full build on the existing safeguarded Linux VM; do not restore historical
inputs to the Mac just to use this command.

## Selection and labels

The existing selector receives only January 2024–June 2025 daily station activity.
Each station must have at least 15 active local dates in every training month,
with no inactivity run longer than 14 local dates, including edge gaps. Rank by
accepted training starts descending, then exact station ID ascending. Freeze
exactly 100 stations and codes 1–100. Insufficient eligibility fails; future
survival, validation demand and test demand never affect selection. These are
activity proxies, not station operating-status evidence. Selection metadata
explicitly identifies this full-period final cohort rather than the prototype.

Before selection, the immutable history reader verifies all target months and
adjacent sources, the completed global duplicate audit, source inventories and
the explicit February CSV-only certificate. Each bounded quarter is certified
by the existing `AcceptedRecordedStarts.review`, including an exact accepted-row
recount against canonical station-hour aggregates. The policy remains
`accepted_recorded_starts_v1`. A zero means no accepted recorded starts in this
verified source set; it says nothing about lost demand or station closure.
Unverified coverage fails closed. Operating availability remains unknown unless
separately evidenced. Rejected-record diagnostics stay in quarter audit JSON,
never future-known model inputs.

## Storage and chronological windows

Eight quarterly Parquet panels store unique station hours. A quarter contains at
most 220,900 rows for 100 stations. Checkpoints bind each completed panel and label
evidence to the build identity and hashes. A partial quarter is retained and
requires inspection; it is never silently overwritten. A completed rerun verifies
artifact hashes and reuses the same dataset. Source manifests and canonical
snapshots are never rewritten.

Splits are NYC-local, end exclusive:

- Training: 2024-01-01 through 2025-07-01.
- Validation: 2025-07-01 through 2025-10-01.
- Test: 2025-10-01 through 2026-01-01.

`windows.parquet` stores station/origin keys, split, encoder/target index ranges
and input/target eligibility. `excluded-origins.parquet` records structural
warm-up/boundary exclusions once per origin. Origins are NYC-local midnights;
encoders are 336 elapsed UTC hours and targets are exactly 24 elapsed hours.
Spring horizons extend into the following local date, and fall horizons can end
before the next local midnight. Split-crossing horizons are excluded. History
from a preceding split is permitted; no targets enter encoder features.

Preprocessing is fitted one station at a time on its unique trusted **training**
hours, including certified zeros. It uses the original population mean/standard
deviation (constant scale fallback 1), preserving the saved format consumed by
`forecasting.tft_preparation.load_preprocessing`. No torch dependency is needed
on the ETL VM. Future calendar variables use the existing eleven-feature schema;
there are no fitted calendar statistics or weather inputs.

```python
from citypulse_ai.features.final_station_dataset import LazyStationWindows
from citypulse_ai.forecasting.tft_preparation import windows_to_batch

samples = LazyStationWindows("outputs/tft-final/station100-20261024b", "train")
window = samples[0]  # reads only this station's 360-hour interval
batch = windows_to_batch([window], samples.scalers)
# Model inputs: station (B,1), encoder-known (B,336,11),
# encoder-observed (B,336,5), decoder-known (B,24,11).
# Loss-only target/mask: (B,24,1). Original model quantiles: (B,24,3).
```

The lazy reader checks all artifact hashes once on open. Keep the directory
immutable; call `verify()` again after transfer or suspected changes. It holds
window indices, not all training tensors. An ordinary PyTorch DataLoader can use
its `__len__`/`__getitem__` interface with a collator calling `windows_to_batch`.

## Build and recovery

```bash
PYTHONPATH=src .venv/bin/python scripts/build_final_station_dataset.py \
  --audit /verified/completed-audit.json \
  --manifest /verified/manifest.json \
  --february-certificate /verified/source-inventory.json \
  --certificate-sha256 TRUSTED_CERTIFICATE_SHA256 \
  --output-dir outputs/tft-final/station100-20261024b
```

Use the existing runtime monitor around this command: 512 MB DuckDB, 2 GiB RSS,
normal host pressure, zero swap, bounded workspace and at least 15 GiB free disk.
Keep the four-hour server-side VM stop and trial-only billing. The plain CLI is
also useful for synthetic/local development; it does not provision resources or
alter the machine's runtime policy itself.

The cloud run publishes a create-only recovery ZIP containing panels, indices,
preprocessing, selection, source/audit copies, evidence, code and runtime reports.
It verifies object generations and full downloaded SHA-256, safely extracts the
bundle to a new VM directory, and opens the restored lazy reader. The manifest
and publication receipt are separately published. Original historical recovery
objects remain necessary to independently re-audit labels; the dataset bundle
is sufficient to load prepared training/evaluation samples. Only small reports
return to the Mac. No historical input or model artifact is deleted.

This is a retrospective benchmark with eventual accepted counts assumed available
at bucket end. It is not an operational publication-latency backtest. Unknown
station operation, publisher omissions and unmet physical demand are not resolved
by complete accepted-record label masks. Final model fitting, validation-based
choices, fair paired model evaluation and test scoring are separate later tasks.
